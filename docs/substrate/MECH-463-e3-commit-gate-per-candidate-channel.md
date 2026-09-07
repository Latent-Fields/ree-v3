## MECH-463: E3 commit-gate + per-candidate channel-term diagnostics (arousal-conditioned variance decomposition instrumentation) (2026-07-18)

Diagnostics-only extension of the existing V3-EXQ-571 `e3_score_decomp_enabled` flag
(`ree_core/predictors/e3_selector.py`; default False). No new config params, no new SD,
no behaviour path touched. Added so MECH-463 -- "the three global-scalar affective routes
(D1/D2 gain, harm-urgency threshold shrinkage, LC-NE temperature) are a channel-agnostic
VARIANCE AMPLIFIER of the already-dominant selection channel, not a source of behavioural
differentiation" -- becomes measurable. Per the 2026-07-18 instrumentation audit it was
NOT computable from existing runs.

Three additions, all inside `if self.e3_score_decomp_enabled:`:
1. Commit-gate scalars into `last_score_diagnostics`: `urgency_applied`,
   `effective_threshold`, `commit_variance`, plus `commit_gate_mode`
   ("harm_score_variance" | "world_variance") and `committed`. These were pure locals
   that died at end of `select()`; `urgency_applied` escaped only as
   `SelectionResult.urgency`. `commit_variance` is the quantity the gate ACTUALLY
   compared against `effective_threshold` under either commit mode.
2. `self.last_channel_terms`: the per-candidate [K] channel-bias tensors (`_lcg_terms`)
   retained UNREDUCED, keyed by channel name. This is what makes the decomposition
   covariance-correct -- `agent.py:6816-6845` keeps only marginal scalars, losing
   channel-F covariance. Detached clones (the tensors stay live in the recompose /
   eligibility paths).
3. Confirmed `last_score_decomp["per_candidate"][i]["f"]` already isolates the
   per-candidate F component once the flag is on (verified: K-length vector).

Backward compatible: VERIFIED bit-identical with the flag OFF -- seeded 3-seed x 60-step
behavioural-trace digests (actions, final E3 scores, commit flags) match pristine HEAD
exactly. NOTE when writing such a check: `CausalGridWorld` carries its OWN
`np.random.default_rng(seed)`, so omitting its `seed=` kwarg makes the trace
non-reproducible run-to-run and any bit-identity comparison meaningless.

MECH-463 NON-VACUITY GATE -- validated runnable config (inherited from V3-EXQ-643a).
The probe is vacuous, and returns a spurious FLAT F-share (a FALSE REFUTES), unless all
three hold. Measured GREEN at 400 ticks with:
  - `urgency_weight=0.12` + `use_affective_harm_stream=True` -> `urgency_applied`
    non-constant (45 distinct, 0.036-0.128, no saturation). At `urgency_weight=0.5` it
    SATURATES against `urgency_max` (median == p90 == 0.5) and the top deciles collapse.
  - `use_e3_score_diversity` + `use_e3_diversity_entropy_bonus` (MECH-341) -> the only
    channel that genuinely carries cross-candidate range (mech341 range ~0.465).
    Curiosity/tonic-vigor biases alone are UNIFORM across candidates
    (`score_bias_abs_mean=0.025`, `score_bias_range_mean=0.0`), so
    `modulatory_authority_active` can never fire and every non-F channel contributes
    zero variance -> F-share trivially 100% at every decile.
  - SD-056 `e2_action_contrastive_enabled` + `e2_rollout_output_norm_clamp_enabled`
    keeps E3 scores bounded (raw range ~0.034, not the ~1e32 that killed V3-EXQ-643 by
    float32 catastrophic cancellation).
DRIVER CONSTRAINT: `act_with_split_obs()` calls `sense(obs_body, obs_world)` with NO
`obs_harm_a`, so `z_harm_a` is None on that path and `urgency_applied` is pinned at 0.
A MECH-463 driver MUST feed the harm stream explicitly via
`sense(..., obs_harm=..., obs_harm_a=..., obs_harm_history=...)` and then replicate the
wrapper (`clock.advance()` -> `_e1_tick` -> `generate_trajectories` -> `select_action`).

Validation experiment: V3-EXQ-785 (queued). See MECH-463 in
REE_assembly/docs/claims/claims.yaml, MECH-439 (the F-dominance variance monopoly this
claim says arousal amplifies rather than breaks), MECH-359, MECH-390, SD-011.

- SD-MECH457-POLICY-KL-ANCHOR: mech457 policy trust-region anchor -- IMPLEMENTED 2026-07-19.
  New `PolicyKLAnchor` in experiments/_lib/mech457_explorer_classes.py: a KL penalty pinning the
  policy to a FROZEN DEEP-COPY SNAPSHOT of itself taken at entry to `train_a2c`. Because callers
  install the BC prior BEFORE that call (baselines/mech457_retention.install_bc_prior ->
  train_off_arm), "at entry" IS the post-install checkpoint -- which is why the snapshot needs no
  explicit checkpoint argument.
  Config: BootstrapExplorerConfig.use_policy_kl_anchor (default False) + .kl_anchor_coef (default
  0.0), BOTH declared in as_slice() so the knobs land in the arm fingerprint config_slice; also
  threaded through baselines/mech457_retention.reference_config(). Applied inside
  train_bootstrap_explorer (an UPDATE-rule knob), NOT at rep construction like the critic swap.
  Data flow: rep.z_detached(state) -- on BOTH reps precisely the tensor ActorCriticPolicy.select()
  consumes -> frozen snapshot forward under no_grad -> ref logits -> coef * KL(pi || pi_ref),
  gradient through the LIVE logits only -> added to the episode loss AND to the credit-replay
  loss. Direction per the substrate_queue implementation_hint.
  Backward compatible: disabled by default; contract K1 asserts default-OFF trains BIT-IDENTICAL
  weights, verified end-to-end on the raw_view retention path (BC-install -> RL refine).
  ANTI-ALIAS (both load-bearing, both structural rather than conventional):
    (1) vs mech457_bc_aux_schedule / H-retention-auxiliary-decay -- anchors to the INSTALLED
        POLICY, never the demonstrator. The class never sees bc_demo and works with bc_demo=None
        (contract K4). Anchoring via bc_aux_coef would anchor to the demonstrator and alias.
    (2) vs mech457_distributional_critic / H-retention-critic -- the KL term is a function of
        `logits` alone and puts EXACTLY ZERO gradient on value_head/value_bins (contract K3); the
        fan.critic_value_loss dispatch is byte-identical on both branches. HONEST LIMIT: the
        trunk is shared, so the critic's INPUT FEATURES do move. That is the exact mirror of the
        sibling build's own situation (its CE loss moves the trunk too), whose contract C2
        asserted only that the CE loss puts no gradient on the policy HEAD.
  CREDIT-REPLAY IS ANCHORED TOO, and this is scientific rather than stylistic: the reference
  retention build runs credit_replay=True, so `_prioritized_credit_replay` applies a SECOND
  policy-gradient update per episode. Anchoring only the main loss would leave the constraint
  leaky and make a null from this leg unreadable -- "anchoring does not preserve competence" and
  "the unanchored replay update drifted the policy anyway" would be indistinguishable. The
  penalty sits INSIDE the CREDIT_LR_SCALE parenthesis so the constraint scales with the update it
  constrains (contract K9).
  THE PENALTY IS STATIONARY AT ZERO DRIFT, not merely small: KL(pi || pi) = 0 is the minimum, so
  its gradient vanishes there and an anchored arm is unpenalised until it starts to move
  (contract K6). Discovered by contract K3 failing its own non-degeneracy check on first run --
  a gradient probe taken AT the snapshot point passes vacuously, so K3 perturbs the policy first.
  Guard dict gains policy_kl_anchor_installed / policy_kl_anchor_coef /
  mean_policy_kl_to_anchor_recent, emitted unconditionally. The MEASURED KL is what lets a
  manifest verify the anchor actually BOUND rather than assuming it (same reasoning as
  bc_aux_coef_first/_last): a ~0 realised KL on an arm labelled anchored means the policy never
  tried to leave the snapshot, which is a different reading from a retention null.
  Mis-wiring RAISES (contract K7): switch without weight, weight without switch, non-positive
  coefficient, or rep.policy() returning None -- each would otherwise yield an arm that IS the
  control while labelled the treatment.
  Z_WORLD COTRAIN CAVEAT (documented, not defended against): under cotrain_encoder=True the
  encoder moves, so pi_ref is evaluated on a drifting input and the anchor is only approximate.
  The retention reference build is raw_view with cotrain_encoder=False, on which it is exact.
  NOT BUILT, deliberately: adaptive coefficient control / a target_kl trust-region radius. A
  second moving part would confound the leg's single declared intervention.
  Motivation (measured): V3-EXQ-780 raw_view post-BC competence 20.933 eroded to 11.667 under
  unconstrained RL refinement, 3/3 seeds having taken the install.
  ML statement (engineering counsel only): KL-penalty trust region (Schulman 2015/2017 TRPO/PPO);
  the biology is the consolidation/protection pathway the portfolio names, not the ML framing.
  Phased training required: no (no encoder head on a moving latent target). MECH-094: not
  applicable (no memory writes).
  Validation experiment: NOT QUEUED -- queueing the leg is governed by GOV-FANOUT-1 and routed
  through /queue-experiment. Unblocks hypothesis H-retention-consolidation (competence_floor
  question, evidence/planning/hypothesis_space_registry.v1.json), completing the three-leg
  retention portfolio alongside V3-EXQ-788 / V3-EXQ-789. MECH-457 stays candidate/v3_pending;
  this build promotes and demotes nothing.
  11 new contracts: tests/contracts/test_mech457_policy_kl_anchor.py.
  See REE_assembly/evidence/planning/mech457_retention_portfolio_2026-07-18.md and the
  substrate_queue node mech457_policy_kl_anchor.

- arm-fingerprint-executed-substrate-identity -- arm_fingerprint records the EXECUTED
  substrate, not the on-disk substrate -- IMPLEMENTED 2026-07-20.
  experiments/_lib/arm_fingerprint.py, experiments/_lib/manifest_core.py,
  experiments/_lib/arm_reuse.py. INSTRUMENTATION ONLY -- no ree_core change, no config
  param, no behavioural effect on any experiment; nothing here can move a metric.
  DEFECT: compute_substrate_hash reads source FROM DISK at cell entry, while the cell
  executes in-memory bytecode frozen in sys.modules at first import. When the checkout
  moves mid-run -- routine on a fleet that pulls continuously -- the manifest records the
  DISK state, not the EXECUTED state. Worked instance V3-EXQ-778a: 6 of 8 cells stamped
  c8d6d0e2 while provably executing e9a22a91 (commit da873a1 landed 85s into a 418s run,
  editing a _lib harness the driver had already bound at module scope), all 8 carrying
  reuse_eligible: true. That is a FALSE-HIT channel -- the one failure mode
  arm_fingerprint.py:20-27's governing asymmetry says must never occur, since
  over-inclusion was designed to buy false MISSES only.
  SCOPE OF THE CLAIM (the sweep headline was corrected hours after it was written):
  intra_run_substrate_divergence_sweep_2026-07-20.md's "42 of 164 (25.6%)" counts runs
  whose RECORDED HASH VALUE changed -- NOT runs that lost experimental control, and it
  must not be cited as the latter (failure_autopsy_V3-EXQ-782 + the sweep author's own
  07:30Z WORKSPACE_STATE correction). _SUBSTRATE_GLOBS is uniformly wider than what any
  run imports, so an unrelated parallel edit moves the hash without touching the run; on
  782 the closure-restricted hash was byte-identical across all four bands. 778a's own
  confound was likewise REFUTED -- all 8 seeds provably executed one build. What STANDS,
  and what this lands, is the INSTRUMENT defect that autopsy routed here: the recorded
  identity was never guaranteed to be the executed identity.
  FIX, three parts:
  (1) resolve_substrate_identity() memoises the substrate hash for the process lifetime,
      keyed by (repo_root, declared scope, extra paths); compute_arm_fingerprint serves
      every cell from that one snapshot. The first resolution happens at the first cell,
      after the driver's module-scope imports have frozen the executed bytecode and
      before any mid-run move can be observed -- so recorded identity IS executed
      identity by construction. driver_script_hash is frozen the same way.
  (2) substrate_stability_report() re-hashes from disk at stamp time;
      stamp_recording_core writes top-level substrate_stable_across_run (+ a
      substrate_stability_detail block when False). Two independent tests, either of
      which can only prove INSTABILITY: per-cell hash cardinality > 1, or process
      snapshot != disk. Catches the one residual the freeze cannot -- a module imported
      LAZILY after the first cell. NOT added to ALWAYS_CORE_KEYS: the pre-2026-07-20
      corpus cannot carry it and making it core would WARN on every legacy manifest.
  (3) arm_reuse.source_run_substrate_unstable() + REFUSE_SUBSTRATE_UNSTABLE: the reuse
      consumer refuses to serve any cell out of a run whose cells disagree about their
      substrate. This is the RETROACTIVE handling of the 42 known-divergent runs -- their
      manifests are NOT edited (completed runs are re-adjudicated by autopsy, never
      rewritten), so the guard lives in the only path that could ever act on them. It is
      computed from the manifest at lookup time, so a stale index cannot bypass it.
      Absence of substrate_stable_across_run is NOT read as instability (that would
      refuse every banked mint); it falls through to the cardinality test.
  NOT a hard cut like torch-in-machine_class: on a stable run the emitted hash is
  byte-identical to before, so every banked fingerprint still matches. The new payload
  fields (substrate_identity_source, substrate_identity_resolved_at) are observability
  only and deliberately outside fp_input.
  KNOWN OVER-REFUSAL, and the 782 caution it answers: cardinality > 1 over-reports, via
  driver_script_in_substrate_hash being toggled mid-run (one corpus run, V3-EXQ-788) and
  more importantly via the over-wide globs above. failure_autopsy_V3-EXQ-782 warns that a
  whole-glob divergence flag "would institutionalise this false positive as a standing
  warning" and asks for a closure-restricted check. That caution governs a signal that
  ADJUDICATES SCIENCE, where a false positive wrongly impeaches a real result. The reuse
  gate is the opposite asymmetry -- a false MISS costs compute, a false HIT corrupts a
  conclusion -- so over-refusal is correct HERE precisely because it is wrong THERE; the
  two must not be collapsed. substrate_stable_across_run is therefore scoped as an
  INSTRUMENT event, never a confound verdict, and nothing adjudicates a claim off it.
  When the closure-restricted recomputation lands (782's secondary routing), it should
  narrow BOTH -- recovering the wrongly-refused runs at no cost to safety.
  Also corrected: _hoist_multi_arm_substrate_hash's docstring asserted "all arms of one
  run execute against the same substrate", which the sweep falsified; the hoist kept only
  the first hash and so actively HID divergence at the top level. Behaviour unchanged
  (back-compat), premise documented, per-cell set now exposed via
  multi_arm_substrate_hashes().
  Phased training: N/A. MECH-094: N/A (no memory writes).
  Validation: no EXQ -- this changes no mechanism and gates no claim; it is verified by
  17 new contracts in tests/contracts/test_arm_fingerprint_executed_substrate.py, which
  encode the failure record's acceptance target directly (recorded identity == executed
  identity for 100% of cells, OR the run is stamped substrate_stable_across_run: false).
  Source: REE_assembly/evidence/planning/failure_autopsy_V3-EXQ-778a_2026-07-20.json
  targets[0].recommended_substrate_queue_entry, and
  REE_assembly/evidence/planning/intra_run_substrate_divergence_sweep_2026-07-20.md.

- dose_saturation lint -- IMPLEMENTED 2026-07-22. experiments/_lib/dose_saturation.py,
  stamped from manifest_core.stamp_recording_core beside stamp_inert_arm_knob.
  CATCHES: two DECLARED DOSE LEVELS whose per_level float readouts are equal beyond
  float noise. V3-EXQ-794 ran SD-076's asymmetry at LO=0.6 and HI=0.8 and got
  overconfidence_score = -1.004111904519277 at BOTH, plus calibration_ratio =
  2.7564936387545953 at both, because rv was clamped at a floor above the operating
  point. A genuine dose-response -- INCLUDING A GENUINELY NULL ONE -- gives different
  values at different doses with seed-level variance; agreement to the last bit means the
  quantity saturated before the dose could express itself.
  COST OF NOT HAVING IT: SD-076 was recorded does_not_support, charging a refutation to a
  claim whose lever never moved, and MECH-204's correction was left with no drift to
  correct. Both claims went untested while appearing tested, and it took a full autopsy
  to withdraw the direction.
  SIBLING, NOT DUPLICATE, of inert_arm_knob (c040d28): there the knob never reached a
  live code path so the arms RAN IDENTICALLY; here the knob DID move the dynamics and a
  bound downstream erased the difference. 794's arms are not bit-identical cell-wide, so
  inert_arm_knob does not and should not fire on them.
  Emits dose_levels_separable (bool) + dose_saturation_detail (offenders only, on the
  False verdict). NOT in ALWAYS_CORE_KEYS -- the pre-2026-07-22 corpus cannot carry it.
  POSTURE: record-and-WARN at write, gate at adjudication, same as its sibling -- by
  manifest-write time the compute is spent, and 794's green arms stayed scorable. The
  autopsy's "REFUSE the dose-response criterion" is honoured by emitting the flag for the
  experiment's own scoring to read.
  FALSE-POSITIVE DISCIPLINE (all pinned by contracts): tied INTEGERS never fire
  (n_seeds_overconfident = 0 at both levels is how a count says "no effect"); tied
  strings/bools never fire; exact 0.0/0.0 ties are recorded under zero_ties but do NOT
  flip the verdict (zero is overwhelmingly a not-applicable sentinel). Only a tie between
  two NON-ZERO floats fires -- which different trajectories do not produce by chance.
  Dose identification excludes EVERY fully-varying numeric key, not one: such a key can
  never appear in tied_fields anyway, so the exclusion is lossless and cannot manufacture
  the identity it reports (the reason inert_arm_knob had to reject its analogous
  inference). manifest["dose_key"] declares it explicitly.
  Validation: no EXQ -- it adjudicates nothing and gates no claim; verified by 22
  contracts in tests/contracts/test_dose_saturation_lint.py, including a replay of the
  real 794 per_level block to a firing verdict.
  Source: REE_assembly/evidence/planning/failure_autopsy_V3-EXQ-794_2026-07-22.md sec 6
  item 2.

- SD-077: goal.common_mode_invariant_super_ordinal_cue_key — IMPLEMENTED 2026-07-21.
  SuperOrdinalGoalMemory (ree_core/goal.py) — the MECH-189 super-ordinal goal-anchor store.
  Config: GoalConfig.super_ordinal_cue_centering (default False = bit-identical OFF; set
  True to enable) and GoalConfig.super_ordinal_cue_baseline_alpha (default 0.02, matching
  SD-066). Both plumbed through REEConfig.from_dims.
  Data flow: z_world -> observe() slow-EMA common-mode baseline -> centered residual
  (z_world - baseline) -> _best_match cosine -> contextual_complexity / retrieve.
  Backward compatible: disabled by default; observe() returns immediately, _centered() is
  the identity, and no baseline tensor is allocated, so existing experiments are unaffected.
  WHY. The store keyed anchors on RAW z_world cosine. Under SD-008 z_world
  under-differentiation the untrained encoder maps every context into a common-mode cone,
  so that cosine measures the shared offset rather than the context. Measured 2026-07-21
  on the actual V3-EXQ-669b Stage-0 forced-feed nursery (155 contexts, seed 101): raw
  world_obs pairwise cosine min 0.216 / mean 0.608 (90.7% of pairs below 0.8), but z_world
  pairwise cosine min 0.9641 / mean 0.9898 (ZERO pairs below 0.8), with
  ||mean(z_world)|| / mean||z_world|| = 0.9949. The nursery IS context-diverse; the encoder
  buries it. Result: 1 allocation + 159 reinforcements into slot 0, anchor_count == 1, and
  669b self-routed on its own R3 readiness gate with C1/C3 unmeasurable.
  THRESHOLD TUNING CANNOT SUBSTITUTE, and this is provable rather than observed:
  contextual_complexity = 1 - best_cosine and best_cosine >= 0.9641 everywhere, so
  complexity <= 0.036 under ANY (merge_similarity, complexity_threshold) pair -- strictly
  below 669b's pre-registered COMPLEXITY_MARGIN of 0.05 (measured mean over 160 fired
  writes: 0.0077). Contract C3 pins this across a 4x3 threshold grid. NOTE the 669b
  docstring proposes a LOWER merge_similarity: that is the WRONG SIGN (it moves more
  contexts into the REINFORCE branch and worsens saturation). Do not follow it.
  Measured with centering ON, same nursery, 160 writes: n_slots=64 / merge 0.8 / cthr 0.2
  -> 26 anchors, mean complexity 0.076. R3 clears and C3's margin becomes reachable.
  FOR THE V3-EXQ-669c RE-ISSUE: 669b's super_ordinal_n_slots=16 CAPS the bank (28
  allocations into 16 slots) and would re-flatten C1 by saturating every arm at the same
  ceiling. Set super_ordinal_n_slots=64 with super_ordinal_cue_centering=True and leave
  669b's other thresholds alone.
  Biological basis / architecture: this is the SD-066 fix (common-mode-invariant centered
  readout, validated on the SD-051 ConditionedSafetyStore) applied to a SECOND consumer of
  the same broken z_world geometry. Any future consumer taking an ABSOLUTE cosine on
  z_world should be assumed to need centering until SD-008 is resolved; SD-070 records that
  the prescribed P0 training recipe collapses z_world rather than repairing it.
  Design detail: keys are stored RAW and centered at comparison time, NOT stored
  pre-centered — so a drifting baseline moves query and stored keys together and can never
  leave the store internally inconsistent. The baseline advances BEFORE the write_enabled
  gate (it is cue geometry, not anchor content, so it must keep tracking the context
  distribution through the adult write-frozen phase) and also on retrieve().
  Phased training: NOT required — pure stateful tensor store, no nn.Module, no trainable
  parameters, no gradient flow. MECH-094: simulation_mode contexts never advance the
  baseline (replay/DMN must not shape waking cue geometry).
  Contracts: tests/contracts/test_sd_077_centered_super_ordinal_cue_key.py (C1 default-off
  bit-identity + reproduced 669b signature, C2 centering separates, C3 no threshold setting
  substitutes, C4 lazy seed + MECH-094, C5 raw-key self-match under baseline drift,
  C6 persistence incl. pre-SD-077 checkpoint load).
  Validation experiment: V3-EXQ-669c (queued 2026-07-21) — the 669b ordering test re-issued
  on the centered cue key. Never reuse the id V3-EXQ-669b: its coordinator DB row is
  terminal and POST /queue/add refuses it with HTTP 409.
  See SD-066, SD-008, SD-070, MECH-189, MECH-329, DEV-NEED-006, DEV-NEED-024.

- SD-078: policy.common_mode_invariant_candidate_rule_field_context_key — IMPLEMENTED 2026-07-22.
  ARC-063 CandidateRuleField, ree_core/policy/candidate_rule_field.py.
  Config: CandidateRuleFieldConfig.cue_centering (default False; set True to enable) +
  cue_baseline_alpha (default 0.02). REEConfig knobs crf_cue_centering /
  crf_cue_baseline_alpha, plumbed through all three from_dims sites and read in
  agent.py's CRF config build.
  Data flow: z_world context -> slow-EMA common-mode baseline (advanced in step(),
  waking only) -> centered residual -> mint-block cosine + _context_bucket sign
  pattern + gate_and_select cosine. context_tags stored RAW, centered at comparison
  time; baseline PERSISTS across reset() (cue geometry, not rule content).
  Backward compatible: disabled by default; OFF allocates no baseline and _centered()
  is the identity.
  Why: under SD-008 the raw context key sits in a ~0.98-cosine cone, so the mint-block
  fires against the first rule for every later context at ANY expressible threshold —
  the pool is structurally capped at ONE rule and crf_max_pairwise_rule_dist == 0.0 is
  a tautology, not a churn symptom. Measured on the V3-EXQ-669b nursery: 1 rule /
  dist 0.0000 raw vs 9 rules / dist 1.7011 centered.
  The two 654b-amend mitigations are measured INEFFECTIVE and deliberately left in
  place unchanged: mature_mint_block_threshold=0.8 cannot clear a 0.9426 floor, and
  crf_context_from_e2_world_forward routes to a context carrying the same offset
  (still 1 rule, dist 0.0000). Do not re-tune either as the fix — contract C3 pins it.
  Phased training required: no (pure stateful tensor store, no nn.Module).
  Contracts: tests/contracts/test_sd_078_centered_candidate_rule_field_context_key.py
  (C0 fixture geometry guard, C1 default-off bit-identity + reproduced 654b signature,
  C2 centering separates, C3 no mint-block threshold in [0.5, 0.94] substitutes,
  C4 lazy seed + MECH-094, C5 raw-tag self-match under baseline drift, C6 bucket
  is centered too).
  See SD-066, SD-077, SD-079, SD-008, SD-070, ARC-063, SD-033a, MECH-262.

- SD-079: hippocampal.common_mode_invariant_goal_anchor_match — IMPLEMENTED 2026-07-22.
  SD-039 Anchor.goal_match + AnchorSet + MECH-292 GhostGoalBank,
  ree_core/hippocampal/anchor_set.py + ree_core/hippocampal/ghost_goal_bank.py.
  Config: AnchorSetConfig.goal_cue_centering (default False; set True to enable) +
  goal_cue_baseline_alpha (default 0.05). REEConfig.from_dims(goal_cue_centering=,
  goal_cue_baseline_alpha=) plumbed to the nested AnchorSetConfig.
  Data flow: z_goal cue -> AnchorSet slow-EMA baseline (advanced on BOTH the write
  path, write_anchor with a goal_payload, and the read paths query_by_goal_match /
  GhostGoalBank.rank) -> Anchor.goal_match(z_goal, baseline=) on centered residuals.
  Snapshots stored RAW, centered at comparison time. baseline=None is the identity,
  so the pre-SD-079 call form is bit-identical.
  Backward compatible: disabled by default; OFF allocates no baseline.
  Why: z_goal is an EMA attractor pulled toward z_world and carries the SD-008 offset
  MORE strongly than its source (pairwise cosine min 0.9878 vs 0.9767). Measured over
  a 24-anchor pool: goal_match spread 0.0111 raw vs 0.9709 centered; MECH-292's
  goal_match_floor excluded 0/24; MECH-339's outshining gate sat at EXACTLY 0.0 for
  every anchor, i.e. its context channel was unconditionally dead whenever enabled.
  ALPHA IS 0.05, NOT SD-066/077/078's 0.02 — do not harmonise it. z_goal is an
  integrator so its common mode DRIFTS and 0.02 lags it (measured spread 0.0942 /
  0.1508 / 0.3319 at 0.02 vs 0.9995+ at 0.05 over seeds 101/202/303); 0.2+ over-tracks
  (14-18 of 20 anchors driven below the floor).
  The WRITE-path advance is load-bearing: advancing on reads alone seeds the baseline
  FROM THE QUERY, zeroing every residual (measured — the first ON arm scored 0.0000
  across all 20 anchors). Contract C6 pins it.
  Phased training required: no.
  Contracts: tests/contracts/test_sd_079_centered_goal_anchor_match.py (C0 fixture
  geometry, C1 default-off bit-identity, C2 centering widens the match range, C3 both
  downstream ABSOLUTE gates unpin, C4 lazy seed + MECH-094, C5 raw-snapshot self-match,
  C6 write path advances the baseline, C7 alpha not in the over-tracking regime — only
  that bound is fixture-assertable, see the test's scope note).
  See SD-066, SD-077, SD-078, SD-008, SD-070, SD-039, MECH-292, MECH-339, MECH-293, MECH-340.

- SD-081: e3.dualsystem_uncertainty_arbitration -- IMPLEMENTED 2026-07-22.
  ree_core/predictors/e3_selector.py (_arbitrate_dual_system + _get_world_states depth
  limit), ree_core/agent.py (u_habit resolution in select_action).
  Config: E3Config.use_dualsystem_arbitration (default False; set True to enable), plus
  dualsystem_arbitration_gain 4.0, dualsystem_arbitration_bias 0.0,
  dualsystem_uncertainty_ema_alpha 0.05, dualsystem_habit_depth 2.
  Data flow: familiarity(z_world) -> u_habit (fallback E1 novelty EMA) + E3
  _running_variance -> u_planned -> per-pathway EMA normalisation u/(u+ema) ->
  w = sigmoid(gain*(u_habit_n - u_planned_n) + bias) -> blend of the z-scored HABIT
  (depth-2) and PLANNED (full-horizon) score vectors -> E3 select() scores, upstream of
  raw_scores / score_bias / commit gate / argmin.
  Backward compatible: disabled by default; the block is skipped entirely (no second
  scoring pass, no familiarity query, last_arbitration stays None).
  Biological basis: Daw, Niv & Dayan 2005, Nature Neuroscience 8(12):1704-1711 (conf
  0.79) -- control is allocated to whichever controller is less uncertain; differential
  recruitment is the OUTPUT of an arbitrator, not a property of having two pathways.
  Phased training required: no (no learned parameters -- deliberately NOT a learned gate,
  which would confound "the arbitrator works" with "the gate trained"). MECH-094 N/A
  (nothing written to memory).
  THE PARAMS LIVE ON E3Config, NOT REEConfig. E3Selector.config IS the E3Config, so a
  REEConfig-level field reads as a missing attribute in the selector and defaults to
  False -- the silently-unreachable-flag hazard one level below the documented from_dims
  one. This build tripped exactly that (arbitrator wired, kwargs arriving, 45 select()
  calls, zero arbitrations, no error). Both levels are pinned by contract.
  HABIT DEPTH IS 2 AND FLOORED AT 2 -- do not "simplify" it to 1. Index 0 of the z_world
  sequence is the CURRENT state, shared by every candidate, so a depth-1 score vector has
  cross-candidate range EXACTLY 0.0 and carries no ranking. This is not hypothetical: it
  is the confirmed defect in V3-EXQ-786a's recruitment DV, whose first-step vector was
  constant (n_unique=1 over 32 candidates on every tick, measured under 786a's own
  config). Its _spearman degeneracy guard could not fire, because it tests the std of the
  RANKS and double-argsort of a constant vector is a permutation of 0..K-1. The DV
  therefore measured tie-break noise: simulated mean 1.0173 sd 0.1871 against the
  manifest's reported 1.01725 with per-layout sds 0.149-0.207. Re-adjudicating that run
  is /failure-autopsy + /governance work, NOT settled here.
  Contracts: tests/contracts/test_sd081_dualsystem_arbitration.py (defaults off, flag
  reachable through from_dims at BOTH config levels, OFF bit-identity + no state written,
  ON changes the action stream with a live paired series, weight monotone in relative
  uncertainty = MECH-477's mandatory manipulation check, depth floor blocks the 786a
  degeneracy, depth limit never leaks out of the habit pass).
  Validation experiment: V3-EXQ-811 (script
  experiments/v3_exq_811_mech477_dualsystem_arbitration_falsifier.py, lineage module
  experiments/_lib/baselines/mech477_dualsystem_arbitration.py) ran 2026-07-23 (run_id
  v3_exq_811_mech477_dualsystem_arbitration_falsifier_20260723T054309Z_v3),
  evidence_direction=non_contributory, interpretation=substrate_not_ready_requeue -- both
  arms measured an exact-zero score range against the readiness gate
  (full_score_range_non_degenerate and habit_score_range_non_degenerate both failed on
  arm_off and arm_on). A /failure-autopsy has been separately spawned to root-cause the
  discrepancy against the smoke test's non-zero ranges; this citation should be revisited
  once that lands. NOTE the OFF arm cannot be 786a as-run (degenerate DV, see above); both
  arms needed a fresh run with the depth-2 habit read.
  See MECH-477, MECH-163, ARC-071 (transfer -- DISTINCT from this allocation mechanism;
  BUILT 2026-07-22, see the entry below), ARC-007, ARC-016, MECH-112,
  REE_assembly/docs/architecture/sd_081_dualsystem_uncertainty_arbitration.md.

- SD-082: pfc.lateral_pfc.rule_selection_action_consumer -- IMPLEMENTED 2026-07-26.
  ree_core/pfc/lateral_pfc_analog.py (LateralPFCAnalog.compute_bias + __init__).
  Config: LateralPFCConfig.rule_readout_consumer (default False; set True to enable) +
  readout_init_scale (default 0.25); REEConfig.lateral_pfc_rule_readout_consumer /
  lateral_pfc_readout_init_scale plumbed through all three from_dims sites and read in
  agent.py's LateralPFCConfig build (getattr-fallback).
  Data flow: SD-078 centered CandidateRuleField -> differentiated rule_state ->
  LateralPFCAnalog.rule_state -> compute_bias: (i) subtract the common mode from the
  per-candidate z_world summaries (when K>=2) so the SD-008 ~0.98-cosine cone no longer
  saturates every candidate to the same rail, then (ii) bound with bias_scale*tanh(raw/
  bias_scale) instead of a hard clamp -> E3 per-candidate score_bias.
  WHY: V3-EXQ-822 found the differentiated rule_state (on_rule_state_diff 0.644) was
  behaviourally SILENT -- propagation to the action bias was exactly 0.0 on BOTH arms.
  Root cause (reproduced): the hard clamp on the raw common-mode-dominated summaries maps
  every candidate to the identical rail, so zeroing rule_state changes nothing (structural
  zero) AND the clamp's flat region has zero gradient so REINFORCE cannot train the head
  (the observed 70ep null). Centering de-saturates (robust prop) + tanh restores the
  gradient (grad-norm 0.0 hard-clamp -> ~6.1 soft-tanh). Same magnitude bound
  (|bias| < bias_scale), so the SD-033a "bias cannot dominate E3" guarantee is preserved.
  Backward compatible: disabled by default; OFF path is bit-identical to the SD-033a
  landing (hard clamp on raw input) -- verified torch.allclose, and the existing
  v3_exq_822 script (flag absent) still reproduces prop=0.0 on both arms.
  Biological basis: corticostriatal rule-to-action mapping -- selection without a trained
  read-out to action is inert. Completes the SD-033a signature-(iv) trained-head variant
  (DESIGN ALTERNATIVE A2) that the landing deferred.
  Phased training required: yes (P0 warmup, P1 frozen-encoder+CRF bias-head REINFORCE, P2
  eval -- the existing 822 P0/P1/P2 protocol). MECH-094 N/A (pure forward read; no memory
  write; rule_state is MECH-261 gate-protected).
  NOT captured by test_flag_registry_is_current (name has the lateral_pfc_ prefix, not a
  use_*/`*_enabled` name -- same convention as lateral_pfc_train_rule_bias_head).
  Validation experiment: V3-EXQ-822a queued (re-run of 822, same question, consumer flag
  ON on both arms). Acceptance: on_prop_delta_mean >= 0.001 with an ON>OFF contrast.
  See SD-033a, SD-078, ARC-063 (GAP-B/GAP-D), SD-008, SD-066/SD-077,
  REE_assembly/docs/architecture/sd_082_rule_selection_action_consumer.md.
