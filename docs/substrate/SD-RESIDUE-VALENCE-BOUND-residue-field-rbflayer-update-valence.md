## SD-RESIDUE-VALENCE-BOUND: residue.field.RBFLayer.update_valence.accumulator_bound -- IMPLEMENTED (2026-08-11)
- SD-RESIDUE-VALENCE-BOUND: residue.field.RBFLayer.update_valence.accumulator_bound --
  IMPLEMENTED 2026-08-11. Full design: `REE_assembly/docs/architecture/sd_residue_valence_bound.md`.
  Built from `failure_autopsy_V3-EXQ-906a_894b_2026-08-09.md` (original finding, `excite` only)
  + `failure_autopsy_906b-906c-911-cluster_2026-08-10.md` (scope widened to all 6 valence
  components), routed via `substrate_queue.json` (`sd_id: SD-RESIDUE-VALENCE-BOUND`,
  `severity: degrading`, `node_class: complicated (buildable)`).
  Fixes `RBFLayer.update_valence()` (`ree_core/residue/field.py`) -- the single write path
  behind all 6 SD-014/ARC-036 valence components (wanting/liking/harm_discriminative/surprise/
  positive_surprise/negative_surprise, MECH-307 Gap-1 Option-b) -- which was a raw unclamped
  `self.valence_vecs[center_idx, component] += value` with no decay. Only 32 RBF centers exist
  by default, so a long-lived agent revisiting the same regions drove the same centers'
  components unboundedly: `z_world_norm` ~150-320 vs ~0.5-0.7 at smoke scale (906a); `liking`
  mean=19.88 vs a 0.39 smoke ceiling, `dread` rising 40-110x across episodes (906b/906c cluster).
  Fix: a leaky-integrator decay + hard clamp (`v <- v*(1-decay_rate) + value`, then
  `clamp(v, -clamp_abs, clamp_abs)`), gated behind a NEW master switch rather than an
  unconditional default change -- contrast SD-ORIENTING-DECISION-SCALE immediately above, whose
  buggy path lived behind an off-by-default switch only one experiment ever exercised.
  `valence_enabled` defaults `True` and this write path is exercised by dozens of historical
  experiments with real `claims.yaml` evidence, so an unconditional fix would be a mechanism
  change requiring the `/implement-substrate` Step-8.5 evidence-staleness audit across all of
  them -- disproportionate to a plumbing bound-fix. `ResidueField.update_valence()` and
  `ResidueField.update_wanting_sensitized()` (the two callers, covering all 13 `agent.py` call
  sites plus the SD-014 incentive-sensitized WANTING path) resolve both parameters from config
  through one shared helper, `_valence_bound_params()`, so the fix applies from a single point.
  Config: `ResidueConfig.valence_bounding_enabled` (default `False` -- master switch; `False` ->
  `update_valence()` is the exact pre-fix `+=`, bit-identical) + `valence_decay_rate` (default
  `0.02`, mirrors this class's existing `integration_rate` timescale convention) +
  `valence_clamp_abs` (default `5.0`, headroom above the smoke-scale ceiling while staying far
  below the observed contamination). All three NEW, on `ResidueConfig` (not `REEConfig`
  top-level) -- unlike `REEConfig`-level knobs, `ResidueConfig` fields are not threaded through
  `REEConfig.from_dims()` as named kwargs (`valence_enabled`/`safety_terrain_enabled` are not
  either); set by direct post-construction assignment
  (`cfg.residue.valence_bounding_enabled = True`), matching the existing sibling-flag
  convention. No `from_dims()` signature change was needed or made.
  Backward compatible: confirmed by direct unit test (200 successive `+1.0` writes to the same
  component/center produce exactly `200.0` under default config; bounded at `valence_clamp_abs`
  under `valence_bounding_enabled=True`; `update_wanting_sensitized` tested identically). A full
  `--dry-run` of `v3_exq_887_sd014_node_valence_representational_functional.py` (an existing
  experiment exercising this exact write path at default config) ran end-to-end with no error.
  Phased training / MECH-094: not applicable -- `valence_vecs` is a `register_buffer`, not an
  `nn.Parameter`; every write already executes under `torch.no_grad()`. No gradient flow, no
  trainable parameters. The existing `hypothesis_tag` gate sits upstream of the new decay/clamp
  logic and is untouched (confirmed by a dedicated unit test: with `hypothesis_tag=True`
  throughout, the tracked value never leaves its 0.0 init, under BOTH bounding conditions).
  Validation experiment: `V3-EXQ-918` queued (`v3_exq_918_sd_residue_valence_bound_validation.py`,
  `experiment_purpose=diagnostic`, `claim_ids=[]`) -- unit-level ON/OFF ablation directly against
  `ResidueField`/`RBFLayer` (no `CausalGridWorld`/`REEAgent` needed, matching the `V3-EXQ-520`
  Part-1 precedent), reproducing sustained same-center writes across
  `positive_surprise`/`negative_surprise` (MECH-307 split-surprise) and `wanting` (via
  `update_wanting_sensitized`). Smoke PASS: OFF arm reproduces unbounded growth (200.0/395.5
  after 200 writes), ON arm plateaus exactly at `clamp_abs=5.0`.
  Descoped: the same failure-autopsy cluster also flagged `update_benefit_salience()`/
  `update_schema_wanting()` (the `residue_wanting`/`VALENCE_WANTING` writer methods) as "never
  called from the 906-family agent step loop." Traced directly against source: `REEAgent` has no
  internal step loop at all -- every experiment driver calls these explicitly itself (confirmed
  via `experiments/_harness.py`'s `StepHarness` and ~20 other scripts that already call them
  correctly). The 906-family driver simply omits these calls -- an experiment-script gap, not a
  `ree_core/` substrate gap, so left as a reported follow-on rather than folded into this change
  (this file's mandatory skill-path rule restricts `experiments/` edits to `/queue-experiment` or
  `/diagnose-errors`).
  No `claims.yaml` claim gates on this SD (`unblocks_claims: []` in `substrate_queue.json` --
  all three motivating runs, 906a/906c/the cluster, are `claim_ids: []` diagnostics), so no
  `v3_pending` flip was made.
  Registry-drift side-fix (same session, user-directed): `tests/test_flag_inertness.py`'s
  `test_flag_registry_is_current` previously scanned only `REEConfig`'s own top-level fields,
  silently missing every `use_*`/`*_enabled` flag on one of `REEConfig`'s 13 nested config
  classes (85 flags total, including this SD's own new `valence_bounding_enabled`, which would
  itself have landed uncategorized under the old scan -- the gap that surfaced this fix).
  Widened to scan every dataclass in `ree_core/utils/config.py` (`_current_nested_flags()`);
  `valence_bounding_enabled` is individually probed
  (`test_sd_residue_valence_bound_bounds_the_accumulator`), the other 84 pre-existing nested
  flags are bulk-seeded into `KNOWN_UNPROBED_NESTED` with a generic placeholder reason (a real
  per-flag audit is out of scope for this session -- tracked as a follow-on chip). Also caught
  and categorized one genuinely pre-existing top-level gap the widening exposed as a side
  effect: `use_defensive_orienting` (SD-ORIENTING-DECISION-SCALE / SD-099, landed
  2026-08-08/2026-08-10, unrelated to this session) was already missing from the registry before
  this widening.
  See MECH-307 (the split-surprise mechanism whose excite/dread writes are the most-affected
  components), SD-014 / ARC-036 (the valence-vector mechanism this accumulator belongs to).

- SD-MECH303-THRESHOLD-SOURCING: mech303.contextual_safety_gate.dedicated_proximity_signal --
  IMPLEMENTED 2026-08-14. ree_core/environment/causal_grid_world.py (dedicated EMA + obs
  channel), ree_core/agent.py (sense() gate selector), ree_core/utils/config.py (selector +
  threshold).
  Gives MECH-303's contextual-safety accumulate gate a DEDICATED anticipatory hazard-proximity
  signal, decoupled from SD-022's damage-sourced z_harm_a. V3-EXQ-917 found the gate cannot
  discriminate safe-vs-unsafe under damage-sourcing (AUC <=0.52, chance) at any of 18 thresholds,
  while a proximity signal reaches AUC 0.84-0.97; a single z_harm_a cannot serve both SD-022
  (wants body/context decoupled) and MECH-303 (wants body/context coupled), so this adds a second
  signal for MECH-303's gate alone -- every other z_harm_a consumer is untouched.
  Env: CausalGridWorldV2(safety_proximity_signal_enabled=False default; True emits
  obs_dict["safety_proximity_harm"], a scalar tau~20 EMA of hazard-proximity-at-agent, updated
  BEFORE the Q-080 effort injection so it is decoupled from both SD-022 and Q-080;
  safety_proximity_ema_alpha=0.05).
  Config: REEConfig.contextual_safety_gate_source ("z_harm_a" default -> unchanged z_harm_a.norm()
  gate; "proximity_signal" -> dedicated signal), contextual_safety_proximity_threshold (0.25;
  "harm absent" gate for the dedicated signal, ~0 safe / ~0.8 dense-hazard). Existing
  contextual_safety_harm_threshold (0.05) untouched.
  Data flow: hazard_at_agent -> env _safety_proximity_ema -> obs_dict["safety_proximity_harm"]
  -> agent.sense(obs_safety_proximity=...) -> MECH-303 accumulate_safety gate (proximity path;
  NO silent z_harm_a fallback when the signal is absent).
  Backward compatible: both switches default no-op; env output bit-identical OFF, gate reads
  z_harm_a by default; new sense() kwarg defaults None. Smoke (2026-08-14): safe(nh=0) mean signal
  0.00 vs unsafe(nh=8) 0.87; end-to-end proximity gate fires 150/150 safe vs 6/150 unsafe.
  Not a learning module -- no encoder, no phased training. MECH-094 N/A (accumulate retains its
  existing hypothesis_tag waking-path guard; signal is a waking env observable).
  Validation experiment: V3-EXQ-930 queued (917-style AUC sweep; acceptance AUC >=~0.84 >> chance
  0.52). See docs/architecture/sd_mech303_threshold_sourcing.md, MECH-303, SD-052, SD-011, SD-022.

- contextmemory-write-path-addressing-degeneracy: e1.context_memory.write_usage_balancing --
  IMPLEMENTED 2026-08-19 (chip-20260816-implsub-contextmemory-writepath-degeneracy).
  ree_core/predictors/e1_deep.py (ContextMemory.__init__ / .write()), ree_core/utils/config.py
  (E1Config fields + REEConfig.from_dims threading).
  `ContextMemory.write()` addresses by a hard `scores.mean(0).argmin()`, which under a
  near-constant query stream is a deterministic single-slot fixed point --
  failure_autopsy_V3-EXQ-436e_2026-08-13.md established a closed-form sign discriminator
  `q . (write_signal - memory[argmin])` that predicted lock-vs-rotate 5/5, and V3-EXQ-436f
  confirmed it live (n_occupied_slots = 1 of 16 in BOTH arms on 3/5 seeds despite 2,837-4,903
  write() calls per arm, with the full SD-016 production combination armed and engaged). This
  is the WRITE-side sibling of SD-016 (cue-indexed RETRIEVAL, already implemented) -- the
  READ-path fix does not touch this defect: write() runs entirely under torch.no_grad() and
  compute_diversification_loss() only ever trains self.memory, never the write-address
  selection itself.
  Fix: a frequency-sensitive competitive-learning "conscience" bias (DeSieno 1988) on the
  write-address selection score -- a slot's eligibility for re-selection is penalized in
  proportion to an EMA of how recently/often it has already won, scaled to
  sqrt(memory_dim) (the expected order of magnitude of the raw dot-product scores by CLT), so
  a self-reinforcing lock cannot persist. No gradient is required (write() is already
  no_grad), so this is a pure selection-score adjustment, not a differentiable relaxation like
  the SD-016 read-path Gumbel-softmax (V3-EXQ-908) it is the write-side sibling of; the
  annealed-Gumbel alternative named in the substrate_queue implementation_hint was
  investigated and not used, since an annealing schedule and a straight-through estimator both
  exist to preserve gradient flow that write() never has.
  Config: E1Config.contextmemory_write_usage_balancing (bool, default False -- bit-identical
  to the legacy argmin path, no extra buffer constructed), .contextmemory_write_usage_bias_weight
  (float, default 1.0), .contextmemory_write_usage_decay (float, default 0.99, ~100-write EMA
  time constant). All three threaded through REEConfig.from_dims().
  Data flow: query_proj(state) . memory.T -> mean_scores -> + usage-EMA bias (when enabled) ->
  argmin -> write_usage_ema EMA-updated toward the winning slot.
  Backward compatible: default False; with the flag off, `selection_scores` is exactly
  `scores.mean(0)` and `write_usage_ema` is None -- verified bit-identical against the legacy
  write() expression.
  Validated (contract-test level, ree-v3/tests/contracts/test_contextmemory_write_usage_balancing.py):
  replaying a perfectly constant query stream (the failure_autopsy's own H-degenerate-query
  regime) against 3 seeds independently confirmed to lock the legacy path to n_occupied_slots=1
  over 3000 write() calls -- all 3 reach n_occupied_slots >= 2 under the fix (measured 3, 3, 16
  in a broader 20-seed sweep: 11/20 seeds locked under the legacy path, 20/20 reached >= 2
  occupied slots under the fix).
  Not a learning module in the gradient sense -- write() is no_grad throughout; no phased
  training. MECH-094 N/A (write() is a waking-stream operation, not simulation/replay content).
  Validation experiment: diagnostic-purpose ablation queued via /queue-experiment (see queue
  entry for EXQ id) -- feature ON vs OFF, same seeds, measuring n_occupied_slots against the
  substrate_queue entry's own registered acceptance target (>= 2 occupied slots in both arms
  on >= 3/5 seeds).
  See SD-017, ARC-045, MECH-166 (unblocked for their own validation now that the write-path
  precondition is repaired), SD-016, V3-EXQ-436e, V3-EXQ-436f.

- contextmemory-write-path-addressing-degeneracy: e1.context_memory.write_selection="refractory" --
  IMPLEMENTED 2026-08-19 (chip-20260819-contextmemory-add-refractory-mode; USER-AUTHORISED build,
  option (c) of chip-20260819-contextmemory-writesel-disposition-evidenced).
  ree_core/predictors/e1_deep.py (ContextMemory.__init__ / .write() / ._select_write_slot() /
  ._record_write()), ree_core/utils/config.py (E1Config fields + REEConfig.from_dims threading).
  SECOND mechanism for the same substrate_queue entry, landed ALONGSIDE the conscience bias
  above. That one is NOT reverted, replaced, or re-scaled -- both are available and both are
  default-off.
  The two are orthogonal by construction: `contextmemory_write_usage_balancing` adjusts the
  SCORE, `contextmemory_write_selection="refractory"` restricts the ELIGIBLE SET (the k
  most-recently-written slots are ineligible). They compose, and all four combinations are
  legal. Deliberate rather than a missing mutual exclusion -- mutual exclusion would need a
  raise or a silent precedence rule, whereas the composition is well defined and measurable.
  Config: E1Config.contextmemory_write_selection (str, default "argmin" = every slot eligible =
  bit-identical legacy path; only "argmin" and "refractory" accepted, anything else raises at
  construction), .contextmemory_write_refractory_k (int, default 2, inert unless the mode is
  "refractory", capped internally at num_slots - 1 so a large k degrades instead of deadlocking).
  Both threaded through REEConfig.from_dims() at all three sites (from_dims silently swallows
  unknown kwargs, so a knob wired at two of three fails open and silently).
  Data flow: query_proj(state) . memory.T -> mean_scores -> + usage-EMA bias (when enabled) ->
  _select_write_slot() masks the last-k window with +inf -> argmin over the remainder ->
  _record_write() updates slot_write_counts / last_write_index.
  Backward compatible: default "argmin"; verified bit-identical against the verbatim legacy
  write() expression over 5 seeds x 200 writes (identical slot sequence AND identical final
  memory tensor). Consumes NO RNG in any mode -- refractory is fully deterministic -- so every
  existing seeded trajectory is unchanged. state_dict() unchanged (slot_write_counts is
  persistent=False, so existing checkpoints load untouched); named_buffers() does gain
  slot_write_counts, which the landed sibling contract's negative control tolerates by scoping
  itself to "write_usage_ema" names (pinned, not assumed).
  WHY BOTH MECHANISMS EXIST, AND WHAT DOES NOT JUSTIFY EITHER. Both clear the registered
  acceptance floor (>= 2 occupied slots on >= 3/5 seeds), so occupancy CANNOT choose between
  them. The independently pre-registered probe (REE_assembly/evidence/planning/
  contextmemory_write_selection_comparison_20260819.md; pre-registration REE_assembly
  fcfb311e4b, results b7e072ddf0) found the occupied-slot COSINE column cannot discriminate the
  arms at 5 seeds -- every contrast |dz| <= 0.47, |t(4)| <= 1.04, sign-inconsistent across
  seeds, INCLUDING the +0.6060 -> +0.5919 refractory-over-legacy gap (dz = -0.06). Do NOT cite
  that column for either mode. The case for refractory is STRUCTURAL and rests on the probe's
  deterministic columns, reproduced exactly by this landing: (1) occupancy >= k+1 holds BY
  CONSTRUCTION for any stream/seed/init, where the conscience bias reaches full occupancy only
  empirically; (2) with the bias on, the sqrt(memory_dim) = 11.31 scaling puts the usage term
  2-3 orders of magnitude above the ~0.026 across-slot spread of mean_scores, so the address
  becomes a function of the write COUNTER rather than the query -- a strict LRU cycle on ~99%
  of writes (round-robin agreement 0.991, entropy exactly 4.00 bits, HHI exactly 1/16, all 16
  slots on every seed regardless of content). Real occupancy, and not globally content-blind
  since the cycle ORDER is content-determined once, but occupancy without addressing.
  Refractory scores 0.000 on that same index and its slot count tracks legacy's on the
  non-locking seeds. (3) An absolute refractory period on a recently-active unit is a
  first-order property of real neurons; a global usage-EMA conscience bias is not.
  NOT ADOPTED: the salvaged branch's "usage" and "gumbel" modes (ree-v3 tag
  stash-archive/20260819-dd4b0a4, LOCAL-ONLY to DLAPTOP). Measured content-blind -- cluster
  Jaccard 0.600 and exactly 1.000 on 5/5 seeds respectively, i.e. both contexts write the same
  slot set. The salvaged "usage" mode is also a DIFFERENT algorithm from the conscience bias
  above (argmax(-z(sim) - w*z(usage)) vs argmin(sim + w*usage*sqrt(d))); they must not be
  conflated. Both names are rejected explicitly by the config validator so a config carrying
  them over errors rather than silently falling back to the defect.
  Phased training required: no (write() is entirely under torch.no_grad(); no encoder head and
  no gradient path is added or altered). MECH-094 hypothesis_tag: not applicable (no
  simulation, replay, or non-waking-state content).
  Instrumentation (maintained in EVERY mode, including the legacy default): ContextMemory
  .last_write_index, .slot_write_counts, .occupied_slots(). V3-EXQ-436f's occupancy tracker
  learned the written slot by DUPLICATING write()'s own argmin expression, which reports the
  wrong slot the moment the selection rule changes -- and it has now changed twice. Any new
  driver must read these instead of re-deriving; pinned by
  test_stale_reimplementation_of_the_old_rule_disagrees.
  Contracts: ree-v3/tests/contracts/test_contextmemory_write_address_selection.py (51 with the
  landed sibling file; roughly half negative controls -- the defect pin, bit-identity, RNG
  non-consumption, unchanged state_dict, the landed knobs untouched, and the load-bearing
  test_landed_usage_balancing_is_a_fixed_cycle_and_refractory_is_not). Every numeric assertion
  is on a deterministic quantity; there is deliberately NO assertion on occupied-slot cosine.
  Validation experiment: still PENDING for BOTH modes -- substrate_queue status stays
  implemented_pending_validation. Chipped as
  chip-20260819-queueexp-contextmemory-writesel-validation (covering both arms; the
  previously-recorded chip for the conscience-bias arm was never actually written to
  TASK_CHIPS.json, so this repairs that gap rather than adding a second unowned one).
  Does NOT unblock chip-20260818-sd017-ceiling-retest-gated or
  chip-20260818-mech152-redesign-queue-gated: a default-off knob changes no driver, no driver
  sets either flag, and the occ_cos DV still cannot discriminate at those experiments' powers.
  Plan of record: REE_assembly/evidence/planning/contextmemory_refractory_mode_dataflow_plan_20260819.md.
  See substrate_queue contextmemory-write-path-addressing-degeneracy, SD-017, ARC-045, MECH-166, SD-016.

- contextmemory-write-path-addressing-degeneracy: e1.context_memory.write_selection="gumbel_learned" --
  IMPLEMENTED 2026-08-27 (chip-20260826-contextmemory-gumbel-writeselect-build; HUMAN-DIRECTED build,
  substrate_queue HUMAN DECISION 2026-08-26 via /metaworker-orchestrate).
  ree_core/predictors/e1_deep.py (ContextMemory.__init__ / .write() / ._select_write_slot() /
  ._select_write_slot_gumbel() / .compute_write_addressing_loss(), E1DeepPredictor.__init__),
  ree_core/utils/config.py (E1Config fields + top-level REEConfig field + REEConfig.from_dims
  threading), ree_core/agent.py (REEAgent.compute_prediction_loss() wiring).
  THIRD mechanism for the same substrate_queue entry. Bias and refractory (above) are NOT
  reverted, replaced or re-scaled -- all three are available, all three default-off.
  2026-08-26 HUMAN DECISION: neither bias nor refractory closes the corrupting defect as a
  matter of addressing POLICY (both are mechanical occupancy workarounds, not a learned
  write-selection mechanism); build the entry's own implementation_hint -- annealed
  Gumbel-softmax matching V3-EXQ-908's confirmed READ-path mechanism.
  THIS IS NOT A RESURRECTION OF THE REJECTED "gumbel" MODE FROM THE 2026-08-19 ENTRY ABOVE --
  read that entry's "NOT ADOPTED" paragraph first. The rejected mode perturbed write()'s own
  UNTRAINED query.memory score with Gumbel noise and nothing ever shaped it (write() is
  entirely no_grad), reproducing exactly the same failure V3-EXQ-418i independently diagnosed
  on the read path's legacy q.k attention (uniform softmax saddle at self.memory's 0.01 init
  scale). This build's write_addr_tagger is a DEDICATED feedforward MLP (matching
  cue_slot_tagger's actual shape, not the q.k attention path), scoring state -> num_slots
  logits directly (never touching self.memory's scale), and it receives a REAL, VERIFIED
  gradient via the new compute_write_addressing_loss(), called explicitly by
  REEAgent.compute_prediction_loss() -- the piece compute_diversification_loss() never
  provided (it only ever trains self.memory content, never write-ADDRESS selection). Deliberately
  spelled "gumbel_learned", not "gumbel" -- that string stays rejected, pointing here, so no
  stray reference to the old name silently resurrects it.
  MEASURED, HONESTLY, AND THE RESULT IS MIXED. A first-attempt loss design (standard MoE
  load-balancing / "importance" loss, Shazeer et al. 2017) was built, wired, and then MEASURED
  (5 seeds x 300 SGD steps on the 2-cluster content-conditioning stream) to reproduce the
  IDENTICAL failure signature as the rejected mode -- 2-cluster Jaccard EXACTLY 1.000 on 5/5
  seeds -- because batch-mean uniformity has a degenerate minimum satisfied equally by
  content-blind near-uniform-everywhere and by genuine content-conditioning. REPLACED with a
  pairwise-diversity loss mirroring compute_diversification_loss()'s own structure (mean
  squared off-diagonal cosine similarity) applied to PER-EXAMPLE selection distributions
  instead of memory rows -- this has no such degenerate minimum (near-uniform distributions
  are mutually HIGH-similarity, so the gradient pushes away from them) but its
  content-discrimination effect is NOT YET DEMONSTRATED either: a toy 300-step training loop
  did not move it off its own near-uniform starting point in the time available, which is
  consistent with (not distinguishable from) the SAME symmetry-breaking difficulty
  compute_diversification_loss() itself needed a full experiment (V3-EXQ-907) rather than a
  toy script to overcome.
  Config: E1Config.contextmemory_write_selection accepts "gumbel_learned" (third value;
  "gumbel" bare stays rejected), .contextmemory_write_gumbel_tau_init (1.0),
  .contextmemory_write_gumbel_tau_min (0.1), .contextmemory_write_gumbel_anneal_steps (2000),
  .contextmemory_write_gumbel_tagger_hidden (32). Top-level REEConfig.contextmemory_write_addressing_loss_weight
  (float, default 0.0 -- mirrors sd016_diversification_weight's own top-level placement and
  the reason for it). All threaded through REEConfig.from_dims() at all three sites.
  Data flow (selection): write_addr_tagger(state) -> mean over batch -> + usage-EMA bias (when
  composed) -> annealed straight-through Gumbel-max (train) / plain argmin (eval) ->
  _record_write() updates slot_write_counts / last_write_index -- write() itself stays
  ENTIRELY torch.no_grad() in this mode too, exactly like argmin/refractory.
  Data flow (training signal, SEPARATE call, not reused from write()):
  compute_write_addressing_loss(states) -- a fresh forward pass over a caller-supplied batch
  of already-detached states -- -> write_addr_tagger(states) -> per-example softmax(-scores)
  -> pairwise cosine similarity, minimized -> REEAgent.compute_prediction_loss() adds
  weight * this loss, gated on weight > 0 AND write_selection == "gumbel_learned". Recomputed
  fresh (not retained from write()) because write()'s .data content update bypasses autograd's
  version-counter check -- a retained graph consumed after a later write() would silently
  differentiate through the WRONG (since-mutated) memory values.
  Backward compatible: default "argmin"; write_addr_tagger is None (not constructed) and no
  state_dict keys are added unless "gumbel_learned" is selected. RNG: eval mode is
  deterministic (bit-identical to plain argmin on the tagger's scores, verified exactly, not
  approximately) and consumes no RNG; TRAIN mode DOES consume RNG -- a real, documented
  difference from argmin/refractory, which stay fully deterministic in every mode.
  Occupancy: >= 2 occupied on >= 3/5 seeds cleared decisively (16/16 on all 5 measured seeds),
  delivered by Gumbel noise alone, independent of whether write_addr_tagger has been trained
  at all.
  Contracts: ree-v3/tests/contracts/test_contextmemory_write_gumbel_learned.py (26,
  time-independent). Assertion policy: no assertion claims content-discrimination is achieved
  (that is explicitly not yet proven); assertions cover gradient reachability into
  write_addr_tagger (the concrete difference from the rejected mode), write()'s continued
  full no_grad status, RNG consumption exactly where documented, config validation, and the
  occupancy floor.
  Phased training required: the write_addr_tagger / compute_write_addressing_loss() pairing
  IS a learning module in the gradient sense (unlike bias/refractory) -- MECH-094 N/A (write()
  remains a waking-stream operation; the new loss trains on already-detached buffer states,
  not simulation/replay content).
  Validation experiment: PENDING, and needs MORE than a flag flip -- unlike bias/refractory, a
  gumbel_learned driver must actually run contextmemory_write_addressing_loss_weight > 0
  through real training steps, or it only exercises the (already-decisive) occupancy effect
  and says nothing about the still-open content-discrimination question. Does NOT unblock
  chip-20260818-sd017-ceiling-retest-gated or chip-20260818-mech152-redesign-queue-gated: a
  default-off knob changes no driver, and even with the flag on, content-discrimination is
  unproven.
  Architecture doc: REE_assembly/docs/architecture/contextmemory_write_address_selection.md
  ("THIRD mechanism" section, 2026-08-27) has the full measurement record for both loss-design
  attempts.
  See substrate_queue contextmemory-write-path-addressing-degeneracy, SD-017, ARC-045, MECH-166, SD-016.
