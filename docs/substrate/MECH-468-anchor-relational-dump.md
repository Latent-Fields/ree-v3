## MECH-468 Anchor Relational-Edge Recording (2026-09-24)
- MECH-468 anchor_relational_dump: A/C/D/E per-anchor relational-edge recording -- IMPLEMENTED 2026-09-24.
  Four independent read-side (A/C/D) plus in-loop (E) recording additions over the
  existing dual-trace anchor pool, proposed by
  `REE_assembly/evidence/planning/mech468_edge_type_inventory_spike.md` Section 4 and
  registered as `substrate_queue.json` entry `mech468-anchor-relational-dump`. Relation
  type B (action transition) is explicitly OUT OF SCOPE -- it needs new substrate
  wiring on the action/decision path, a different kind of change from these four
  read-side dumps; not bundled here.

  **A (latent proximity):** `AnchorSet.dump_relational_snapshot(scale=None)`
  (`ree_core/hippocampal/anchor_set.py`) -- a bounded pairwise cosine-proximity score
  over `z_world` (same non-negative-clamped cosine recipe as `Anchor.goal_match`, no
  baseline centering) for every pair in `all_anchors(scale)`. Returns `{}` unless
  `AnchorSetConfig.record_relational_snapshot` is True.

  **C (shared event):** `consume_boundary_events` tags each installed anchor with
  `{t, scale, segment_id_old, segment_id_new, sources, anchor_key}` into a bounded
  ring buffer (`AnchorSetConfig.relational_event_log_max_len`, default 2048),
  returned by `dump_relational_snapshot()["shared_event_edges"]` -- joinable on `t`
  to recover which anchors, across families, shared a causing event.

  **D (shared goal/valence):** `GhostGoalBank.rank()` (`ree_core/hippocampal/ghost_goal_bank.py`)
  additionally retains the RAW per-anchor `{anchor_key, goal_match, wanting_strength,
  arousal_tag}` (as opposed to the WEIGHTED composite terms already in
  `GhostGoalBankEntry.components`) via `get_relational_components()`, when
  `GhostGoalBankConfig.record_relational_components` is True. Covers only the
  direct-admission loop; SD-097 relational-successor entries are not yet covered
  (a possible future extension).

  **E (causal/outcome):** `StalenessAccumulator.integrate()` (`ree_core/hippocampal/staleness_accumulator.py`)
  logs each non-zero per-anchor `{t, source_scale, source_segment_id_old,
  target_anchor_key, attribution_weight, strength}` increment via `edge_log()`
  (bounded ring buffer, `StalenessAccumulatorConfig.edge_log_max_len`, default 4096)
  INSIDE the accumulation loop, before it folds into `self._staleness` -- the only
  point the edge-level credit is ever recoverable, since the sum+leak design
  destroys it the tick after `integrate()` runs (`get_stats()`/`snapshot()` expose
  only the post-leak, post-accumulation region aggregate). Gated by
  `StalenessAccumulatorConfig.record_edge_log`.

  Config: `AnchorSetConfig.record_relational_snapshot` (default False),
  `AnchorSetConfig.relational_event_log_max_len` (default 2048),
  `GhostGoalBankConfig.record_relational_components` (default False),
  `StalenessAccumulatorConfig.record_edge_log` (default False),
  `StalenessAccumulatorConfig.edge_log_max_len` (default 4096).
  Data flow: existing tick-loop computation (write_anchor / consume_boundary_events /
  rank() / integrate(), all already called every tick in normal operation) -> bounded
  in-memory log or pull-style accessor -> a future experiment driver calls the
  accessor at manifest-write time, the same idiom as `get_diagnostics()`/`get_stats()`
  elsewhere in this module family. No agent.py/module.py wiring needed -- the
  accessors populate automatically once their flag is on.
  Backward compatible: every flag defaults False; OFF path is bit-identical (proven
  by `tests/contracts/test_mech468_relational_dumps.py` C1, including exact
  `ghost_priority`/`components`/`sums`/`snapshot()` equality between ON and OFF
  configs on the same inputs -- the new state is purely additive).
  Non-degeneracy per the spike's Section 3 floors (A: anchor-count / edge-density
  after thresholding; C: events shared across >=2 concurrently-active families; D:
  goal_match IQR spread; E: stream_overlap mode is NOT fully-connected, in contrast
  to the default equal mode which is fully-connected by construction) is exercised
  in the same contract test file on synthetic-but-realistic pools -- a property of
  the data the recording returns, not enforced by the recording code itself.
  Phased training: not applicable (pure recording, no trainable parameters).
  MECH-094: not applicable (recording only; no simulation/replay write path is
  touched, and none of the four types write anything back into agent state).
  Validation experiment: not queued by this build (chip's explicit instruction --
  substrate landing only). Owed next: `/queue-experiment` re-checks EXP-1107
  (MECH-470, `blocked_substrate` on MECH-468) and a fresh MECH-469 typed-vs-collapsed
  proposal now that the recording exists; any E-type probe must set
  `attribution_mode=stream_overlap`, not the default `equal` (fully-connected by
  construction per the spike).
  See MECH-468, MECH-469, MECH-470.
