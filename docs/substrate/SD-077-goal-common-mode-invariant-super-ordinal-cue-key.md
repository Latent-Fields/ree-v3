## SD-077: goal.common_mode_invariant_super_ordinal_cue_key -- IMPLEMENTED (2026-07-21)
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
