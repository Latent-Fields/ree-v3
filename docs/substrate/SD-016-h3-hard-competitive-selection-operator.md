## SD-016 H3: Hard/competitive selection operator (2026-08-09)
- SD-016 H3: e1.cue_slot_tagger_selection -- IMPLEMENTED 2026-08-09.
  Module: ree_core/predictors/e1_deep.py (E1DeepPredictor.__init__,
  extract_cue_context, new _sd016_gumbel_select / _sd016_topk_select
  helpers), ree_core/utils/config.py (E1Config + from_dims).
  Root cause it addresses (GOV-FANOUT-1 / V3-EXQ-898 failure autopsy,
  2026-08-08): the Path 3 soft feedforward tagger (above), even correctly
  rewarded by terrain_loss, reproduced the EXACT uniform ln(16) attractor
  under C2's control -- suggesting a soft, end-to-end differentiable softmax
  gate may not be able to HOLD a sparse, context-selective optimum regardless
  of what the downstream loss demands (gradient descent relaxes back toward
  the uniform attractor). H3 tests whether a STRUCTURALLY competitive
  selection operator -- independent of what terrain_loss rewards -- breaks
  the saddle where the soft gate could not.
  THE FIX: a selection-mode knob on the Path 3 tagger's slot-SELECTION
  scores only (the slot-CONTENT value_proj/output_proj path, and both
  downstream projections, remain untouched, same discipline as Path 3):
    "soft"   -- the existing Path 3 softmax(logits/temperature) (default,
                bit-identical by construction -- the original line is
                unmodified, just moved into this branch of the dispatch).
    "gumbel" -- straight-through Gumbel-softmax: forward pass is a one-hot
                argmax over Gumbel-perturbed logits (structurally sparse
                regardless of the loss); backward gradient flows through the
                soft relaxation at an annealed temperature (linear anneal
                from tau_init to tau_min over N training-mode forward calls;
                eval-mode calls neither sample noise nor advance the
                schedule, so inference is deterministic).
    "topk"   -- straight-through top-k: forward pass keeps only the top-k
                softmax-weighted slots (renormalised to sum to 1); backward
                gradient flows through the full softmax (same ST mechanics
                as gumbel). k is a config int, clamped to [1, num_slots].
  Config (E1Config + REEConfig.from_dims; all no-op defaults):
    sd016_cue_slot_tagger_selection (str, default "soft") -- "soft" |
      "gumbel" | "topk". Requires sd016_cue_slot_tagger=True; no effect
      otherwise. Invalid values raise ValueError at construction (fail
      loud, not silently-swallowed at forward time).
    sd016_cue_slot_tagger_topk_k (int, default 1) -- topk: slots kept active.
    sd016_cue_slot_tagger_gumbel_tau_init (float, default 1.0) -- gumbel:
      starting temperature.
    sd016_cue_slot_tagger_gumbel_tau_min (float, default 0.1) -- gumbel:
      floor temperature.
    sd016_cue_slot_tagger_gumbel_anneal_steps (int, default 2000) -- gumbel:
      linear anneal horizon in training-mode forward calls.
  Data flow (unchanged from Path 3 except the selection operator):
    z_world -> cue_slot_tagger -> slot_logits[num_slots] -> {soft softmax |
    gumbel ST | topk ST} -> weights @ value_proj(memory) -> output_proj ->
    cue_context (unchanged) -> cue_action_proj([cue_context, z_world]) +
    cue_terrain_proj(cue_context).
  Read-only diagnostic: extract_cue_context still caches the selection
    distribution on E1DeepPredictor._last_cue_slot_weights [batch, num_slots]
    -- for gumbel/topk this is the (near-)one-hot forward value of the ST
    estimator, so selection entropy on this field now also measures
    structural sparsity directly, not just softmax sharpness.
  Backward compatible: sd016_cue_slot_tagger_selection="soft" by default ->
    the exact pre-H3 F.softmax(slot_logits/temp) line, unmoved and
    unmodified, just reached through the new mode dispatch's else branch
    (bit-identical by construction, verified against a manual softmax
    recomputation, not merely by a flag). Existing test_sd016_cue_slot_tagger.py
    5/5 contracts re-verified passing (no regression); existing V3-EXQ-907
    driver (from_dims wiring, not direct E1Config) dry-run completes
    unchanged. 8/8 new contracts in tests/contracts/test_sd016_h3_hard_selection.py
    PASS (direct-execution verified; the fleet's cloud workers were all
    mid-experiment at implementation time, so the pre-commit gate's routed
    remote pytest run is the authoritative CI confirmation).
  Smoke (2026-08-09): gumbel and topk both (a) produce a valid per-context
    distribution summing to 1, (b) are structurally sparse at the forward
    value (gumbel: exactly one nonzero slot per row; topk: exactly k), (c)
    still receive nonzero gradient at cue_slot_tagger through the ST
    estimator, (d) gumbel's anneal step counter advances only in train mode
    and eval-mode forward calls are exactly reproducible.
  Phased training: NOT required -- same as Path 3, the tagger (any selection
    mode) trains jointly with cue_terrain_proj on the same terrain_loss.
  MECH-094: N/A -- extract_cue_context is a waking E1 query (_e1_tick); no
    replay/simulation write surface (same as Path 3).
  Null declaration (per the H3 design, scope doc section 3): if gumbel/topk
    arms fail C1 (entropy<2.5) / C1b (context-divergence>0.1) with readiness
    met, hard selection alone does not break the saddle -- implicating the
    drive (H1) or the representation (H2), not the softness of the gate. A
    C1-pass/C1b-fail topk run (selects a fixed slot regardless of context) is
    the "constant-peaky" degeneracy C1b exists to catch.
  Validation experiment: to be queued via /queue-experiment (H3 leg of the
    GOV-FANOUT-1 portfolio; arms A0_OFF / A1_tagger_soft / A2_tagger_gumbel /
    A3_tagger_topk, inheriting the V3-EXQ-898 scaffold).
  Design doc: REE_assembly/evidence/planning/sd016_selection_fanout_portfolio_scope_staged_20260809.md
    (section 3, "Leg H3").
  See SD-016, SD-016 Path 3 (above, the soft tagger this extends), MECH-150 /
    MECH-151 / MECH-152, ARC-041, V3-EXQ-898 (the failure autopsy that
    motivated H3), GOV-FANOUT-1 (the discrimination portfolio this is one
    leg of).
