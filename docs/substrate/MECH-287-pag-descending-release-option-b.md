## MECH-287 PAG Descending Release -- Option B (2026-09-25)
- MECH-287 option B: hippocampal-invalidation -> PAG freeze-EXIT descending release -- IMPLEMENTED 2026-09-25.
  Files: ree_core/pag/freeze_gate.py (PAGFreezeGateConfig.alpha_descending,
  tick(descending_release=...), PAGFreezeGateOutput.descending_release);
  ree_core/hippocampal/module.py (per-step invalidation drive accumulators +
  consume_invalidation_drive()); ree_core/agent.py (trace in sense(), consumer at
  the MECH-279 PAG tick, pag_descending_release_diagnostics()).
  Config: REEConfig.use_pag_descending_release (default False; set True to enable,
  requires use_pag_freeze_gate), pag_descending_release_alpha (1.0),
  pag_descending_release_decay (0.95 per env step), pag_descending_release_source
  ("invalidation" = T3 broadcast resets + H hysteresis resets; "broadcast" = T3 only).
  Data flow: anchor INVALIDATION in sense() (T3: a MECH-287 broadcast that marked an
  ACTIVE anchor inactive, weighted by strength; H: a MECH-284/269 hysteresis reset;
  ordinary boundary remaps and FIFO evictions do NOT count) -> trace
  r = max(r * decay, min(1, drive)) -> PAG exit_threshold *= (1 + alpha * r) at the
  next E3 tick. Exit only; z_harm_a and freeze entry untouched.
  Backward compatible: disabled by default; with the switch on and alpha 0 also
  bit-identical, so a driver may run warmup at alpha 0 and set
  agent.pag_freeze_gate.config.alpha_descending at eval entry.
  Biological basis: hippocampal context -> mPFC -> l/vlPAG descending control of
  freezing (Rozeske et al. 2018 Neuron; Sotres-Bayon et al. 2012; Maren, Phan &
  Liberzon 2013).
  Measured 2026-09-25: the T3 half is effectively redundant on this substrate. A
  boundary's own dual-trace remap deactivates the segment_id_old anchor BEFORE
  apply_invalidation_broadcasts_to_regions runs, so a real broadcast finds no
  active target. The live reach is broadcast -> MECH-284 staleness -> H reset.
  An untrained V3-EXQ-1097 D_BOTH_ON rollout (2 x 200 steps) produced 0 T3, 0 H
  and 0 freeze commits: reach-in-regime is unmeasured (Stage 0 of the design).
  Phased training required: no (no learning).
  Contracts: tests/contracts/test_pag_descending_release.py (C1 config/bit-identity,
  C2/C3 D2 reach, C4 source exclusion, C5 liveness).
  Validation experiment: NOT queued -- Stage 0 precondition gate owed first.
  Design: REE_assembly/evidence/planning/mech287_anchor_freeze_exit_design_20260925.md
