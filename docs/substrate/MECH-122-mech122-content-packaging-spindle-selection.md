## MECH122-CONTENT-PACKAGING-SPINDLE-SELECTION (MECH-122 content-packaging half; V3 proxy) -- IMPLEMENTED (2026-08-02)
- MECH122-CONTENT-PACKAGING-SPINDLE-SELECTION (MECH-122 content-packaging half; V3 proxy) --
  IMPLEMENTED 2026-08-02 (IGW-20260801-197, `/implement-substrate`).
  `ree_core/agent.py` `REEAgent.run_sws_schema_pass()`; `ree_core/latent/theta_buffer.py`
  `ThetaBuffer.set_consolidation_mode()` / `.consolidation_summary()` (unchanged, pre-existing).
  Config: `REEConfig.use_mech122_spindle_content_selection` (default `False`, bit-identical off)
  + `REEConfig.mech122_spindle_selection_gain` (default `1.0`). Three-site wiring (dataclass field,
  `from_dims` signature, `from_dims` assignment).
  Gap this closes: `ThetaBuffer.set_consolidation_mode()`/`.consolidation_summary()` already
  implemented a V3 proxy for spindle-mediated content packaging (MECH-122) per their own
  docstrings but were never called from anywhere in `ree_core/` (confirmed by grep) -- dead code.
  `failure_autopsy_V3-EXQ-861_2026-08-01` traced MECH-180's `spindle_density` DV
  (`mean_sws_new_slot_diversity`) failing to track novelty/MEL, while its two sibling DVs
  (`sws_power`, `replay_rate`) succeed, to exactly this gap: REE's SWS pass writes sampled
  prototype vectors into `ContextMemory` via a plain, undifferentiated EMA-blend with no
  spindle-analog content-SELECTION step, so there is no channel through which higher novelty
  could produce more differentiated writes.
  Data flow: `ThetaBuffer._z_world_buffer` (rolling, populated every E1 waking tick, unconditional
  on any flag) -> `set_consolidation_mode(True)` -> `consolidation_summary()` (recency-weighted
  z_world reference, reverse-temporal-order theta-packaging) -> per-prototype cosine-novelty
  weight against that reference in `run_sws_schema_pass()`'s sampling loop -> blended
  `z_world_selected` (habitual content pulled toward the reference; novel content keeps its own
  direction) -> `e1_input` -> `ContextMemory.write()` -> `set_consolidation_mode(False)`.
  Why this design and not a uniform scale: `sws_slot_diversity` (the pre-existing metric) and
  the redesigned `mean_sws_new_slot_diversity` DV are both COSINE similarity over NORMALISED
  memory rows, which is scale-invariant to any uniform per-write multiplier -- the existing
  MECH-272 `anchor_weight` lever structurally cannot move this DV no matter how it is tuned. Only
  a DIRECTIONAL, per-prototype-selective transform can.
  Why this design differs from V3-EXQ-246's prior MECH-122 Phase-3 proxy: EXQ-246 (2026-04-05)
  already tried `consolidation_summary()` -> `ContextMemory.write()`, but as a single extra
  post-hoc write appended AFTER the SWS+REM cycle -- and measured ZERO effect on its downstream
  behavioural metric across both runs / all 3 seeds (`claims.yaml` MECH-122
  `evidence_quality_note`). This landing instead modulates every one of the
  `sws_consolidation_steps` schema-installation writes DURING the pass itself, which is a
  structurally different (and, per the diversity-metric argument above, actually load-bearing)
  intervention on the DV the current gap concerns.
  Backward compatible: `use_mech122_spindle_content_selection=False` (default) skips the whole
  block; `run_sws_schema_pass()` is bit-identical to pre-landing behaviour. Not a learning
  module -- pure arithmetic on existing tensors, no new parameters, no phased training, MECH-094
  N/A (no memory write outside the pre-existing ContextMemory write target; no residue/anchor
  touch).
  Two new metrics on `run_sws_schema_pass()`'s return dict (always present, values 0.0 when the
  flag is off or no theta reference was available): `sws_spindle_selection_applied` (1.0 if
  selection ran this pass), `sws_spindle_selection_mean_weight` (mean per-prototype selection
  weight, in [0, 1]).
  Contracts: `tests/test_flag_inertness.py`
  `test_mech122_spindle_content_selection_fires_and_differentiates_writes` (OFF-inert: zero
  weight/applied, byte-identical to plain `sws_enabled=True`; ON-live: in-range weight AND a
  DIFFERENT post-pass `ContextMemory` tensor than OFF on the same buffered content -- the bar
  the naive EXQ-246 proxy failed to clear) and
  `test_mech122_spindle_content_selection_gain_zero_collapses_to_off` (gain=0.0 sanity check).
  Flag registered in `PROBED`. Full `test_flag_inertness.py` (27) and
  `test_sleep_phase_c_routing_consumer.py` + `test_multi_content_theta_packet.py` (14) pass
  unmodified. `--dry-run` smoke test of `experiments/v3_exq_861_mech180_ecological_novelty_sleep_consolidation_decoupled_diversity.py`
  (unmodified, flag not exercised) runs to completion unaffected.
  Validation experiment: none queued this session (substrate-build scope only, per this IGW
  item's instructions) -- recommended follow-on is a `/queue-experiment` re-run of the
  V3-EXQ-861 driver family with `use_mech122_spindle_content_selection=True` against the
  `spindle_density` DV. `substrate_queue.json` entry status moved
  `pending_implementation` -> `implemented_pending_validation`.
  MECH122-SENSORY-GATING-OFFLINE-PROTECTION (the OTHER half of MECH-122) is a SEPARATE,
  still-unimplemented `complex (probe-gated)` substrate_queue entry -- not touched by this
  landing.
  See MECH-122, MECH-180, `REE_assembly/docs/architecture/sleep/offline_phases.md`,
  `failure_autopsy_V3-EXQ-861_2026-08-01`.
