## SD-PP-3 Replay Provenance Packet (2026-09-22)
- SD-PP-3: hippocampal.replay_provenance_packet — IMPLEMENTED 2026-09-22.
  Module: `ree_core/hippocampal/replay_provenance.py`
  (`ReplayProvenancePacket`, `ReplayProvenanceRecorder`, `PROVENANCE_SCHEMA_VERSION=1`).
  One packet per replay-buffer entry, recorded at the moment `_e1_tick` appends the
  outcome state to `agent._world_experience_buffer`, bound by INDEX to that buffer so
  the consolidator can look it up for the triple it drew. A packet carries three
  distinct precision-provenance quantities read at the moment of the transition: the
  point prediction the model would have made (`pred_at_test`), the model's own
  EPISTEMIC precision about that prediction BEFORE the outcome was seen (`pi_hist`,
  `v_tot_hist`, `v_ale_hist`, `precision_source`), and the evidence-channel reliability
  at that tick (`evidence_variance_z`, `evidence_precision_z`, `sigma_obs`, `kappa`,
  `evidence_ready`) — plus the realised prediction error (`pe`) and a standardised
  epistemic surprise (`surprise = pi_hist * max(pe - noise_gain*evidence_variance_z, 0)`).
  Config: `REEConfig.use_replay_precision_provenance` (bool, default False; wired by
  the session, requires SD-PP-1 `use_observation_reliability` and SD-PP-2
  `use_world_forward_epistemic_precision` both True — the agent raises `ValueError`
  otherwise per the integration contract in
  `REE_assembly/docs/architecture/precision_provenance_substrate_spec.md` section 6).
  Data flow: `agent._e1_tick` buffer append -> `ReplayProvenanceRecorder.record` ->
  `packets[index]` -> `compute_e2_world_loss` lookup `packets[i+1]` -> SD-PP-4 gain.
  Alignment: packet `p[j]` describes the transition `(world[j-1], action[j], world[j])`;
  the training triple drawn at replay index `i` by `compute_e2_world_loss` is
  `(world[i], action[i+1], world[i+1])`, so the packet that describes it is `p[i+1]` —
  the SAME +1 offset `compute_e2_world_loss` already applies when it reads
  `action[i+1]` as "the action that led INTO `world[i+1]`" (pinned in this build against
  the actual tensors, mirroring `test_e2_world_forward_sleep_trainer.py::test_w6`).
  Recorder ordering (the whole point of the module): `epistemic.prediction_at_test`,
  then `epistemic.precision_at` (historical, no-future-info read), then
  `reliability.snapshot()`, and ONLY THEN `epistemic.observe_outcome` — so `pi_hist`
  on every packet is genuinely the precision BEFORE this outcome was seen. An
  episode-start tick (`z_prev=None`) calls NO epistemic method at all and records a
  placeholder (`has_prev=False`, `pe`/`pi_hist`/`surprise` NaN, `pred_at_test` zeros).
  Bounded ring buffer: `max_len` (default 1000), trimmed in lockstep with the agent's
  own experience buffers via `del packets[:-max_len]`, so `get(i)` by current list
  position always names the packet aligned to buffer position `i`.
  Backward compatible: disabled by default; no serialisation format change (the packet
  list is a runtime-only buffer, never checkpointed or saved).
  MECH-094: the recorder runs on WAKING ticks only; a hypothesis-tagged tick is skipped
  by the caller (`_e1_tick`), not by this module.
  Biological basis: CA1 mismatch signal bound to prediction strength rather than a bare
  prediction-error magnitude (Chen 2015); behavioural tagging and persistent eligibility
  marks that let a later, unrelated salient event retroactively strengthen a weakly
  encoded trace (Takeuchi 2016; van der Meer & Bendor 2025) — the packet's historical
  precision + realised surprise together are the artificial analogue of that tag.
  Validation experiment: V3-EXQ-1073 (reserved).
  See MECH-572, MECH-269, MECH-284/285, MECH-368/431.
