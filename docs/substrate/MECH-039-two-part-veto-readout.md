## MECH-039 Two-Part Veto Readout: interrupt part (CeA + SD-037) vs control part (freeze + habenula + MECH-449), instrument only (2026-09-25)

- MECH-039: control_plane.channel_mode_landscape.veto_readout -- IMPLEMENTED 2026-09-25
  (instrument only; no behaviour). User decision rec-20260925-3b215584 (option 1) on
  `REE_assembly/evidence/planning/claim_synthesis_MECH-039_20260925.md` as revised by its
  red-team `claim_synthesis_MECH-157-039_redteam_20260925.md` (578e61dd0a).
  Chip: `chip-20260925-mech039-two-part-veto-readout`.
  `ree_core/agent.py`: `REEAgent._record_veto_readout` (called at all three
  `select_action()` exits: `between_e3`, `e3_shortcircuit`, `e3`),
  `_veto_readout_note_habenula` (called in `update_residue()` where the habenula abort
  fires), `_veto_readout_episode_boundary` (called in `reset()`), and the accessors
  `get_veto_readout()` / `reset_veto_readout()`.
  Config: `REEConfig.use_mech039_veto_readout` (default False; set True to enable) and
  `veto_readout_override_onset_threshold` (0.5). Both plumbed through `from_dims`
  (field + signature + assignment).
  The readout arms NO producer: each producer keeps its own flag.
- **The two parts.**
  - INTERRUPT = the only veto-shaped producers wired INTO the SalienceCoordinator, and so
    the only ones that can force a multi-channel transition: SD-035 CeA
    (`cea_mode_prior`, `cea_fast_prime`, `urgency_fire`; MECH-046) and SD-037
    `override_signal`. CeA is active when `urgency_fire` or `mode_prior != 0` (it is
    exactly 0 at rest). The override signal is an EMA'd sigmoid and never exactly 0, so it
    is active at `>= veto_readout_override_onset_threshold`.
  - CONTROL = the local vetoes, none with a coordinator path (red-team D5): MECH-279
    `PAGFreezeGate` `freeze_active`, the ARC-108 JOB-2(d) habenula de-commit abort (a
    post-commit abort, folded in post-action), and MECH-449 safety No-Go (per-tick delta of
    `n_safety_nogo_applied`, plus `gng_safety_all_unsafe`: every candidate crossed the floor,
    so the selector's last-resort fallback commits to a vetoed candidate and the veto is
    overridden). The MECH-449 fields read `gng_safety_diagnostics` state and are populated
    only when `E3Config.use_gng_endogenous_safety` is on. **Do not arm the MECH-449 part in
    an experiment before V3-EXQ-1090 reports.** The MECH-489 orienting arrest is recorded
    (`orienting_arrest_active`) but is not in either part.
- **Per-step record** (`get_veto_readout()["last"]`): the fields above, per-part `*_active` /
  `*_onset` (rising edges), coordinator state on the step (`coord_ticked`,
  `coord_current_mode`, `coord_mode_switch_trigger`, `coord_operating_mode`), per-part
  latency when this step's switch resolved an onset, and a `channels` snapshot of the
  other MECH-039 channels that already exist (`arousal_volatility` =
  `e3.volatility_estimate`, `e3_steps_per_tick`, `beta_elevated`, `commit_readiness`).
  The record for a step is complete once `update_residue()` has run for that step.
- **Summary** (`["summary"]`): onset counts per part and per producer, active-step counts,
  mode-switch counts (and the switches with no pending onset), and the
  **onset -> next `mode_switch_trigger` latency** per part, in agent steps and in
  coordinator ticks. Latency runs from the FIRST onset since the last switch. An onset still
  pending at `reset()` is right-censored (`n_censored_onsets`). Counts persist across
  episodes; `reset_veto_readout()` clears them.
- **MECH-046 shared arm.** The interrupt-part latency IS MECH-046's registered DV
  (time-to-mode-switch on threat onset, CeA-ON vs amygdala-OFF). Record it once and tag the
  run to both claims. Do not build a second instrument.
- **Timing caveat.** CeA and SD-037 tick in `sense()` (every step), but the coordinator and
  the freeze gate tick only on the full E3 path (`site == "e3"`). So switch latency is
  quantised to E3 ticks (`e3_steps_per_tick`, default 10, varied by MECH-093); the
  coordinator-tick latency is reported alongside for that reason.
- **Circularity (red-team D6).** With the coordinator ON, the MECH-261 replay/learning
  scheduling channel is partly derived from the coordinator's label. A MECH-039 clustering DV
  must exclude coordinator-derived channels or use a coordinator-OFF arm. This readout's
  coordinator fields are for the forced-transition timing contrast only.
- Data flow: producers (unchanged) -> read after they were computed this step ->
  `_veto_readout_last` / `_veto_readout_state` -> `get_veto_readout()` -> experiment driver.
  No consumer inside ree_core, by design (instrument).
- Backward compatible: default off, every call site gated on the flag. Bit-identical
  measured: action trace + final parameter hash equal for origin/main, this build OFF, and
  this build ON, with producers off and with CeA + SD-037 + coordinator + PAG freeze +
  closure/habenula armed. Pinned by `tests/contracts/test_mech039_veto_readout.py` (C3).
  Registered in `tests/test_flag_inertness.py` KNOWN_UNPROBED (telemetry by design).
- Note: MECH-094 does not apply here. The readout reads waking select/update state only and writes no memory.
- Phased training required: no.
- Validation experiment: EXP-0787 (MECH-039), with a precondition gate that each part
  actually fires under hazard onset. See the queue entry.
- See MECH-039, MECH-046, MECH-040, MECH-053 (the habenula abort is the post-commit boundary
  MECH-053's R6 disposition rejected for a PRE-commit veto; MECH-039 reads it as an
  interrupt of the current regime), MECH-279, MECH-449, SD-035, SD-037, ARC-108.
