## SD-099 / MECH-489: Defensive-Orienting Response -- IMPLEMENTED (2026-08-09)
- SD-099 (architecture) / MECH-489 (mechanism): pag.defensive_orienting_response --
  IMPLEMENTED 2026-08-09. Full design + trigger-channel derivation:
  `REE_assembly/docs/architecture/sd_094_defensive_orienting_response.md`. Built from
  `observational_review_V3-EXQ-906b_2026-08-09.md` Sections 11b/11d (design), 12h (trigger
  calibration risk), 12j (adjacent-claim reconciliation). Closes the gap that a tree-wide grep for
  `orient|reorient|startle|fright` across `ree_core/` previously returned nothing.
  `ree_core/pag/defensive_orienting.py` (`DefensiveOrientingGate` / `DefensiveOrientingConfig` /
  `DefensiveOrientingOutput`) -- non-trainable, pure-scalar-arithmetic gate matching the
  `PAGFreezeGate` (MECH-279) pattern. Deliberately a SEPARATE gate, not a modification of
  `freeze_gate.py`: the two freeze causes (MECH-279 chronic/accumulated-suffering vs MECH-489
  phasic/unidentified-onset) are architecturally distinct and are composed via OR at the
  action-constraint site in `agent.py select_action()`, so MECH-279's own behaviour is unchanged
  bit-for-bit.
  Config: `REEConfig.use_defensive_orienting` (default `False`, bit-identical off -- gate not
  instantiated) + 11 tunables (`orienting_surprise_ema_alpha`, `orienting_harm_s_ema_alpha`,
  `orienting_surprise_onset_delta`, `orienting_harm_s_onset_delta`,
  `orienting_confidence_rise_rate`, `orienting_confidence_floor_rise`,
  `orienting_sufficiency_threshold`, `orienting_max_duration`, `orienting_decision_epsilon`,
  `orienting_decision_bias_scale`, `orienting_post_override_bias_ticks`), all present at all
  three sites (`REEConfig` dataclass field, `from_dims()` parameter, `from_dims()` assignment --
  see `[memory] reference-reeconfig-from-dims-silent-kwargs`). No separate no-op action-class
  knob -- deliberately reuses `pag_freeze_noop_action_class` so both freeze causes constrain to
  the same "hold still" action.
  **Trigger (resolves 12h's finding that `residue_surprise > p90` under-fires on the ground-truth
  injected events it exists to catch):** a positive-derivative onset detector over TWO
  already-phasic channels, not one absolute threshold -- `residue_surprise` (MECH-205 unsigned
  `VALENCE_SURPRISE`, catches `external_hazard_injected` / delayed `world_rule_shift_occurred`)
  OR `z_harm_s` norm (SD-010 sensory-discriminative harm, catches `limb_damage_injected` --
  Section 12g established this is the clean-phasic channel a trigger mechanism would want, NOT
  the chronic `z_harm_a` MECH-279/vigor already read). Each channel's rolling EMA baseline is
  FROZEN while orienting is active (confirmed necessary by an inline smoke test during the build:
  without freezing, the baseline chases a sustained non-decaying signal upward and the freeze
  silently "auto-expires" via baseline creep -- exactly the fixed-timer-in-disguise behaviour
  11b step 2/4 rules out).
  **Identification-confidence dynamics** (the orienting-reflex return path to planned action, 11b
  step 3 -- load-bearing, not deferred): rises toward `orienting_sufficiency_threshold` at a rate
  scaled by how much the triggering channel's excess-over-frozen-baseline has decayed, not on a
  clock. Verified: a decaying spike resolves in a handful of ticks; a sustained, non-decaying
  elevation never overrides across 200 ticks (confidence stalls at/near 0).
  **Action decision (approach/withdraw/resume, on override):** resolved by the AGENT (not the
  gate, mirroring `freeze_gate.py`'s own "pure arithmetic over scalars" scope) by comparing
  `residue_field.evaluate_benefit(z_world)` (ARC-030, always available) against `z_harm.norm()`
  (SD-010, always available) at the current z_world -- deliberately NOT the MECH-307
  `VALENCE_POSITIVE_SURPRISE`/`NEGATIVE_SURPRISE` split, which is off by default and would leave
  this step inert in most configurations. Expressed as a per-candidate `score_bias`
  (`E3Selector.select()`'s existing directed-bias hook, composed additively into `dacc_score_bias`
  via the same idiom every other agent-level bias contributor in `select_action()` already uses)
  toward/away from `self._orienting_trigger_z_world` (captured at trigger time), using each
  candidate's `Trajectory.world_states[-1]`. Degrades gracefully to inert (no bias, no crash) when
  `world_states` are not tracked (SD-003 attribution not wired) -- confirmed via smoke test.
  Data flow: `z_harm (current tick) + residue_surprise (cached from the PREVIOUS tick's
  update_residue()) -> DefensiveOrientingGate.tick() -> {trigger capture z_world, override
  resolve decision + score_bias, orienting_active -> motor no-op}`.
  MECH-094: `simulation_mode=True` returns a zeroed output and updates NO internal state
  (baselines, confidence, counters all frozen) -- mirrors `freeze_gate.py` exactly.
  Not a learning module: no `nn.Module`, no trainable parameters -- phased training not
  applicable (same category as MECH-279 and SD-069 `PhasicSurpriseBurst`).
  Backward compatible: disabled by default; confirmed via a stashed-vs-unstashed A/B on
  `experiments/v3_exq_906b_full_stack_observational_fishtank.py --dry-run` (same PASS/FAIL
  pattern with and without the change -- the aggregate dry-run FAIL is a pre-existing short-run
  artifact, unrelated to this SD).
  Validation experiment: queued via `/queue-experiment` (see queue for the ID) -- ablation
  ON/OFF against the Section 12h ground-truth injected events
  (`limb_damage_injected`/`external_hazard_injected`/`world_rule_shift_occurred`) and the Section
  4/12b coupling baselines (P(moved@t+1|spike)=44.3% vs 24.0%;
  P(mode-change@t+1|spike)=15.4% vs 11.1%).
  See MECH-279 (sibling, disjoint trigger, NOT modified), MECH-205 (surprise source), MECH-395 /
  MECH-482 / MECH-483 (adjacent orienting-territory claims, cross-referenced not duplicated --
  see the SD doc's Related Claims and `docs/claims/claims.yaml` MECH-489 `depends_on`), SD-010 /
  SD-011 (the two trigger/decision channels), SD-014 / ARC-036, ARC-030 / MECH-117 (resolution-read
  channels), SD-037 (explicitly distinct override channel), SD-069 (sibling non-trainable phasic
  regulator pattern, different consumer).
