## SD-ORIENTING-DECISION-SCALE / MECH-489: Defensive-Orienting Decision Normalization -- IMPLEMENTED (2026-08-10)
- SD-ORIENTING-DECISION-SCALE: pag.defensive_orienting_response.decision_normalization --
  IMPLEMENTED 2026-08-10. Full design: `REE_assembly/docs/architecture/sd_orienting_decision_scale.md`.
  Built from `failure_autopsy_V3-EXQ-910_2026-08-10.md`, routed via `substrate_queue.json`
  (`sd_id: SD-ORIENTING-DECISION-SCALE`, `severity: corrupting`, `node_class: complicated
  (buildable)`). Fixes a bug WITHIN SD-099's already-implemented Component 4/5 block (SD-099
  itself, `ree_core/pag/defensive_orienting.py`, is UNCHANGED by this fix) -- `agent.py`
  `select_action()` was comparing `_do_harm_val` (SD-010 `z_harm_s` L2 norm -- structurally
  non-negative, persistent nonzero ambient floor) directly against `_do_benefit` (ARC-030
  `residue_field.evaluate_benefit()` RBF value -- near-zero everywhere except very close to a
  benefit center) as if the two raw magnitudes were commensurable. This structurally biased every
  override toward `withdraw` independent of actual event valence: V3-EXQ-910 logged 206 overrides,
  0 approach / 0 resume / 206 withdraw.
  Fix: both channels are z-scored against their own running distribution
  (`(value - EMA_mean) / (EMA_mean_absolute_deviation + scale_floor)`) before comparison, instead
  of comparing raw magnitudes. The two EMA baselines (`_orienting_decision_harm_mean/_mad`,
  `_orienting_decision_benefit_mean/_mad`) live at the AGENT level (`agent.py`, not
  `defensive_orienting.py`) since SD-099 deliberately keeps the gate itself free of residue-field
  reads; they update every waking tick and FREEZE while orienting is active, mirroring
  `DefensiveOrientingGate`'s own onset-baseline freeze (same rationale: prevents the triggering
  elevation from pulling its own baseline toward itself over the episode, which would blunt the
  z-score exactly when the override decision needs it). `residue_field.evaluate_benefit(z_world)`
  is now called every waking tick while the gate is enabled (previously only at override time) so
  the benefit baseline is well-formed by the time an override actually happens.
  Config: `REEConfig.orienting_decision_baseline_ema_alpha` (default `0.02`, matches the existing
  onset-baseline alphas) + `orienting_decision_scale_floor` (default `0.01`) -- both NEW, present
  at all three sites (dataclass field, `from_dims()` parameter, `from_dims()` assignment). The
  existing `orienting_decision_epsilon` default changed `0.01` -> `0.25`: it is now a z-score
  margin, not a raw-magnitude margin, and the old numeral had no valid meaning post-fix. Safe:
  the whole mechanism is `use_defensive_orienting=False` by default, and the only run that ever
  exercised the old semantics (V3-EXQ-910) is exactly what this fix corrects -- no passing
  experiment or test depended on it.
  Not a learning module: no `nn.Module`, no trainable parameters -- pure scalar EMA arithmetic,
  same category as SD-099/MECH-279/SD-069. Phased training and MECH-094 do not apply beyond
  SD-099's own existing treatment (the new code sits inside the same
  `if self.defensive_orienting is not None:` guard).
  Backward compatible: confirmed via a direct `REEAgent` construction smoke test
  (`use_defensive_orienting=False` -> `agent.defensive_orienting is None`, all new
  `_orienting_decision_*` attributes stay at their init values) and via
  `experiments/v3_exq_910_mech489_defensive_orienting_validation.py --dry-run` (both
  `orienting_off`/`orienting_on` arms complete with per-seed PASS; aggregate FAIL is the
  pre-existing short-run artifact already noted under SD-099, unrelated to this fix).
  New-feature smoke test: a synthetic `REEAgent` run with varying `obs_harm` and periodic
  `accumulate_benefit()` calls confirmed (a) the new baselines update over ticks and (b) decisions
  are no longer unconditionally `withdraw`; a direct regression case reproducing the bug's exact
  shape (harm_val=0.30 only ~1 MAD above its baseline, benefit_val=0.05 numerically smaller but
  ~8-10 MADs above ITS baseline) flips from `withdraw` (old raw-magnitude comparison) to
  `approach` (new z-scored comparison).
  Validation experiment: V3-EXQ-910a queued (script copied to
  `v3_exq_910a_mech489_defensive_orienting_decision_retest.py`, 910's own script untouched; adds
  a NEW pre-registered criterion (c) `C_decision_alignment_non_degenerate` -- the actual
  acceptance test for this fix; criteria (a)/(b) still computed for context but already stand as
  fairly falsified per the autopsy and are not re-litigated), `supersedes: V3-EXQ-910`.
  See MECH-489, SD-099 (parent gate, unchanged), `failure_autopsy_V3-EXQ-910_2026-08-10.md`.
