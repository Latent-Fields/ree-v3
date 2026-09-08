## SD-024 LIVE-PATH PRODUCER: residue.benefit_terrain_live_producer -- IMPLEMENTED (2026-07-20)
- SD-024 LIVE-PATH PRODUCER: residue.benefit_terrain_live_producer --
  IMPLEMENTED 2026-07-20. Closes a four-day gap in the SD-024 landing of
  2026-07-16, which built the benefit terrain and its READ but never wired a
  WRITER into the agent loop.
  THE DEFECT: `ResidueField.accumulate_benefit` had NO CALLER anywhere in
  `ree_core/`. Its only two write sites into `benefit_rbf_field`
  (`residue/field.py:673`, `:682`) sit inside that one method, and agent.py
  called `update_valence` / `accumulate` / `accumulate_safety` /
  `evaluate_safety` -- never `accumulate_benefit`. Measured consequence on a
  real warmup_train loop (darwin-arm64, curiosity_weight=0.5), with BOTH
  `benefit_terrain_enabled` and `use_da_modulated_rbf_density` True:
  `benefit_rbf_field.active_mask.sum() == 0`, `num_benefit_events == 0.0`;
  `RBFLayer.compute_local_density` early-returns zeros on an empty active mask
  (`field.py:273`) so `compute_representational_density` returned exactly 0.0;
  therefore `HippocampalModule._curiosity_bonus` (`hippocampal/module.py:870-885`)
  computed `novelty = density * (1 - familiarity) = 0` and returned 0.0 on all
  14432 live calls. The `use_curiosity_familiarity` True/False ablation was
  BIT-IDENTICAL, confirming familiarity was not the binding constraint. Net: the
  SD-025 curiosity drive contributed EXACTLY ZERO to CEM trajectory scoring in
  every live agent run.
  Modified: `REEAgent.update_z_goal` in `ree_core/agent.py` (producer block).
  Config: `ResidueConfig.benefit_terrain_live_producer` (default False; set True
  to enable) + `ResidueConfig.benefit_live_producer_threshold` (default 0.1,
  the consummatory-contact gate, mirroring `liking_threshold`).
  Data flow: reward contact -> `update_z_goal(benefit_exposure, drive_level)` ->
  `ResidueField.accumulate_benefit(z_world, benefit_magnitude=benefit_exposure,
  dopamine_signal=benefit_exposure * drive_level)` -> `benefit_rbf_field` ->
  `compute_benefit_density` -> `compute_representational_density` ->
  SD-025 `_curiosity_bonus` -> CEM scoring.
  SEPARATE FLAG, NOT `benefit_terrain_enabled` -- deliberate: V3-EXQ-767/767a set
  `benefit_terrain_enabled=True` and populate the terrain THEMSELVES via direct
  `rf.accumulate_benefit()` calls (767a lines 236, 238, 305). Gating the live
  producer on that existing flag would silently double-populate those in-vitro
  designs on re-run. 767/767a remain valid in-vitro validations of the drive
  mechanism; they simply never established live-path efficacy.
  PLACEMENT -- read this before moving the block: it sits ABOVE
  `update_z_goal`'s `if self.goal_state is None ... return` guard, not below.
  The default config has `goal_state = None`, so a block after the guard never
  runs; the benefit terrain is a ResidueField concern and SD-024 has no
  GoalState dependency, so gating it there would ship a SECOND instance of the
  same no-producer defect. Not hypothetical -- the block was first written
  after the guard and contracts C2/C4/C5 failed at 0 active centers.
  `dopamine_signal` uses BASE `drive_level`, not `pacc.effective_drive` nor the
  SD-037 override-amplified value: those are goal-SEEDING gains, whereas the
  SD-012 phasic signal `accumulate_benefit` documents is
  `benefit_magnitude * drive_level`.
  ALSO IN THIS LANDING: `HippocampalConfig.familiarity_bandwidth` 1.0 -> 0.20.
  `FamiliarityTracker.query` (`hippocampal/curiosity.py:72-100`) is a CLAMPED SUM
  over anchors, not a normalised average, and the same constant is the
  association threshold in `update()` (`thresh_sq = bw*bw`, `:115`), so at 1.0
  only ~3 anchors go active and their near-unit weights pin the clamp. This did
  not bite while density was identically zero, but becomes load-bearing the
  moment it is not. V3-EXQ-786a swept it on real z_world: 0.05 -> +0.019,
  0.10 -> +0.103, 0.20 -> +0.171, 0.30 -> +0.065, 0.50 -> -0.053 (INVERTED),
  1.00 -> +0.000 -- the old default was the one value measured at exactly zero
  effect. Same root cause as SD-067's dedicated safety bandwidth (the shared
  1.0 is ~15x too wide for the z_world residual scale). Consulted only when
  `curiosity_weight > 0` (default 0.0), so no-op for every default config.
  Backward compatible: producer disabled by default; full suite 2065 passed.
  No phased training (no new encoder head).
  MECH-094: the producer reads `hypothesis_tag` off the live latent rather than
  hardcoding False, so a future replay/DMN caller cannot build benefit terrain
  by accident; `accumulate_benefit` applies the gate internally too.
  Contracts: `tests/contracts/test_sd024_benefit_terrain_live_producer.py`
  (7, C1-C6), which assert through the real REEAgent API against a real
  CausalGridWorldV2 episode loop and NEVER call `accumulate_benefit` directly --
  the 13 pre-existing SD-024 contracts all populate the field themselves, which
  is precisely why a missing producer was invisible to them.
  LANDING NOTE: the substrate edits (agent.py, config.py) were swept into
  commit `8ac193d` by a concurrent MECH-204 session's whole-file write; the
  contract test landed separately in `7f16f25ceb`. Content is complete and
  verbatim -- see CLAUDE.md "Read-modify-write contamination" remedy (a)/(a2).
  Validation experiment: V3-EXQ-795 queued 2026-07-20 (diagnostic; producer ON vs
  OFF at identical seeds, four legs: producer / density / DRIVE / selection
  authority). It never calls accumulate_benefit except in its P0 readiness
  control, so each arm's terrain can only come from the agent's own reward
  contacts. Smoke: ARM_OFF reproduces the defect signature exactly (0 centers,
  density 0.0, bonus 0.0); ARM_ON 1 center / density 0.997 / bonus 0.064.
  See SD-024, SD-025, MECH-232, ARC-030, ARC-057, MECH-117.
