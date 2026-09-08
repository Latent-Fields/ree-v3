## SD-025: hippocampal_module.curiosity_drive -- IMPLEMENTED (2026-07-16)
- SD-025: hippocampal_module.curiosity_drive -- IMPLEMENTED 2026-07-16.
  The SECOND component of ARC-057 (approach-emergence): an information-seeking bias in
  hippocampal CEM trajectory scoring that favours regions of higher REPRESENTATIONAL
  DENSITY in the SD-024 benefit RBF map. Modules: ree_core/hippocampal/curiosity.py
  (FamiliarityTracker) + ree_core/hippocampal/module.py (HippocampalModule).
  novelty(z) = density(z) * (1 - familiarity(z)); _score_trajectory subtracts
  curiosity_weight * mean_over_trajectory(novelty) from the terrain score (CEM
  minimises -> lower = better, same convention as wanting_weight). density =
  compute_representational_density (the SD-024 WEIGHT-INDEPENDENT active-center count ->
  the drive follows representational QUALITY, not a positive-valence gradient).
  familiarity = FamiliarityTracker, a proximity-weighted visit-count EMA (soft
  visitation KDE, clamped [0,1]) that rises on revisit so novelty decays there
  (anti-perseveration); it is instantiated ONLY when curiosity_weight > 0.
  Config (HippocampalConfig): curiosity_weight (master, default 0.0 -> tracker never
  built, scoring untouched, update_familiarity a no-op -> bit-identical OFF),
  familiarity_ema_alpha (0.01), use_curiosity_familiarity (True; set False for the
  density-only ablation), familiarity_bandwidth (1.0). Data flow: benefit RBF density
  (SD-024) -> compute_representational_density [read-only] -> novelty=density*(1-familiarity)
  -> _score_trajectory terrain_score -= curiosity_weight*novelty -> CEM elite selection.
  Waking visit -> agent.sense() -> update_familiarity(z_world, is_waking=not hypothesis_tag).
  Backward compatible: curiosity_weight=0.0 -> bit-identical OFF (full pytest tests/ 1488
  passed). Phased training: NOT REQUIRED (no encoder head; read + EMA state only).
  MECH-094: familiarity is real memory state -> update_familiarity advances on WAKING
  visits ONLY (agent gates on hypothesis_tag, identical to the MECH-314a visitation
  buffer); the density read during CEM scoring writes no memory.
  7 contracts in tests/contracts/test_sd025_curiosity_drive.py.
  SCOPE NOTE: the SUBSTRATE is buildable now on SD-024, but the full ARC-057 ecological
  approach-emergence claim (SD-024 x SD-025 interaction) is ENV-CONSTRAINED -- the
  CausalGridWorld cannot faithfully test it (a cell is a cell; nothing more to discover
  at higher resolution; see claims.yaml ARC-057 SUBSTRATE CONSTRAINT). The validation is
  scoped to the DRIVE MECHANISM (does curiosity propagate into CEM selection toward
  higher-density regions? -- the propagation MECH-111's broken broadcast-novelty->E3 path
  could not achieve, EXQ-141b/590a), NOT the interaction claim.
  Validation experiment: V3-EXQ-767 (curiosity ON vs OFF; leg-1 propagation: CEM
  selection biased toward the dense cluster only when ON; leg-2 anti-perseveration:
  familiarity discount attenuates density-following on revisit).
  Design doc: REE_assembly/docs/architecture/sd_024_da_modulated_rbf_density.md#curiosity-drive
  See SD-024 (the density substrate it reads), ARC-057 (the approach architecture, env-
  constrained), MECH-111 (curiosity/novelty-drive grounding; distinct broken path).
