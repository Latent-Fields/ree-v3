## SD-024: hippocampal_module.da_modulated_rbf_density -- IMPLEMENTED (2026-07-16)
- SD-024: hippocampal_module.da_modulated_rbf_density -- IMPLEMENTED 2026-07-16.
  Built as the DIAGNOSTIC instrument that RESOLVES MECH-232 (DA representational
  expansion), NOT a feature gated behind it. Modules: ree_core/residue/field.py
  (RBFLayer, ResidueField) + ree_core/hippocampal/module.py (HippocampalModule).
  DA at reward encounters allocates a local CLUSTER of RBF centers (representational
  expansion) + optional finer per-center bandwidth on the BENEFIT terrain -- higher
  information density / sharper place fields -- WITHOUT writing an explicit
  positive-valence gradient. Modulates ONLY benefit_rbf_field; harm/safety fields keep
  single-center standard-bandwidth allocation (MECH-233 asymmetry preserved).
  RBFLayer: per_center_bandwidth buffer (default off -> byte-identical scalar reads);
  add_residue_cluster() [n = 1 + int(da_signal * allocation_scale) jittered centers,
  per-center bandwidth floored at 0.5*base]; compute_local_density() [WEIGHT-INDEPENDENT
  proximity-weighted active-center count]. ResidueField: benefit field built per-center
  + optional da_benefit_num_centers capacity when master on; accumulate_benefit(...,
  dopamine_signal=) routes to the cluster path; compute_benefit_density() wrapper.
  HippocampalModule.compute_representational_density() read-through (SD-025 hook).
  Config: ResidueConfig.use_da_modulated_rbf_density (master, default False),
  da_allocation_scale (0.0), da_jitter_radius (0.1), da_bandwidth_narrowing (0.0),
  da_benefit_num_centers (None). Data flow: reward contact -> accumulate_benefit(
  dopamine_signal=benefit*drive) [MECH-094 hypothesis_tag gate] -> add_residue_cluster
  -> benefit_rbf_field density up -> compute_local_density (read) -> [SD-025 curiosity
  drive, downstream] -> approach. THE MECH-232 DISCRIMINATOR: density is weight-INDEPENDENT,
  so DA raises density even when evaluate_benefit (weight sum) is held flat -> approach
  from representational QUALITY alone, not a valence tag. Backward compatible: all
  defaults no-op -> bit-identical OFF (full pytest tests/ 1475 passed). Phased training:
  NOT REQUIRED (no encoder head; allocation + read only). MECH-094: DA expansion inherits
  accumulate_benefit's hypothesis_tag gate (replay cannot expand).
  13 contracts in tests/contracts/test_sd024_da_modulated_rbf_density.py.
  Validation experiment: V3-EXQ-766 (MECH-232 DA-ON vs DA-OFF: leg-1 representational
  expansion at reward locations; leg-2 CRUX approach-without-explicit-gradient). PASS
  promotes MECH-232 candidate->provisional; FAIL refutes.
  Design doc: REE_assembly/docs/architecture/sd_024_da_modulated_rbf_density.md
  See MECH-232 (the mechanism it resolves), MECH-233 (asymmetry preserved), ARC-057
  (the approach-emergence architecture), SD-025 (curiosity_drive it unblocks downstream).
