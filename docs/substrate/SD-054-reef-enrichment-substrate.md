## SD-054: Reef Enrichment Substrate (2026-05-04)
- SD-054: environment.reef_enrichment_substrate -- IMPLEMENTED 2026-05-04.
  (Note: this entry was originally labelled SD-050 in error; SD-050 is the
  Suffering-Derivative Comparator per claims.yaml. Renamed to SD-054 on
  2026-05-08; SD-053 is informally reserved for a sustained-drive claim.)
  Module: ree_core/environment/causal_grid_world.py (CausalGridWorldV2).
  Monostrategy-breaking behavioral-diversity substrate. Adds reef safe zones and
  food-attracted hazard drift to CausalGridWorldV2. Motivated by the observation
  that monomodal policy prevents balanced agent-vs-env event distributions (SD-029
  C2/C3 blocker). Creates two behavioral attractors -- "flee to reef" vs "forage"
  -- to break the single fixed route the agent otherwise exploits.
  Two mechanisms:
    Reef safe zones: circular patches (Manhattan radius) around corner-adjacent
      centers where hazards cannot enter and no food/hazards spawn. Agent CAN enter
      reef cells (attractor for safety-seeking). Static scent gradient: 5x5
      normalized Manhattan-decay kernel appended to world_state when reef_enabled=True.
      world_obs_dim: 250 -> 275 (+ 25 reef_field_view dimensions).
    Food-attracted hazards: during _drift_hazards(), with probability
      hazard_food_attraction, each hazard biases its random walk step toward the
      nearest food cell rather than sampling uniformly. Default 0.0 (bit-identical OFF).
      Makes foraging actively more dangerous; paired with reef safety creates a
      structurally richer strategy space.
  Config (CausalGridWorldV2 __init__):
    reef_enabled (bool, default False) -- master switch for reef zones.
    n_reef_patches (int, default 3) -- number of circular reef patches.
    reef_patch_radius (int, default 2) -- Manhattan radius of each patch.
    hazard_food_attraction (float, default 0.0) -- probability hazard steps
      toward nearest food. 0.0 = legacy random walk (bit-identical OFF).
  State: self._reef_cells: Set[Tuple[int,int]] (populated at reset());
    world_obs_dim property returns 275 when reef_enabled else 250.
  Observation: obs_dict["reef_field_view"] [25] static Gaussian-decay view of
    reef proximity; appended to world_state in _get_observation_dict().
  Backward compatible: reef_enabled=False and hazard_food_attraction=0.0 by default;
    _reef_cells is empty set; world_obs_dim=250; existing experiments unaffected.
  V3-EXQ-521 substrate readiness diagnostic PASS 7/7 criteria (2026-05-04):
    ARM_0 baseline obs_dim=250 reef_cells=0; ARM_1/ARM_2 obs_dim=275 reef_cells=33;
    0 reef violations in ARM_1/ARM_2 across 30 ep x 200 steps x 3 seeds;
    ARM_2 food_dist=2.057 vs baseline 3.790 (46% reduction); entropy not collapsed
    (ARM_2=4.049 >= 0.7 x baseline 4.557=3.190); agent reef_visits=1987 in ARM_1.
  Biological analog: coral reef refugia as behavioral partitioning substrate --
    distinct safe-zone vs. foraging patch microhabitats force context-dependent
    strategy selection rather than single-template exploitation.
  MECH-094: not applicable (env observation stream, not replay content).
  No trainable parameters. Pure env substrate.
  Validation experiment: V3-EXQ-521 PASS 2026-05-04 (substrate readiness diagnostic).
    V3-EXQ-522 (monostrategy-breaking behavioral diversity test) is the next experiment,
    gated on V3-EXQ-521 PASS.
  See SD-029 (self_attribution.comparator_z_harm_s -- SD-054 substrate unblocks C2/C3
    behavioral diversity measurement), MECH-256 (SD-029 successor), SD-023 (env
    gradient texture -- parallel substrate enrichment pattern).
