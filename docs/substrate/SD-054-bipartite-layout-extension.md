## SD-054 bipartite layout extension (2026-05-11)
- SD-054 bipartite layout: environment.reef_bipartite_spawn_partition -- IMPLEMENTED 2026-05-11.
  Module: ree_core/environment/causal_grid_world.py (CausalGridWorldV2).
  Extends SD-054 with a geometric bipartite spawn structure so reef-vs-forage
  trajectories require categorically-different first-action argmaxes by
  construction. Resolves the upstream CEM-candidate-distinguishability
  bottleneck surfaced by V3-EXQ-543b diagnose-errors (TASK_CLAIMS session
  diagnose-v3-exq-543c-2026-05-11T0635Z; arc_062_rule_apprehension_plan.md
  decision-log 2026-05-11 option 3a).
  Three new __init__ kwargs (all default to legacy SD-054 behavior; env-only,
  NOT surfaced through REEConfig.from_dims per SD-022 / SD-023 / SD-029 /
  SD-047 / SD-048 / SD-054 precedent):
    reef_bipartite_layout (bool, default False) -- master switch.
    reef_bipartite_axis (str, default "horizontal") -- "horizontal" -> reef
      bottom rows, food top rows. "vertical" -> reef right cols, food left
      cols. Validation: raises ValueError on construction if axis is neither.
    reef_bipartite_agent_band_radius (int, default 1) -- half-width of the
      agent spawn band measured from the midline (inclusive). 0 = midline
      only; 1 = midline +/- 1 (3 rows/cols); 2 = midline +/- 2 (5 rows/cols).
  Geometry (axis="horizontal", radius=1, size=12 default):
    Reef half: rows in (midline + radius .. size - 2] = rows 8, 9, 10
    Agent band: rows in [midline - radius .. midline + radius] = rows 5, 6, 7
    Forage half: rows in [1 .. midline - radius) = rows 1, 2, 3, 4
    Reef patches placed along the bottom edge (row sz-3) with column centres
    evenly distributed across interior columns. Patches that would intersect
    the agent band or forage half are clipped via _is_in_reef_half guard.
  Reset partitioning: when reef_bipartite_layout=True, reset() calls the new
    _place_reef_patches_bipartite() (does NOT consume from `available`) and
    _build_bipartite_pools(available) which returns (agent_pool, forage_pool)
    as two disjoint subsets. Agent pops from agent_pool; hazards / resources /
    waypoints pop from forage_pool. Legacy mode aliases both pools to a single
    `available` list so the pre-existing single-pool pop-from-shared behavior
    is bit-identical.
  Fallback: if agent_pool would be empty (degenerate config, e.g. radius=0 on
    a size where the midline is mostly walls), widens the band by +1 radius
    iteratively until a valid cell exists; records the widen count in
    self._sd054_bipartite_band_widen_count (0 in legacy mode and successful
    bipartite resets).
  No new state in obs_dict, no change to world_obs_dim (still 275 with
    reef_enabled=True), no new training target. Pure env substrate refinement.
  Why the extension was needed: legacy SD-054 places reef patches in fixed
    corners but agent / hazards / food spawn at uniformly-random positions in
    the remaining cells. Per-episode reef / food geometry is randomized; the
    mean optimal policy across episodes converges to a single "head toward
    nearest food" template (a direction-following heuristic that works
    regardless of episode-specific reef-food geometry). The agent-side
    consequence (verified 2026-05-11 by direct numerical probe in
    TASK_CLAIMS session diagnose-v3-exq-543c-2026-05-11T0635Z): CEM proposer
    at init produces 8 candidates all sharing argmax-first-action=3 with
    continuous-action spread ~1e-4 and post-action z_world spread ~1e-5,
    leaving the ARC-062 head reading near-identical inputs and unable to
    discriminate. Bipartite layout forces reef-bound and forage-bound
    trajectories to have categorically opposite first-action argmaxes
    (action 1 = down toward reef, action 0 = up toward food on horizontal
    axis), restoring the structural condition under which a well-trained
    ARC-062 substrate WOULD produce per-candidate first-action argmax
    diversity at probe states sampled from typical training rollouts.
  Diagnostic counter: env.info dict surfacing TBD with the validation EXQ;
    self._sd054_bipartite_band_widen_count is exposed as an attribute for
    direct read.
  Backward compatible: all three new kwargs default to legacy SD-054
    behavior; agent_pool and forage_pool are aliased to a single `available`
    list; pop()-from-shared-pool semantics preserved; bit-identical to
    pre-extension HEAD when reef_bipartite_layout=False.
  Smoke (2026-05-11): backward-compat with legacy reef kwargs reproduces 33
    reef cells spanning rows 1-10, world_obs_dim 275, bipartite_band_widen
    count 0. Activation with reef_bipartite_layout=True, axis=horizontal,
    radius=1 across seeds 0/1/2: agent always spawns in rows [5,6,7]; reef
    cells always in rows [8,9,10] (28 cells with edge-row centres clipped
    by reef-half predicate); hazards + resources always in rows [1,2,3,4];
    band_widen_count 0 in all seeds. Vertical axis with radius=2 verified
    independently (agent col in [4..8], reef cols [9,10], hazards cols
    [1,2]). Bad-axis construction raises ValueError as expected.
  Biological analog: coral-reef refugia in marine systems force categorically
    opposite swim-direction choices (toward reef-shelter vs toward open-water
    foraging grounds) because the two microhabitats are spatially anti-
    correlated, not interleaved. The legacy SD-054 corner-placement is a
    weaker geometric expression of the same claim; the bipartite extension
    is the sharper instantiation.
  No trainable parameters. Pure env substrate. No phased training. No
    MECH-094 interaction (env observation stream, not replay content).
  Validation experiment: V3-EXQ-548 substrate-readiness diagnostic to be
    queued via /queue-experiment immediately following this entry. Will
    measure CEM-candidate first-action argmax entropy at probe states with
    bipartite ON vs OFF across 3 seeds + structural-only (no full P1 falsifier
    rerun in this pass; that decision waits on V3-EXQ-548 PASS).
  See SD-054 (parent claim, unchanged in semantics), MECH-309 (logical-
    necessity claim for which the substrate enables a sharper falsifier),
    ARC-062 (downstream consumer; this extension creates the structural
    conditions for ARC-062 GAP-B falsifier testability), MECH-269 (V_s
    primitive; orthogonal cluster but parallel substrate-readiness pattern),
    SD-023 / SD-047 / SD-048 / SD-049 (parallel env-only substrate-
    enrichment kwargs precedent for not surfacing through REEConfig.from_dims).
