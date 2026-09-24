## MECH-365 Provenance-Bearing One-Way Commit-Status Gate: replay label + REM boundary knobs (2026-09-24)

- MECH-365: memory.provenance_bearing_event_token (V3 half) -- IMPLEMENTED 2026-09-24.
  V3's `committed_vs_imagined` label is `Trajectory.hypothesis_tag` (plus `LatentState.hypothesis_tag`),
  carried ON the representation object, not only passed as a per-call mode argument.
  Modules: `ree_core/hippocampal/module.py` (`HippocampalModule.replay`,
  `HippocampalModule.spread_reverse_replay_wanting`), `ree_core/agent.py`
  (`REEAgent.run_rem_attribution_pass`), `ree_core/utils/config.py` (`HippocampalConfig`).
  Config (all `HippocampalConfig`, all reachable through `REEConfig.from_dims`, all no-op at default):
  - `mech365_suppress_replay_provenance_stamp` (bool, default False). **E1 -- a correctness fix,
    ON by default.** `replay()` now stamps `hypothesis_tag=True` on every returned trajectory (an
    E2 rollout of random actions = imagined content), as its docstring always claimed. Before
    2026-09-24 the output carried the dataclass default `False`; that was latent only because no
    consolidation writer received `replay()` output. Bit-identical in production: the only
    ree_core reader of `Trajectory.hypothesis_tag` is `spread_reverse_replay_wanting` (fed only
    `reverse_replay()` output as shipped); no experiment driver reads the tag on replay output
    (audited 2026-09-24). `True` reproduces the old unlabelled output and exists only as the
    V3-EXQ-1085 source-side canary arm.
  - `rem_route_forward_replay_to_consolidation` (bool, default False). **E2.** Presents each
    FORWARD (imagined) REM replay trajectory to the same MECH-217 writer the reverse (real) replay
    reaches, so the carried label -- not routing exclusion -- is what keeps imagined content out
    of committed history. OFF = as shipped (forward replay scored read-only; no new metric keys).
    ON emits `rem_fwd_*` (n_scored / n_presented / n_reached_gate / n_accepted / n_refused /
    accepted_mass / n_lesion_overrides) and `rem_rev_spread_*` (n_accepted / accepted_mass /
    n_sham_overrides). The scoring (read) path is unchanged in both states; routing consumes no RNG.
  - `mech365_provenance_lesion` (str, default "off"; "drop_at_consolidation" | "sham_real_only";
    anything else raises `ValueError` in `from_dims` and at the write site). **E3.** A BOUNDARY
    lesion: the tag passed to `update_valence` is dropped while the sender `Trajectory` is never
    mutated. The sham runs the identical assignment on already-untagged (real) trajectories only.
    `spread_reverse_replay_wanting` additionally returns `n_steps_accepted`,
    `n_steps_refused_provenance`, `accepted_mass`, `mech365_lesion_override`,
    `mech365_sham_override` (existing keys and the early-return `{}` unchanged).
  Data flow: `replay()` (label stamped) -> `run_rem_attribution_pass` forward loop (scored; with E2
  also presented) -> `spread_reverse_replay_wanting` (effective tag = trajectory tag unless E3
  drops it) -> `ResidueField.update_valence` (refuses iff the tag it receives is True).
  MECH-094: this build IS the MECH-094 label on replay content; nothing new writes under
  `hypothesis_tag=True`.
  Backward compatible: all defaults bit-identical (pinned by
  `tests/contracts/test_mech365_provenance_gate.py::test_rem_routing_default_off_is_bit_identical_and_emits_no_new_keys`).
  Regression latch: `test_replay_output_carries_imagined_label` FAILS on the pre-2026-09-24 tree.
  Measured liveness (V3-EXQ-1085 harness, 5 seeds x 6 cycles): gate intact -> 900 refused
  imagined waypoint writes/seed, committed wanting map bit-identical to unrouted; boundary lesion
  -> relative L1 divergence 2.3-860x of the intact map. Note the lesioned contamination
  SELF-AMPLIFIES: forward replay starts at the wanting-seeded terminus, imagined waypoints write
  onto nearby centers, and later spreads' kernel-weighted terminus READ sees the inflated value --
  MECH-217's no-self-write guard does not cover that read-side loop.
  Phased training required: no (no learned component).
  Not built here (known, owed elsewhere): MECH-290 `record_committed_trajectory` strips the tag
  at commit ENTRY on the proposal's PREDICTED world_states (ghost probes: rooted at a remote
  anchor) -- a latent one-way-gate non-conformance under `use_backward_credit_sweep` +
  `use_mech293_ghost_probes` (both default OFF); proposed as a substrate_queue entry by
  governance-20260924.
  Validation experiment: V3-EXQ-1085 (`experiments/v3_exq_1085_mech365_provenance_gate_boundary_lesion.py`).
  See MECH-365, MECH-365a (v4 unified-store schema), MECH-094, INV-011, MECH-037, MECH-217, MECH-545
  (the five-way contract-lesion assay this single-gate lesion is NOT), MECH-271.
