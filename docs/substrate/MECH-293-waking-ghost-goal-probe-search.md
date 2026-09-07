## MECH-293 Waking Ghost-Goal Probe Search (2026-04-27)
- MECH-293: hippocampal.awake_ghost_goal_probe_search -- IMPLEMENTED 2026-04-27.
  Modules: ree_core/hippocampal/module.py (HippocampalModule.propose_trajectories
  extended; new private methods _propose_ghost_seeded + _mix_value_flat_with_ghost;
  diagnostic accessor get_last_propose_diagnostics); ree_core/predictors/e2_fast.py
  (Trajectory dataclass extended with hypothesis_tag: bool=False and
  metadata: Optional[Dict[str, Any]]=None fields); ree_core/agent.py
  (REEAgent._e3_tick threads current_z_goal=goal_state.z_goal into
  propose_trajectories when goal is active; record_committed_trajectory
  explicitly sets hypothesis_tag=False / metadata=None on the executed
  committed trajectory). Read-side consumer of MECH-292 ranked ghost-goal
  bank: extends propose_trajectories with a minority budget of CEM probes
  seeded around the highest-priority bank entries' anchor.z_world rather
  than the agent's current z_world. Each ghost trajectory carries
  hypothesis_tag=True and metadata={"source": "mech293_ghost_probe",
  "anchor_key": ..., "ghost_priority": ..., "goal_match": ...} for
  downstream provenance.
  Algorithm:
    1. n_ghost = clamp(round(n_total * mech293_ghost_fraction),
                      [mech293_min_ghost_candidates, mech293_max_ghost_candidates])
       bounded by len(bank.rank()).
    2. For each top entry: seed action-object distribution mean from
       _get_terrain_action_object_mean(anchor.z_world, e1_prior). Single
       noise draw (no inner CEM refit -- ghosts are exploratory probes,
       not optimised plans, so probe cost <= one value-flat sample).
    3. e2.rollout_with_world(z_self, anchor_z, actions, action_bias=...)
       produces the candidate trajectory; tag + metadata stamped.
    4. Mix with value-flat candidates per mech293_replace_lowest_ranked:
         True (default): drop the highest-cost (worst) value-flat
                          candidates, append ghosts at the tail. Preserves
                          downstream E3 selection cost; len(candidates)
                          stays at n_total.
         False: append ghosts on top of the value-flat pool (raises total
                count). Diagnostic-only path.
    5. Diagnostics dict surfaced on _last_propose_diagnostics:
       {mech293_n_ghost_proposed, mech293_n_ghost_admitted,
        mech293_max_ghost_priority, mech293_mean_goal_match_at_seed,
        mech293_reason in {"ok","no_z_goal","empty_bank","n_ghost_zero"}}.
  Config: REEConfig.use_mech293_ghost_probes (bool, default False) +
    sub-knobs: mech293_ghost_fraction (0.2), mech293_min_ghost_candidates
    (1), mech293_max_ghost_candidates (8), mech293_replace_lowest_ranked
    (True). All wired through REEConfig.from_dims.
  Precondition (raised on HippocampalModule.__init__):
    use_mech293_ghost_probes=True requires use_mech292_ghost_bank=True.
    The MECH-292 block transitively guarantees use_anchor_sets=True and
    AnchorSetConfig.use_sd039_anchor_payload=True, so only the bank
    flag needs explicit enforcement here. Loud-not-silent failure mode
    matches the MECH-292 / SD-039 precondition pattern.
  Backward compatible: use_mech293_ghost_probes=False by default;
    propose_trajectories returns value-flat candidates; new current_z_goal
    arg is ignored when MECH-293 is off; new Trajectory.hypothesis_tag
    and .metadata fields default to backward-compat values (False / None).
    record_committed_trajectory now constructs Trajectory with explicit
    hypothesis_tag=False + metadata=None so the executed committed
    trajectory drops any ghost provenance from the source proposal (the
    executed trajectory IS real, regardless of its origin -- spec
    requirement). 12/12 MECH-293 contracts + 183/183 full preflight +
    contracts PASS with flag OFF (bit-identical to pre-MECH-293 HEAD).
  Activation smoke (2026-04-27, V3-EXQ-497 5/5 PASS, 34s on Mac):
    UC1 module surface (config flags + methods + Trajectory fields all
      exposed with correct defaults); UC2 master-OFF no-op (n_ghost=0,
      diagnostics={}, all candidates default-clean); UC3 ghost branch
      fires (n_ghost_admitted=4, max_priority=1.61,
      mean_goal_match_at_seed=0.998, reason='ok'); UC4 hypothesis_tag
      preserved on every ghost + metadata complete + 28 value-flat
      candidates remain default-clean; UC5 budget arithmetic
      (round(0.25*8)=2 in [1,4] arm A; bank-size cap to 1 in arm B;
      min-floor wins over fraction=0.0 in arm C).
  No trainable parameters. Pure routing logic. No phased training needed.
  ASCII-safe (no print() output added).
  MECH-094: ghost trajectories carry hypothesis_tag=True for
    provenance-routing. CEM rollout itself does not write residue or
    anchors during proposal (those are observation-side paths in
    agent.sense()), so no inline gate is needed during proposal. At
    commit boundary, record_committed_trajectory explicitly strips the
    tag (and metadata) so the executed trajectory is treated as real
    for downstream backward-credit-sweep / valence-write paths.
    SD-039's build_goal_payload(simulation_mode=True) path returns None
    already (handled at the SD-039 layer). No new MECH-094 plumbing
    required at the MECH-293 layer.
  ARC-007 strict: ghost probes do NOT add a hippocampal value head.
    Goal-match enters via MECH-292's external ranking over stored
    payloads, which lives outside HippocampalModule. The proposer is
    still proposing trajectories without an internal value computation;
    the ghost-seeded ones are biased BY LOCATION (the anchor's z_world)
    not by an internal value head.
  Validation experiment: V3-EXQ-497 5/5 PASS 2026-04-27 (UC1-UC5 above).
    Behavioural validation = V3-EXQ-495 (V3 full-completion gate,
    MECH-163 dual systems test); already drafted, gated on this
    substrate. queue V3-EXQ-495 as a separate decision after reviewing
    V3-EXQ-497 -- 3 conditions x 2 paradigms x 7 seeds is a several-hour
    behavioural run, not a substrate-readiness diagnostic.
  Design doc: REE_assembly/docs/architecture/mech_293_ghost_goal_probe_search.md
  See MECH-293 (this claim), MECH-292 (upstream ranked-bank source),
    SD-039 (transitive payload substrate), ARC-007 strict (no value head),
    ARC-018 (waking trajectory proposal loop being modified), ARC-032
    (goal-biased sequence generation -- one instantiation), MECH-089
    (theta-packaged waking E3 updates -- architectural context),
    MECH-094 (hypothesis-tag invariant -- preserved at proposal,
    stripped at commit), MECH-269 (anchor / probe substrate -- transitive
    via MECH-292), MECH-291 (mode-sensitive sequence generator framing --
    MECH-293 is the waking arm).
