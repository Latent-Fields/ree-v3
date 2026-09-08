## stdlib-`random` seeding in ree_core (`REEConfig.stdlib_rng_seed`) -- IMPLEMENTED (2026-07-28)
- stdlib-`random` seeding in ree_core (`REEConfig.stdlib_rng_seed`) -- IMPLEMENTED 2026-07-28.
  NOT an SD -- a reproducibility knob over existing mechanisms, same class as
  `ScaffoldedSD054OnboardingConfig.scaffold_env_seed` (which likewise got contracts, no SD doc).
  THE GAP: three ree_core call sites draw from the stdlib `random` module, which auto-seeds
  from OS entropy at import, while experiment drivers seed only torch and numpy (`_run_seed`
  in a typical driver calls `torch.manual_seed` + `np.random.seed` and nothing else). Any run
  reaching one of those sites is therefore NOT reproducible across processes. `ree_core` seeds
  stdlib `random` nowhere -- verified by grep across the whole package.
  NOT the scaffold-curriculum non-determinism. That was the unseeded env
  (`np.random.default_rng(None)`), fixed in `7afea288fa`; on the 460c dry_run curriculum this
  path is DORMANT (stdlib-random state never advanced across all 60 Stage-0 select_action
  calls). Do not re-open that question -- it is closed. This landing closes a DIFFERENT,
  still-open exposure that would look identical if it ever fired.
  MEASURED LIVENESS (2026-07-28, per-site call counters on the module-level `random` binding
  each site resolves through -- not inferred from code reading):
    S1  hippocampal/module.py `diverse_replay(mode="auto")` per-step mode roll -- FIRES ONCE
        PER REPLAY STEP whenever `replay_diversity_enabled=True` (measured 5 rolls for a
        5-step replay; 0 with the flag off). Broadest exposure of the three: one config flag,
        and agent.py's SWS replay path passes `mode="auto"` explicitly.
    S2  hippocampal/module.py `_sample_exploration_trajectory` zero-weight fallback -- narrow:
        needs `retrieval_bias` supplied AND summing to zero against the memory_strength
        weights (measured: bias=None -> 0 calls, bias=[1,2] -> 0, bias=[0,0] -> 1). Reachable
        via a degenerate BLA retrieval bias, and called directly by V3-EXQ-659.
    S3  sleep/self_model_aggregator.py `offline_gradient_pass` waking-pair sample -- fires
        whenever the harm replay buffer is non-empty (measured: empty -> 0, non-empty -> 1)
        under `use_mech273_self_model=True`, which `REEConfig.enable_sleep_aggregation_cluster()`
        turns on as one of its eight flags.
  THE FIX (opt-in; default bit-identical): `REEConfig.stdlib_rng_seed: Optional[int] = None`,
  three-site from_dims wiring (dataclass field, signature, assignment) plus a mirror onto
  `HippocampalConfig.replay_rng_seed`, guarding the silently-unreachable-flag hazard.
    Each consumer holds an `_rng` source that DEFAULTS TO THE `random` MODULE ITSELF, so
      `self._rng.random()` / `.choice()` / `.choices()` resolve to the very same bound methods
      of the process-global Random instance the previous bare `random.*` calls used. The
      default is bit-identical BY CONSTRUCTION, not by equivalence.
    When set, each consumer gets its own `random.Random` instance. `random.seed()` is NEVER
      called -- a module-local instance, so seeding an agent cannot perturb the host process's
      global RNG or any other stdlib-random consumer in it.
    `derive_stdlib_rng_seed(base, stream)` namespaces the consumers (stream 0 = hippocampal
      replay, stream 1 = self-model writeback) so one base seed never hands two consumers a
      correlated sequence. Same idiom as `_derive_env_seed` in scaffolded_sd054_onboarding.py.
    `HippocampalModule.seed_replay_rng(seed)` is a no-op on None, so both construction paths
      (from_dims mirror, and a hand-built REEConfig setting only the top-level knob) honour it.
  Backward compatible -- verified TWO ways, not just asserted:
    (a) A/B against pre-change HEAD (`25bf07e`) in a throwaway worktree, knob unset, global
        stdlib RNG pinned: identical replay digest
        (`efcc3ecd07b5f910222d7854a8dc7330643c0a3c2a37c8685b0beaa72c80471e`), identical
        reverse-count (5/12), identical 20-element fallback pick sequence.
    (b) Identity assertions in the contracts (`_rng is random`), which is the structural
        reason (a) must hold.
  DELIBERATE BEHAVIOUR CHANGE WHEN SET: a seeded run reaches different draws and is NOT
  comparable to a landed run. Pin it deliberately, within one experiment, as a seeded pair.
  Do NOT retro-apply to landed runs or re-score claims on it.
  Contracts: `tests/contracts/test_stdlib_rng_seed_determinism.py` (12 tests), pinning BOTH
  directions per the scaffold_env_seed precedent -- D1/D2 default-is-the-global-instance
  (identity), D3 negative control that the unseeded sites really DO consume the global RNG,
  D3b None passthrough, D4 reproducible across constructions, D5 independent of global RNG
  state, D6 distinct seeds -> distinct streams, D7 seeding never perturbs the global RNG,
  D8 S2 fallback, D9 S3 writeback (both with unseeded negative controls), D10 stream
  namespacing, D11 the from_dims three-site mirror. Mutation-checked in both directions:
  making `seed_replay_rng` a no-op fails D4/D5/D7/D8; making the default a fresh
  `random.Random()` fails D1/D3.
  No validation experiment queued: this is a reproducibility knob whose default is proven
  bit-identical and whose ON path is pinned by contracts -- there is no substrate behaviour
  to validate empirically. The natural first consumer is any future cross-process determinism
  check that needs `replay_diversity_enabled=True`.
  CROSS-PROCESS REPRODUCTION VERIFIED 2026-07-28 -- the 12 contracts pin determinism WITHIN
  one process only, so "does a pinned run actually reproduce in a SECOND process, or is there
  a FOURTH entropy source still unfound?" was open until measured.
  `scripts/stdlib_rng_cross_process_probe.py` runs a real agent+env+sleep loop on a config
  reaching ALL THREE sites and digests the full trajectory + sleep metrics + every parameter
  byte. Measured on ree-worker-3 (linux-x86_64, py3.10.12, torch 2.12.0+cpu):
    ARM pinned   (stdlib_rng_seed + env seed + torch/numpy seeded) -- 3 processes, ONE digest
      `8ff7ac0e1d803eb733f7bfd8aa0ebc48539df519496a571edd601e03b467f88d`, identical
      state_dict. Replicated on ree-worker-2: identical in-machine AND byte-identical to
      worker-3's digest (same machine class). Draw counts S1=15 / S2=12 / S3=1 every run.
    ARM unpinned (the typical-driver state: torch+numpy seeded, stdlib `random` on OS
      entropy, everything else held) -- 3 processes, THREE DIFFERENT digests and three
      different state_dicts.
  RE-MEASURED 2026-07-28 ON THE REAL SLEEP ROUTE. The measurement above drove S2 and S3
  directly at their call sites, because Phase E early-returns with
  `mech273_writeback_regions = 0` unless `replay_sampler.draw()` returns non-None -- which
  needs the MECH-269 anchor-set substrate `enable_sleep_aggregation_cluster()` deliberately
  does not bundle. The probe now closes that chain (`use_anchor_sets`, `use_event_segmenter`,
  `use_per_stream_vs`, `use_invalidation_trigger`, `use_staleness_accumulator`,
  `mech285_draws_per_cycle`, `use_affective_harm_stream`, and `sleep_loop_episodes_K` above
  the episode count so `reset()` cannot fire an implicit second cycle), so **S3 now fires
  through `SleepLoopManager.force_cycle` -> Phase-E writeback**, not by direct call. Only S2
  remains direct (`_sample_exploration_trajectory` needs an all-zero retrieval_bias the waking
  route does not produce), still via the agent's own seeded consumer over its real buffer.
  New counts, every run of both arms: `mech273_writeback_regions = 4` in all 3 cycles,
  `self_model.choices = 3` (one per cycle). S1 falls 15 -> 5 and S2 stays 12; the S1 drop is
  the config change altering how many e3_quiescent replay ticks occur, not lost coverage.
    ARM pinned   -- 3 processes on ree-worker-3, ONE digest
      `027b3acdae1177324a9970a0b779ef5cef96a6a3d836680774d567738313e5d0`, identical
      state_dict. Replicated on ree-worker-2: same digest, again byte-identical across the
      two boxes of the same machine class.
    ARM unpinned -- 3 processes, THREE DIFFERENT digests and state_dicts on EACH worker
      (six distinct digests across the two). Sensitivity is visible in the counters too:
      unpinned draws S1=10 rather than 5, and one run diverged far enough to install a 5th
      anchor (`wb_regions [4,4,5]`).
  ANCHORS ARE SEEDED, NOT WAKING-DERIVED -- worth stating plainly. The natural route was
  tried first and measured: over all 36 waking ticks the MECH-288 segmenter emitted ZERO
  BoundaryEvents, so the pool stayed empty and every `draw()` returned None. That is the
  probe's tiny task, not a substrate defect: the policy collapses to one repeated action in
  episode 1, the z_world/z_self deltas then decay monotonically, and a monotonically
  decreasing series never rises 0.65 sigma above its own trailing window mean, which is what
  the fast scale's `pe_threshold` detector needs. So `_seed_anchors` writes the pool directly,
  as V3-EXQ-574 (the run that validated MECH-273 Phase E) does -- with a deterministic arange
  ramp rather than 574's `torch.randn`, since a reproducibility probe should not add an RNG
  consumer it does not need. Anchor provenance is upstream of every RNG site under test; what
  had to be real is the sleep-side route, and that is now entirely `_run_cycle`.
  The coverage check is no longer advisory: `main()` now EXITS 3 if `self_model.choices` is
  absent or `mech273_writeback_regions` is 0 in every cycle, so a config regression fails
  loudly instead of printing a reassuring hash.
  So: NO fourth entropy source, and the unpinned arm is what makes that statement mean
  anything -- it proves the probe is sensitive to exactly the entropy this knob closes.
  Do NOT drop the unpinned arm: without it, two matching pinned digests are equally
  consistent with "the seeding works" and with "the sites never fired". The draw counters
  are the second, direct check on the same thing, and they earned their keep -- the FIRST
  two probe builds reported ZERO draws (the `tiny_loop.step_once` path never reaches
  `_do_replay`, the only waking route to S1; and `agent.run_sleep_cycle()` stops before the
  Phase-E WRITEBACK that reaches S3, which needs `SleepLoopManager.force_cycle`). Both would
  have produced two matching, entirely meaningless digests.
  SCOPE: same machine class only. `torch.multinomial` is not reproducible across
  darwin-arm64/torch 2.10 vs linux-x86_64/torch 2.12 (see the "Running the test suite"
  cross-machine-class note in the umbrella CLAUDE.md), and that is unaffected by this knob.
  One footnote worth keeping, and it still costs time if forgotten: `harm_dim` must equal
  `E2HarmSConfig.z_harm_dim` (32), because `offline_gradient_pass` slices waking pairs to
  that width and a narrower stream slices SHORT rather than erroring, dying later on a
  shape mismatch deep in the transition net.
  See REE_assembly/evidence/planning/scaffold_goal_freeze_e3_read_path_triage_2026-07-27.md
  ("Follow-on: the residual entropy source is NAMED and FIXED", point 4), MECH-165 (S1/S2),
  MECH-273 (S3), and `experiments/scaffolded_sd054_onboarding.py` scaffold_env_seed (the
  design precedent this follows).
