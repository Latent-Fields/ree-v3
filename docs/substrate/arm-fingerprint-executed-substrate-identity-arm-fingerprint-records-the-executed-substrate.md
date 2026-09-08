## arm-fingerprint-executed-substrate-identity -- arm_fingerprint records the EXECUTED substrate, not the on-disk substrate -- IMPLEMENTED (2026-07-20)
- arm-fingerprint-executed-substrate-identity -- arm_fingerprint records the EXECUTED
  substrate, not the on-disk substrate -- IMPLEMENTED 2026-07-20.
  experiments/_lib/arm_fingerprint.py, experiments/_lib/manifest_core.py,
  experiments/_lib/arm_reuse.py. INSTRUMENTATION ONLY -- no ree_core change, no config
  param, no behavioural effect on any experiment; nothing here can move a metric.
  DEFECT: compute_substrate_hash reads source FROM DISK at cell entry, while the cell
  executes in-memory bytecode frozen in sys.modules at first import. When the checkout
  moves mid-run -- routine on a fleet that pulls continuously -- the manifest records the
  DISK state, not the EXECUTED state. Worked instance V3-EXQ-778a: 6 of 8 cells stamped
  c8d6d0e2 while provably executing e9a22a91 (commit da873a1 landed 85s into a 418s run,
  editing a _lib harness the driver had already bound at module scope), all 8 carrying
  reuse_eligible: true. That is a FALSE-HIT channel -- the one failure mode
  arm_fingerprint.py:20-27's governing asymmetry says must never occur, since
  over-inclusion was designed to buy false MISSES only.
  SCOPE OF THE CLAIM (the sweep headline was corrected hours after it was written):
  intra_run_substrate_divergence_sweep_2026-07-20.md's "42 of 164 (25.6%)" counts runs
  whose RECORDED HASH VALUE changed -- NOT runs that lost experimental control, and it
  must not be cited as the latter (failure_autopsy_V3-EXQ-782 + the sweep author's own
  07:30Z WORKSPACE_STATE correction). _SUBSTRATE_GLOBS is uniformly wider than what any
  run imports, so an unrelated parallel edit moves the hash without touching the run; on
  782 the closure-restricted hash was byte-identical across all four bands. 778a's own
  confound was likewise REFUTED -- all 8 seeds provably executed one build. What STANDS,
  and what this lands, is the INSTRUMENT defect that autopsy routed here: the recorded
  identity was never guaranteed to be the executed identity.
  FIX, three parts:
  (1) resolve_substrate_identity() memoises the substrate hash for the process lifetime,
      keyed by (repo_root, declared scope, extra paths); compute_arm_fingerprint serves
      every cell from that one snapshot. The first resolution happens at the first cell,
      after the driver's module-scope imports have frozen the executed bytecode and
      before any mid-run move can be observed -- so recorded identity IS executed
      identity by construction. driver_script_hash is frozen the same way.
  (2) substrate_stability_report() re-hashes from disk at stamp time;
      stamp_recording_core writes top-level substrate_stable_across_run (+ a
      substrate_stability_detail block when False). Two independent tests, either of
      which can only prove INSTABILITY: per-cell hash cardinality > 1, or process
      snapshot != disk. Catches the one residual the freeze cannot -- a module imported
      LAZILY after the first cell. NOT added to ALWAYS_CORE_KEYS: the pre-2026-07-20
      corpus cannot carry it and making it core would WARN on every legacy manifest.
  (3) arm_reuse.source_run_substrate_unstable() + REFUSE_SUBSTRATE_UNSTABLE: the reuse
      consumer refuses to serve any cell out of a run whose cells disagree about their
      substrate. This is the RETROACTIVE handling of the 42 known-divergent runs -- their
      manifests are NOT edited (completed runs are re-adjudicated by autopsy, never
      rewritten), so the guard lives in the only path that could ever act on them. It is
      computed from the manifest at lookup time, so a stale index cannot bypass it.
      Absence of substrate_stable_across_run is NOT read as instability (that would
      refuse every banked mint); it falls through to the cardinality test.
  NOT a hard cut like torch-in-machine_class: on a stable run the emitted hash is
  byte-identical to before, so every banked fingerprint still matches. The new payload
  fields (substrate_identity_source, substrate_identity_resolved_at) are observability
  only and deliberately outside fp_input.
  KNOWN OVER-REFUSAL, and the 782 caution it answers: cardinality > 1 over-reports, via
  driver_script_in_substrate_hash being toggled mid-run (one corpus run, V3-EXQ-788) and
  more importantly via the over-wide globs above. failure_autopsy_V3-EXQ-782 warns that a
  whole-glob divergence flag "would institutionalise this false positive as a standing
  warning" and asks for a closure-restricted check. That caution governs a signal that
  ADJUDICATES SCIENCE, where a false positive wrongly impeaches a real result. The reuse
  gate is the opposite asymmetry -- a false MISS costs compute, a false HIT corrupts a
  conclusion -- so over-refusal is correct HERE precisely because it is wrong THERE; the
  two must not be collapsed. substrate_stable_across_run is therefore scoped as an
  INSTRUMENT event, never a confound verdict, and nothing adjudicates a claim off it.
  When the closure-restricted recomputation lands (782's secondary routing), it should
  narrow BOTH -- recovering the wrongly-refused runs at no cost to safety.
  Also corrected: _hoist_multi_arm_substrate_hash's docstring asserted "all arms of one
  run execute against the same substrate", which the sweep falsified; the hoist kept only
  the first hash and so actively HID divergence at the top level. Behaviour unchanged
  (back-compat), premise documented, per-cell set now exposed via
  multi_arm_substrate_hashes().
  Phased training: N/A. MECH-094: N/A (no memory writes).
  Validation: no EXQ -- this changes no mechanism and gates no claim; it is verified by
  17 new contracts in tests/contracts/test_arm_fingerprint_executed_substrate.py, which
  encode the failure record's acceptance target directly (recorded identity == executed
  identity for 100% of cells, OR the run is stamped substrate_stable_across_run: false).
  Source: REE_assembly/evidence/planning/failure_autopsy_V3-EXQ-778a_2026-07-20.json
  targets[0].recommended_substrate_queue_entry, and
  REE_assembly/evidence/planning/intra_run_substrate_divergence_sweep_2026-07-20.md.
