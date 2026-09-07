## Q081-REACH-CHECK-PAIR-SPECIFIC: cheap empirical pre-flight reach probe for Q-081

- Q081-REACH-CHECK-PAIR-SPECIFIC (substrate_queue.json sd_id, not a numbered SD -- an
  experiment-layer diagnostic library, not a new architectural feature) -- IMPLEMENTED
  2026-07-31. New `experiments/_lib/q081_pair_reach_check.py`: three consecutive Q-081
  landmark-removal runs (V3-EXQ-824, 824a, 838) found RV(z_world, operating_mode)
  bit-identical between the INTACT arm and every manipulation arm at every seed, despite
  `assert_behavioural_reach()` (q081_landmark_removal.py) reporting the REACH_CONSUMERS
  flags MET each time from 824a onward -- a blanket flag check is necessary but not
  sufficient. This module is the EMPIRICAL check: it reads `agent.salience._input_signals`
  (the closed, named set of values `SalienceCoordinator.tick()` computes `operating_mode`
  from -- source-traced in the module docstring) at every tick for an INTACT vs the
  PRIMARY_MODE=`iei_permute` manipulation arm, over a short untrained rollout, and diffs
  them. Because `operating_mode` is a deterministic function of exactly these precursor
  values (plus static config), this is a necessary-and-sufficient test, cheaper than
  waiting for an RV statistic on the softmax OUTPUT across a full multi-seed recording run
  -- and it NAMES which signal (if any) moved and at which tick.
  Config: no new `REEConfig` / `ree_core` flag -- same discipline as
  `q081_landmark_removal.py` / `q081_surrogate.py` (experiment-layer only, no
  backward-compatibility surface).
  Data flow: n/a (no new latent, encoder, or agent-loop wiring; reads an existing plain
  dict attribute non-destructively).
  MATCHED-ARM CONSTRUCTION (load-bearing, found empirically while building this): both
  arms run from a `copy.deepcopy()` of ONE seeded, freshly-constructed agent, with
  `reset_all_rng(seed)` (arm_fingerprint.py) called before each arm's rollout -- exactly
  V3-EXQ-838's `arm_cell` discipline. An early version without this produced a
  false-positive "reach" (a `drive_level` divergence) that was actually uncontrolled
  global torch RNG state ordering between the two arms, not the landmark manipulation;
  fixed and re-verified (5/5 seeds correctly report no reach once RNG-matched).
  NON-DEGENERACY GUARD (load-bearing, found empirically): the hippocampal event
  segmenter's boundary trigger fires SPARSELY on an untrained, randomly-initialised
  network, with large seed-to-seed variance (0 boundaries in 1200 untrained ticks at one
  seed, 76-120 at others, identical config otherwise). A rollout with zero true boundary
  events gave the manipulation nothing to scramble; `run_pair_specific_reach_probe`
  reports `is_degenerate=True` (raises when strict) rather than a misleading "no reach"
  verdict -- the same class of guard MECH-466's own non-degeneracy convention specifies.
  Not a new architectural feature, so no SD doc / phased-training / MECH-094
  considerations apply (no simulation/replay content is written; it is a read-only
  diagnostic).
  Validation: 15 contract tests, `tests/contracts/test_q081_pair_reach_check.py` (pure
  detector-fires-on-real-divergence / detector-does-not-false-positive-on-identical-
  traces / tolerance / first-divergent-tick-is-earliest / length-mismatch-raises tests
  against fakes; a live REEAgent/env smoke test against the real substrate; a live
  degeneracy-guard-raises test) -- all pass in ~7.5s. Manual run at the module's own
  non-degenerate defaults (n_episodes=3, steps_per_episode=400, env_size=6): seeds 0 and 1
  cleared the guard and both reproduced V3-EXQ-824a/838's confirmed finding
  (`has_pair_specific_reach=False`); seed 2 came back correctly flagged degenerate.
  Validation experiment: none queued in this session (scope discipline -- this is a
  substrate readiness / pre-flight-gate fix, not a Q-081 evidence experiment itself). A
  future Q-081 driver should call `run_pair_specific_reach_probe(...)` or
  `assert_pair_specific_reach(..., strict=True)` BEFORE committing to a full multi-seed
  recording run: a raise means a full run would reproduce the same non_contributory
  result -- do not run it; find a different manipulation lever with confirmed reach to a
  named salience signal, or reframe the measured pair onto a confirmed-reachable one (H2).
  See Q-081, INV-091, `REE_assembly/evidence/planning/substrate_queue.json` sd_id
  Q081-REACH-CHECK-PAIR-SPECIFIC.
