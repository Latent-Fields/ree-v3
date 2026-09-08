## SD-075: phasic.ema_episode_continuity -- IMPLEMENTED (2026-07-19)
- SD-075: phasic.ema_episode_continuity -- IMPLEMENTED 2026-07-19.
  Module: ree_core/regulators/phasic_surprise_burst.py (extends SD-069, same file).
  Config: `phasic_burst_baseline_continuity` ("reset" default | "carry") and
  `phasic_burst_warmup_ticks` (0 default = OFF, -1 = DERIVE as
  ceil(3 / surprise_ema_decay) = 30 at the default decay, positive = verbatim;
  anything below -1 RAISES so a typo cannot silently mean OFF).
  BIT-IDENTICAL OFF: both defaults are no-op, so every existing consumer of
  phasic_surprise_burst is unchanged (pinned by tests/test_flag_inertness.py and
  SD-075 D1/D5b).
  WHAT IT FIXES: reset() cleared the surprise-EMA cold at every episode boundary,
  and the first waking tick of an episode cannot fire an event (it SEEDS the
  baseline). With surprise_ema_decay 0.1 the baseline needs ~10 ticks, so a seed
  whose episodes are shorter than that never runs against a converged baseline and
  n_event_ticks measures episode LENGTH rather than surprise. V3-EXQ-779b: seed 23
  ran ~6.9-step episodes vs seeds 29/37 at 300 (43x); raising its budget 835 -> 2400
  env steps bought 345 MORE short episodes and phasic_fires_real_events did not move
  (6 vs 10). The MIN just migrated to another short-episode cell despite ~2.85x
  exposure. NO STEP-BUDGET INCREASE CAN REACH THIS -- the binding axis is episode
  length. Capability is RULED OUT, not unproven: burst_level_max = 1.00 in every
  PHASIC-ON cell including both seed-23 cells.
  LEG (a) "carry": reset() preserves the surprise-EMA while STILL clearing the
  envelope, cached delta, and per-episode diagnostics, so no in-flight burst leaks
  across a boundary. This is the biologically faithful setting (LC baseline
  adaptation is continuous across behavioural episodes; a per-episode reset has no
  biological counterpart). "reset" stays the DEFAULT only for backward compatibility
  with runs already recorded against SD-069 -- NOT because it is the better model.
  Declare "carry" deliberately in new work.
  LEG (b) warmup_ticks is ACCOUNTING ONLY -- IT DOES NOT SUPPRESS THE BURST. During
  warmup the regulator still fires and still perturbs the softmax temperature; only
  the counts split into n_events_prewarmup / n_events_converged. Suppression was
  REJECTED (user-confirmed 2026-07-19) because it would change behaviour in the first
  ticks of a lifetime and confound the MECH-063 (ii) retest with a second mechanism
  change. Do not "tighten" it into a suppressor.
  CONSUMERS: read `n_events_converged` (NOT the per-episode `n_events`) over
  `n_converged_ticks`, and declare a cell UNINFORMATIVE when n_converged_ticks is too
  small -- a MIN-across-cells precondition otherwise treats a starved cell as a real
  measurement, which is exactly how 779b was withheld.
  RETEST BRAKE: MECH-063 sub-claim (ii) is pending_retest_after_substrate against
  this build, BUT the re-derive brake fired both halves on MECH-063 (3rd autopsy) and
  refuses another lettered tonic/phasic iteration against the current regulator.
  DO NOT QUEUE V3-EXQ-779c. A new-number redesign of a DIFFERENT mechanism is allowed.
  LIVE SMOKE + A TRAP FOR THE RETEST AUTHOR (25 eps x 7 steps, seed 23, untrained,
  instantaneous_pe): reset/gate-off reproduces the failed precondition EXACTLY at 6
  events; carry/gate-off lifts the SAME exposure to 12, clearing the threshold of 10.
  But with the gate ON, n_events_converged is 0 in BOTH modes -- every event lands
  inside the first 30 ticks and an untrained agent's PE stream then settles. So the
  legs DISAGREE about this cell: (a) alone reads "12, passes", (a)+(b) reads "0
  converged, UNINFORMATIVE". A retest that enables "carry" and reads the RAW count is
  reporting 12 events the gate says are all warmup-era. Read n_events_converged, and
  expect an untrained short-episode cell may honestly have nothing to report -- a far
  more useful failure than "6 vs 10". SD-074 probe_warmup may be complementary here (a
  trained stream may keep firing past tick 30) but that is an UNTESTED hypothesis.
  Smoke: 22 SD-075 contracts pass; SD-069's 8 contracts + flag-inertness unchanged;
  full suite 1772 passed.
  Design doc: REE_assembly/docs/architecture/sd_075_phasic_ema_episode_continuity.md
  Routed by REE_assembly evidence/planning/failure_autopsy_V3-EXQ-779b_2026-07-19.json
  targets[0] (priority 1). See SD-069 (the module it extends), MECH-063.
