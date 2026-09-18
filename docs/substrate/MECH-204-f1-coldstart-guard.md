## MECH-204 F1 Cold-Start Guard: Waking-Tick Gate on the Persistent Zero-Point (2026-09-18)

- MECH-204: neuromodulation.serotonin.precision_zero_point_require_waking -- IMPLEMENTED 2026-09-18.
  `ree_core/neuromodulation/serotonin.py` (SerotoninConfig + SerotoninModule),
  producer in `ree_core/agent.py` REEAgent.sense().
  Config: `SerotoninConfig.precision_zero_point_require_waking` (default `False`;
  set `True` to enable). Wired at all three mandatory knob sites -- dataclass field,
  `REEConfig.from_dims()` signature, and the post-`cls()` re-apply in the `from_dims`
  body (`from_dims` silently swallows unknown kwargs, so the re-apply is load-bearing).
  Data flow: waking tick -> `REEAgent.sense()` -> `SerotoninModule.note_waking_tick()`
  -> `_waking_ticks_since_capture` -> gate in `enter_rem()` -> `_persistent_zero_point`
  -> `compute_recalibration_target()` -> `sleep/phase_manager.py` WRITEBACK consumer.
  Backward compatible: disabled by default; with the flag off no new call is made
  from `sense()` and `enter_rem()` is bit-identical to the pre-guard version.
  Phased training required: no (scalar arithmetic; no encoder head, no gradient).
  MECH-094: not applicable -- the guard adds no memory writes and no simulated
  content. It is waking-only by construction: `note_waking_tick()` no-ops unless
  `SerotoninModule._phase == "wake"`, so replay / REM ticks cannot satisfy it.
  Contracts: `tests/contracts/test_mech204_f1_coldstart_guard.py` (C1-C8).
  See MECH-204 (F1 cross-cycle persistent zero-point), MECH-016 / EVB-1389.

### The defect

`REEAgent.enter_rem_mode()` calls `serotonin.enter_rem(current_precision=e3.current_precision)`
unconditionally, and `enter_rem()` seeded `_persistent_zero_point` from whatever precision
it was handed, with no gate on whether any waking experience had occurred since the last
capture. The canonical driver start pattern (`env.reset(); agent.reset()` before any waking
tick -- also `StepHarness.run_episode`) makes `SleepLoopManager` fire a cycle with ZERO
waking ticks, so the cold-start anchor was E3 `precision_init` (rv=0.5 -> precision 2.0):
a sentinel reflecting no experience at all. The F1 EMA (alpha=0.1) then carried that
sentinel for ~30+ cycles, so the WRITEBACK consumer pushed an already-calibrated rv AWAY
from calibration (realized z_world PE variance ~0.0039 -> recalibrated 0.0121). Measured
IGW-20260915-243; V3-EXQ-541c shows the same seed at cycle-1 target 2.148.

**A second instance of the same predicate**, found while building this guard (2026-09-18)
and covered by the same fix: a cycle issues more than one `enter_rem()` call, and the later
call re-reads the SAME precision with no intervening waking tick, double-counting one
observation into the EMA.

### Why the producer is `sense()` and not `serotonin_step()`

`SerotoninModule.serotonin_step()` is called by experiment DRIVERS, never from inside
`ree_core`, and `StepHarness` never calls it at all (zero hits for "serotonin" in
`experiments/_harness.py`). A counter fed only from `serotonin_step()` would therefore stay
at 0 forever on the canonical loop, making `enter_rem()` skip EVERY capture -- a permanent
kill switch for MECH-204 recalibration rather than a cold-start guard -- while the
unit-level contracts, which call `serotonin_step()` directly, all still passed. `sense()`
is the one call every waking tick makes on every driver (StepHarness, `act_with_split_obs`,
and hand-rolled per-tick loops such as `v3_exq_541c`). No sleep pass calls `sense()`: its
only internal callers are the waking wrappers `sense_flat()` and `act_with_split_obs()`.
`serotonin_step()` also increments; double-counting is harmless by construction because the
predicate is only ever `_waking_ticks_since_capture == 0`, whereas under-counting silently
suppresses every capture.

### Direction not taken

"Exclude the `precision_init` sentinel from the first capture" by VALUE. Rejected: it needs
float equality against `1.0/rv_init`, and a genuinely-converged agent that happens to sit at
`precision_init` would be silently skipped. A value test cannot distinguish "no experience"
from "experience that landed on the sentinel"; the tick counter tests the actual predicate.

### Liveness evidence (ON vs OFF, real rollout)

Traced 8-episode probe, CausalGridWorldV2 size 8, K=1, recal step 0.25, seed 17, 30 waking
ticks/episode. 17 `enter_rem` calls (one pre-waking sentinel plus two per cycle); with the
guard ON it suppresses 9 and admits 8 -- exactly one genuine capture per cycle.

```
OFF targets: [2.1484, 2.3511, 2.5213, 2.7294, 2.7322, 2.8265, 2.9610, 3.0123]
ON  targets: [2.9191, 2.9834, 3.0381, 3.1279, 3.0942, 3.1205, 3.1787, 3.1910]
OFF fired  : 8/8      ON fired: 8/8      (not inert, and not a kill switch)
```

OFF cycle-1 target 2.1484 reproduces V3-EXQ-541c's documented 2.148. ON, the anchor is the
first REAL capture (2.9191 = 1/rv at REM entry) instead of the 2.0 sentinel.

### Validation experiment

V3-EXQ-541d (queued 2026-09-18): re-run of 541c's design with
`precision_zero_point_require_waking` as a second factor. Pre-registered prediction with the
guard ON: cycle 1 does not fire, and cycle 2+ targets sit within ~2x of
1/realized-PE-variance instead of climbing away from it. FALSIFIER OF THE WHOLE FIX: if with
the guard ON the target STILL climbs monotonically away from 1/realized-PE-variance over the
first ~10 cycles, the `precision_init` sentinel was not the (only) cause -- precision-at-REM-entry
is then not a calibration-relevant quantity at all, and MECH-204 Option A needs REDESIGN
rather than a cold-start guard. That outcome should demote Option A, not retune it.
