## SD-104 / SD-105: phasic burst refractory duty bound + selection-entropy headroom floor (the two coupled regulator defects blocking MECH-063 (ii)) -- IMPLEMENTED (2026-09-04)
- SD-104: phasic.burst_refractory_duty_bound + SD-105:
  control_plane.selection_entropy_headroom_floor -- IMPLEMENTED 2026-09-04. Legs (a) and (b) of
  substrate_queue entry `sd_phasic_burst_decay_and_warmup_headroom` (priority 1, severity
  DEGRADING), created by governance `governance-20260903T2013` from the confirmed autopsy
  `REE_assembly/evidence/planning/failure_autopsy_V3-EXQ-963a_2026-09-02.md`. Design docs:
  `REE_assembly/docs/architecture/sd_104_phasic_burst_refractory_duty_bound.md` +
  `sd_105_selection_entropy_headroom_floor.md`.
  The autopsy's two pillars are BOTH regulator behaviour, not sampling -- raising
  `MAX_ENV_STEPS_PER_CELL` cannot reach either, and its re-derive brake REFUSES a 963-lineage
  successor that tries. Both legs must be shown fixed before V3-EXQ-963b is admissible.

  LEG (a) -- SD-104, `ree_core/regulators/phasic_surprise_burst.py` (extends SD-069/SD-075 in the
  same file, following the SD-075 precedent). THE DEFECT: `tick()` re-arms the envelope with
  `max(decayed, drive)` on EVERY firing tick, so on a warmed agent (SD-074) with
  `signal_source="instantaneous_pe"`, `trigger_ratio=1.2`, `decay=0.5` a fresh event lands before
  the previous decays and the burst occupies 0.390-0.884 of E3 selections (779a, colder agent:
  0.007-0.136; seed 23 T1P1 fired on 1489 of 1684). A "transient" at 88% is quasi-sustained, so
  MECH-063 (ii) has no separable transient to measure.
  Config: `phasic_burst_refractory_ticks` (int, 0 = off) suppresses event FIRING for N ticks after
  an event while the envelope keeps decaying (carry-mode decay); `phasic_burst_extinction_level`
  (float, 0.0 = off) snaps the envelope to exactly 0.0 below that level, making "active" a crisp
  regulator-side predicate instead of a floor each consumer picks (963a's driver used
  `EVENT_LEVEL_FLOOR = 0.05` and could not know whether the regulator agreed).
  THE GUARANTEE (the property the autopsy asked to be ASSERTED): `A = 1 + floor(ln(extinction) /
  ln(1-decay))`, `duty <= min(1.0, A / (refractory_ticks + 1))`. `get_state()` reports
  `burst_duty_cycle_bound`, `realised_burst_duty_cycle`, `max_active_ticks_per_event` and
  `burst_duty_cycle_within_bound` (None, never True, when no finite bound exists). NEITHER KNOB
  ALONE SUFFICES -- a refractory without extinction leaves a tail that never reaches zero (bound
  unprovable); extinction without a refractory still re-arms on the next tick.
  THE REFRACTORY COUNTER IS LIFETIME AND CARRIES ACROSS `reset()`, found by this build's own
  smoke test rather than by reasoning: cleared per episode, a refractory of 29 + extinction 0.05
  (bound 0.167) produced a realised duty of 0.311 over 61 lifetime ticks of short episodes --
  every boundary re-armed immediate firing, so duty was again a function of episode LENGTH (the
  V3-EXQ-779b confound SD-075 exists to close, on a new axis) and the lifetime bound was false.
  The envelope IS still cleared at reset; only the refractory owed carries.
  DELIBERATELY NOT BUILT: a duty-cycle TARGET with a controller tuning the refractory. The
  closed-form bound is a hard ceiling, which is what an assertion needs; a controller would make
  realised duty a function of the surprise stream again -- exactly what made 963a unmeasurable.

  LEG (b) -- SD-105, NEW `ree_core/regulators/selection_entropy_floor.py`
  (`SelectionEntropyFloor`). THE DEFECT: the SD-074 warmup succeeds, and a confident policy has
  almost no selection entropy left to move -- T0P0 baseline 0.0195-0.153 against 779a's
  0.152-0.610, a 3-26x collapse on EVERY seed, pinning R5 at its 0.02 floor. Worse than the gate
  failing: the phasic lever SHARPENS, so a 0.0195 baseline leaves ~2% of the readout's range for
  the manipulation -- the anti-conservative direction `experiments/_lib/entropy_headroom.py`
  already warns about.
  THE AUTOPSY LEFT THE CHOICE OPEN ("either the warmup must leave headroom or R5's band must be
  re-derived; state which and why"). THIS TAKES THE FIRST BRANCH; re-deriving is REJECTED because
  (1) it would let the gate pass while the dynamic-range condition it detects is untouched --
  converting an artifact into a citable result, the warning `precondition_gate.py` carries and the
  same failure the `dv-dynamic-range-precondition-class` gate (ree-v3 8e133d26ed) catches from the
  other end; and (2) the collapse is a property of ANY sufficiently trained policy, not of one
  warmup recipe, so a `probe_warmup` fix is a band-aid the next lineage rediscovers.
  MECHANISM: a ONE-SIDED integral controller in log-temperature. Read the realised normalized
  entropy of the PREVIOUS tick's E3 pre-commit distribution (`e3.last_precommit_probs` -- previous
  because this temperature is an INPUT to the current tick's softmax; one tick of lag,
  deliberately slow), smooth it, raise `log_mult` while below target, relax (never below 0) above
  target + deadband, clamp to `[0, ln(max_temperature_ratio)]`, emit `multiplier = exp(log_mult)
  >= 1.0`.
  Config: `use_selection_entropy_floor` (False), `selection_entropy_floor_target` (0.15 -- inside
  the R5 band and inside 779a's healthy 0.152-0.610), `_gain` (0.5), `_max_temperature_ratio`
  (8.0), `_ema_decay` (0.2), `_deadband` (0.05).
  APPLIED ON THE TONIC SIDE: `temperature -> MECH-313 noise_floor -> [SD-105 multiplier] ->
  SD-069 phasic delta -> e3.select()`. Before the phasic delta so the phasic contribution stays an
  ADDITIVE delta in absolute temperature units on a lifted baseline (rescaling would compress the
  event-locked transient MECH-063 (ii) measures); enabled identically in every arm of a tonic
  contrast, both arms lift together, so `dS_tonic` is preserved rather than compressed.
  `noise_floor_temp` deliberately keeps reporting the PRE-multiplier tonic value so the MECH-313
  readout stays uncontaminated; the multiplier is reported separately as
  `_last_control_vector["entropy_floor"]`.
  TWO LOAD-BEARING PROPERTIES: ONE-SIDEDNESS (never below 1.0 -- a controller allowed below would
  be an entropy REGULATOR clamping the readout from both sides, destroying the dynamic range this
  protects and silently cancelling a tonic manipulation); and THE CAP REPORTS RATHER THAN HIDES
  (`saturated` True with `headroom_met` False means the policy is too confident for this readout
  at this budget -- declare the cell UNINFORMATIVE, do not report the compressed number).
  THE EMA AND INTEGRATOR SURVIVE `reset()`, the opposite of SD-069's default and deliberate: a
  set-point re-converging from cold each episode measures episode LENGTH rather than confidence
  (the 779b confound again). `get_state()` carries `continuity_note:
  "ema_and_integrator_survive_reset"` because a reader is entitled to be surprised.

  Both: pure-arithmetic regulators, no `nn.Module`, no learned parameters, no gradient flow (same
  category as MECH-313 `noise_floor` / SD-069). PHASED TRAINING DOES NOT APPLY. MECH-094:
  `simulation_mode=True` advances nothing in either (replay/DMN must not consume a refractory tick
  or move the waking exploration set-point).
  Backward compatible: all knobs no-op by default. VERIFIED DIFFERENTIALLY, not asserted -- a
  seeded action/temperature/burst-level trace over 150 ticks x {phasic OFF, phasic ON} x seeds
  {11, 23} is BIT-IDENTICAL to a throwaway worktree at `origin/main` (8e133d2).
  Suggested operating point for a warmed agent at `decay=0.5`, `EVENT_LEVEL_FLOOR=0.05`:
  `extinction_level=0.05`, `refractory_ticks=29` (A=5, bound 0.167). Measured on a broad
  heavy-firing surprise stream (1000 ticks, uniform(0,10)): realised duty 0.788 with the knobs at
  defaults vs 0.109 ON -- the OFF configuration reproduces the 963a regime (0.390-0.884) and the
  ON configuration lands inside 779a's healthy band (0.007-0.136). NOTE for anyone writing a
  probe here: a CONSTANT spike stream is the wrong adversary -- its EMA converges to the spike and
  firing stops on the TRIGGER test within ~30 ticks (measured 0.016 duty), so it tests nothing
  about the refractory.
  Contracts: `tests/contracts/test_sd104_sd105_burst_decay_and_entropy_headroom.py` (A1-A11,
  B1-B10). A6 is the POSITIVE CONTROL and is load-bearing: it fails if the OFF configuration stops
  reproducing the 963a regime, so the suite cannot pass on a regulator that simply never fires.
  Validation experiment: V3-EXQ-963b (MECH-063 (ii) tonic/phasic dissociation retest, `supersedes`
  V3-EXQ-963a) -- see the queue entry. Unblocks MECH-063 and SD-069 (both
  `pending_retest_after_substrate`).
  See MECH-063 (sub-claim ii), SD-069 (the module SD-104 extends), SD-075 (baseline continuity),
  SD-074 (the warmup whose success creates the SD-105 defect), MECH-313 (tonic counterpart),
  `dv-dynamic-range-precondition-class` (the harness gate that catches the same failure from the
  criterion side), MECH-094.
