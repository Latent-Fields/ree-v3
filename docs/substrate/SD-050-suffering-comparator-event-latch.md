## SD-050 Suffering-Comparator Event Latch: one relief event per descent (2026-09-26)

- suffering-derivative-comparator-refractory (SD-050 / MECH-302 amend) -- IMPLEMENTED 2026-09-26.
  `SufferingDerivativeComparator` in `ree_core/comparator/suffering_derivative_comparator.py`.
  Config: `REEConfig.suffering_event_latch_enabled` (default False; set True to enable),
  `REEConfig.suffering_rearm_rise` (default None -> uses `suffering_drop_threshold`).
  Both wired through `from_dims()` and passed to the comparator in `REEAgent.__init__`.
  Problem: `tick()` fires on EVERY tick whose rolling-window drop clears the threshold, so
  one damage->heal trajectory emits a TRAIN of relief-completion events (~9-17 per
  scheduled injection in V3-EXQ-517d; failure_autopsy_gflag0452-D1-cluster_2026-09-24),
  each releasing beta and writing VALENCE_LIKING at a different z_world.
  Mechanism (re-arm-on-rise latch): after a fire the comparator latches and tracks the
  lowest norm since the fire; it re-arms only when the norm rises `suffering_rearm_rise`
  above that trough (a new harm onset), and on re-arm restarts its window from that peak,
  so the next event needs a full fresh window of descent. The first event of a descent
  fires on exactly the tick the unlatched comparator fires; latched fires are a strict
  SUBSET of unlatched fires (pinned by contract R4). `select_action()` is unchanged:
  fewer True ticks means fewer beta releases and VALENCE_LIKING writes.
  Diagnostics: `comparator.fire_count`, `comparator.suppressed_count` (per episode;
  cleared by `reset()`), `comparator.latched`.
  Data flow: z_harm_a.norm() -> SufferingDerivativeComparator.tick() [+ latch] ->
  agent._relief_completion_event -> select_action(): beta_gate.release() + VALENCE_LIKING
  write (unchanged) -> SD-051 ConditionedSafetyStore samples ONE z_world per descent.
  Backward compatible: latch OFF is bit-identical to the pre-2026-09-26 tick() (contract
  R2 replays noisy multi-descent streams against a verbatim legacy oracle).
  MECH-094: simulation_mode ticks return False and leave buffer + latch untouched (R6).
  Biological basis: relief DA is a phasic OFFSET transient (Navratilova 2012), not a
  sustained train across the healing descent. Timing choice: first-crossing (onset of
  the counted descent), NOT end-of-descent -- offset-timing would move event timing and
  change what the safety store samples, which is a separate design question.
  Phased training required: no (non-trainable).
  Contracts: `tests/contracts/test_suffering_derivative_comparator_refractory.py` (16).
  Liveness: see "Liveness" below.
  Validation experiment: V3-EXQ-1110 queued 2026-09-26 (diagnostic; latch OFF vs ON with a paired
  shadow-unlatched comparator on the same norm stream; seeds 45/46/47).
  Scope NOT addressed: MECH-302 D2 / SD-050 "Required event" demand relief produced by the
  agent's own action; CausalGridWorldV2 heals passively. Registered as substrate_queue
  `action-contingent-relief-provider` (ready: false). See MECH-302, SD-050, MECH-304, SD-051.

### Liveness

Live consumer: `REEAgent.select_action()` relief block (beta_gate.release() when elevated +
`ResidueField.update_valence(VALENCE_LIKING)`), and SD-051 `ConditionedSafetyStore.update()`
in `sense()` (samples z_world on event ticks).

Measured ON vs OFF on REAL agent rollouts (V3-EXQ-1110 driver, `--dry-run --seeds S
--episodes 20`, 2026-09-26; CausalGridWorldV2 + SD-022 scheduled-limb-damage curriculum,
num_hazards=1, contamination_spread=0.0, window 30 / threshold 0.005 / min_norm 0.01):

| seed | arm | fires | shadow (unlatched) fires | shadow bursts | fires / burst |
|---|---|---|---|---|---|
| 42 | latch OFF | 57 | 57 | 8 | 7.1 |
| 42 | latch ON | 7 | 57 | 8 | 0.875 |
| 43 | latch OFF | 590 | 590 | 20 | 29.5 |
| 43 | latch ON | 20 | 590 | 20 | 1.0 |

Relief-block calls track fires exactly in every cell (57 vs 7; 590 vs 20).

Regime caveat (measured): at V3-EXQ-517d's own settings (num_hazards=3, contamination on)
the current substrate ends every episode in death 12-75 steps in, before any heal descent,
and the 517d driver records 0 events in both comparator arms -- the train that motivated
this build does not reproduce there today; the 1-hazard / no-contamination regime is where
descents occur. With limb_damage_enabled, harm_obs_a IS the limb-damage vector (healed
monotonically), so the norm rises only on damage and a "descent" is delimited by a rise.

Finding (red-team F1, 2026-09-26): in a sense->select_action loop that never calls
`agent.update_residue`, `ResidueField.update_valence` is a silent no-op (no active RBF
center), so the relief block's VALENCE_LIKING write lands nowhere. Measured: 0 inner writes,
`rbf_field.active_mask.any()` False. The same holds for V3-EXQ-517c/517d's "writes" metric.
