## DV-headroom class (`validate_experiments.py` lint + `experiments/_metrics.py` precondition kind, IMPLEMENTED 2026-09-04)

Substrate entry `dv-dynamic-range-precondition-class` (priority 1, severity DEGRADING),
from the confirmed cluster autopsy
`REE_assembly/evidence/planning/failure_autopsy_ext-claim-probe-cluster_2026-09-03.md`.
**Declare a `dv_headroom` precondition on any new driver whose load-bearing criterion
reads a difference, spread or elevation statistic.**

- **What it repairs.** Every readiness gate in this corpus certifies the INTERVENTION --
  was the channel perturbed, did the head train, were there enough samples -- and NONE
  certifies that the DEPENDENT VARIABLE had room to move. Six of the seven 2026-09-03
  pending-review runs passed ALL their preconditions and still discriminated nothing,
  because the registered pass threshold lay outside the range the configuration could
  produce: 981 C1 needed 1.154 from a DV bounded in [0,1]; 981 precision-margin 0.01 floor
  against 0.000195 available (51x); 983 decline_gap realised range 0.0468 vs 0.15 (3.2x);
  993 max |calibration_gap| 0.00152 vs 0.02 (13.1x); 994 retention spread 0.00078 vs 0.02
  (25.6x); 978 an arm difference one third of the DV's quantum; 951c zero reachable ticks.
- **Runtime half (opt-in).** `_metrics.dv_headroom_check(name, dv_name=..,
  criterion_threshold=.., control_values=.., statistic=.., margin=1.0)` ->
  `p0_readiness_gate(...)`. It rides the gate's existing single-bound path -- `measured` =
  what the DV can achieve, `threshold` = what the criterion requires, floor -- so an unmet
  entry raises `P0NotReady` and the caller writes the `substrate_not_ready_requeue`
  manifest it already writes. Pass the criterion's OWN module constant as
  `criterion_threshold`, never a re-typed literal, so the gate cannot drift from the
  science it guards. `margin > 1.0` demands resolving room, not just parity.
- **Four statistics, not interchangeable.** `range` (max-min: 983/994's shape), `max_abs`
  (993's signed gap vs an absolute floor), `ceiling_headroom` (981's saturated baseline --
  needs `dv_bounds`), `floor_headroom` (the suppression mirror). Or pass `achievable=` for
  a DV whose reachable range is analytic rather than sampled (951c's zero ticks).
- **No indexer change, by design.** The REE_assembly indexer recomputes `met` from
  (measured, threshold, direction) and is kind-agnostic, so an unmet entry adjudicates as
  `precondition_unmet` with nothing added downstream. Governance scoped this build to the
  HARNESS precisely so it could not perturb the 1,201 drivers importing the substrate --
  the original autopsy recommendation (severity `corrupting`, `substrate_paths` including
  `ree_core/environment/causal_grid_world.py`) would have blocked every new experiment
  fleet-wide at Step 2.5c.
- **Static half (WARN-only).** `--checks criterion_exceeds_achievable_range` flags a
  LOAD-BEARING criterion, or a readiness precondition, whose threshold's feasibility
  nothing establishes: (a) a multiplicative threshold on a unit-interval DV, (b) an
  absolute floor on a derived-range statistic. It cannot PROVE unreachability -- the
  baseline is a runtime quantity -- so it never hardens the exit code, in any mode. Any
  mention of `dv_headroom` in the file silences it (deliberately generous: the point is to
  make the author answer the question, not to police how); `CRITERION_ACHIEVABLE_RANGE_EXEMPT
  = "<reason>"` is the explicit opt-out. Fires on 112/1448 drivers (7.7%) at landing.
- **Restricted to `load_bearing: True` criteria on purpose.** Scanning every
  criterion-shaped assignment fires on 210 drivers (14.5%); the tagged restriction fires on
  112 while KEEPING both known carriers (981, 983). Noise removed, signal kept.
- **Precedent.** V3-EXQ-777a hand-rolled this guard locally (`score_dv_headroom_seeds`,
  whose own `control` string calls it "the guard V3-EXQ-777 lacked"). One driver inventing
  it after losing a run is the argument for putting it where the next driver inherits it.
- Harness-only: no `ree_core` change, no config flag, no substrate behaviour touched.
  Byte-identical for every driver that does not opt in. Regression guard:
  `tests/contracts/test_criterion_exceeds_achievable_range_lint.py` (47).
