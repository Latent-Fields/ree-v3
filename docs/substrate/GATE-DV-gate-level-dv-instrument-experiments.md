## Gate-level DV instrument (`experiments/_lib/gate_dv.py`, IMPLEMENTED 2026-08-19)

`GateDVRecorder` -- reads the ARC-110 cross-loop arbitration gate ONCE PER GENUINE E3
SELECTION. Use it for any DV about the arbitration gate; do NOT hand-roll a
`agent.e3.last_score_diagnostics` read in a driver loop.

- **What it repairs.** The two halves of this instrument previously lived in different
  drivers and were never combined: `v3_exq_707c` has the fresh-select repair (31 uses)
  but reads NO gate telemetry; `v3_exq_709/711/713` read the gate richly (23/21/21 uses)
  but predate the repair, so `diag = getattr(agent.e3, "last_score_diagnostics", {})`
  (`v3_exq_709:801`) runs on every env step. At `e3_steps_per_tick=10` that replicates
  each reading ~10x, unequally across arms. `clg_limbic_ge_motor_ticks` was the sharp
  case -- a COUNT with an env-step denominator. That hold-weighted C1 is why the
  709/711/713 exhaustion verdict was WITHDRAWN (2026-07-20) and why
  `substrate_queue.v4_loop_segregation` carries
  `713x_re_letter_STILL_REFUSED_corrected_DV_instrument_required`.
- **Not everything was wrong.** MAX-reductions (`*_peak`) are replication-INVARIANT and
  were already sound; counts/fractions/means/entropies are not. The recorder fresh-gates
  all of them anyway and emits mean AND peak so a consumer can see the two agree.
- **API.** `begin_episode()` / `with rec.watch(agent) as sel:` / `rec.record(agent, sel,
  committed_class=..., fallback=...)` / `end_episode()`, then `rec.as_dict()` +
  `rec.gate_readiness()`. Composes `_lib/fresh_select.py` (does not reimplement it) and
  reuses `v3_exq_707c._entropy_from_int_counts` semantics -- a contract extracts 707c's
  own function from source and asserts agreement, so the two cannot drift.
- **Saturation guard (the 711 lesson).** V3-EXQ-711 met `limbic_loop_can_win` 3/3 by
  SATURATION (`M_cross` range 4897.8; `w_eff[limbic]` up to 2274x motor) with entropy
  FALLING. `gate_saturated` fires above `w_eff_ratio_peak > 5.0` or
  `m_cross_range_peak > 50.0` (constructor-overridable), and `gate_ready` is AND-of-all,
  so a saturated cell self-routes `substrate_not_ready_requeue` -- never a false weakens.
- **A `loop_segregation_active == False` selection contributes no sample** (counting zeros
  would dilute every mean and manufacture a false "limbic never wins").
- Measurement-only: no `ree_core` change, no config flag, no substrate behaviour touched.
  PROMOTES NOTHING; it is the precondition the 713x re-letter was refused pending, not a
  re-letter. Regression guard: `tests/contracts/test_gate_dv_instrument.py` (14).
  See `REE_assembly/docs/architecture/learned_cross_loop_arbitration.md` (2026-08-19
  addendum).
