## mode-governance-engagement: regime-occupancy gradedness is a REPRODUCIBILITY test, not an existential one (2026-09-11)

- mode-governance-engagement (instrument half) -- IMPLEMENTED 2026-09-11. PROMOTES
  NOTHING; MECH-266 stays provisional / SD-032a stays stable, and the entry's
  `status`, `severity`, `ready` and `depends_on_unresolved` are UNCHANGED (governance
  adjudicates those, not this session).
  Module: `experiments/_lib/regime_occupancy_gate.py` (measurement instrument, NOT
  `ree_core/` substrate -- there is no config flag and no agent-loop wiring here).
  Contracts: `tests/contracts/test_regime_occupancy_gate.py` (17 tests; replays the
  confirmed cell data from V3-EXQ-464e, V3-EXQ-467e and V3-EXQ-934).

  WHY (the successor defect, severity `corrupting`): the file landed at `bd59f7e` was
  itself the fix for the entry's ORIGINAL `min()`-across-arms defect, and that half is
  correct and untouched. But its own "graded" predicate was a flat per-cell
  EXISTENTIAL --

      if any(floor < f < ceiling for f in fracs): return "graded"

  -- carrying neither seed identity nor sweep position, so ONE mixed cell out of N set
  `route_reason = None` (gate PASSES, regime declared genuinely graded). The entry's
  `severity_note` recorded this and it was confirmed still live in the code today.

  THE SUBTLETY THAT MAKES THE OBVIOUS FIX INSUFFICIENT -- and the reason the dataclass
  had to change rather than only the predicate: V3-EXQ-934's driver did NOT trust one
  cell. It called the gate once per (seed, arm) and applied its own `>= 2/3 seeds` rule
  on top of the per-seed booleans. It STILL routed
  `cap_recalibration_admits_mixed_regime`, from this confirmed data
  (`v3_exq_934_mech266_cap_sweep_mode_occupancy_20260815T015216Z_v3`):

      seed 42  mixed caps {0.75}   -> per-seed graded = True
      seed 43  mixed caps {1.75}   -> per-seed graded = True
      seed 44  mixed caps {}       -> per-seed graded = False   => 2/3 => PASS

  The two seeds were mixed at DISJOINT, opposite ends of the cap sweep; there is no
  common cap. The per-seed existential had already discarded WHICH cap was mixed, so no
  amount of counting seeds afterwards could recover it, and the `[min, max]` band
  summary hid it too. Reproducibility is therefore only answerable in ONE place, over
  ALL (seed, sweep_value) cells at once.

  THE BAR (transcribed from the entry's own two OPEN failure_record targets, not chosen
  here): V3-EXQ-467e asks for occupancy strictly in (0.1, 0.9) "on >= 2/3 seeds at >= 2
  adjacent hysteresis ratios"; V3-EXQ-934 asks for "a COMMON cap value ... on >= 2/3
  seeds ... with a mixed band at least 2 grid steps wide". Same shape, now implemented
  as: GRADED iff a run of >= `min_adjacent` (2) CONSECUTIVE swept values each reads
  mixed on >= `min_seed_fraction` (2/3) of the seeds measured at that value.

  Data flow: per-(seed, sweep_value) occupancy fractions -> `OccupancyCell(label,
  fraction, seed, sweep_value)` -> `classify_regime_shape` -> `regime_shape` +
  `route_reason` -> the driver's manifest `interpretation.occupancy_gate`.

  Taxonomy is now FIVE shapes. The original three keep their exact meanings; two are
  added because the stronger predicate can distinguish cases that fit none of them, and
  folding either into `saturated_bimodal` would state something false (that label means
  "no cell anywhere is mixed", which is not true of the 934 data):
    unreachable            -- every cell at/below floor.
    saturated_bimodal      -- reachable, no cell in the mixed band (464e/467e).
    mixed_not_reproducible -- mixed cells exist but fail the seed/adjacency bar (934).
    graded                 -- the bar is met. The ONLY shape with route_reason None.
    underdetermined        -- mixed cells exist, cells cannot support the bar.

  FAILS CLOSED. `seed`/`sweep_value` are optional on the dataclass so existing callers
  still CONSTRUCT, but a call that cannot support the bar does not fall back to the old
  existential -- it reads `underdetermined` with a non-None `route_reason`. A
  `min_seeds` floor (2) is what stops the 934 CALL SHAPE from re-creating the defect:
  a per-seed call has n_seeds == 1, and 1-of-1 would otherwise satisfy any seed
  FRACTION.

  CALL SHAPE CHANGED -- read before porting a driver: call ONCE over ALL (seed,
  sweep_value) cells; do not call per-seed and count booleans afterwards. For an
  unordered ARM CONTRAST leave `sweep_value` unset and pass `min_adjacent=1`; adjacency
  is then skipped and reported as `adjacency_evaluated: False` (464e's own target has no
  adjacency clause).

  `min_fraction` remains DIAGNOSTICS-ONLY and must never gate -- re-gating on it is the
  original defect. `reproducible_band` is reported only when the adjacent run actually
  met `min_adjacent`; a one-condition run reports `None` plus
  `longest_adjacent_run: 1`, because `[x, x]` reads as success to anyone skimming a
  manifest.

  Backward compatible at CONSTRUCTION, deliberately NOT at VERDICT: the two landed
  importers (`v3_exq_934_*.py`, `v3_exq_935_*.py`) still construct and run, but a
  metadata-free call that previously read `graded` now reads `underdetermined`. That
  behaviour change is the point. Neither driver is queued, and 935's use is marked
  "information-only, explicitly NOT load-bearing" in its own source.
  Phased training required: no (no encoder, no training).
  MECH-094: not applicable (no simulation/replay content written to memory).
  Validation experiment: none queued by this session -- see the entry's
  `depends_on_unresolved` (V3-EXQ-935) and the separate open chip
  `chip-20260909-exq935a-margin-cap-rerun`. The instrument's own validation is the
  17-test contract file, which replays confirmed run data rather than synthetic cases.
  See MECH-266, SD-032a, `failure_autopsy_mech266-464e-467e-cluster_2026-08-13.md`,
  `failure_autopsy_V3-EXQ-934_2026-08-16`.
