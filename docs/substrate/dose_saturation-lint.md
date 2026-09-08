## dose_saturation lint -- IMPLEMENTED (2026-07-22)
- dose_saturation lint -- IMPLEMENTED 2026-07-22. experiments/_lib/dose_saturation.py,
  stamped from manifest_core.stamp_recording_core beside stamp_inert_arm_knob.
  CATCHES: two DECLARED DOSE LEVELS whose per_level float readouts are equal beyond
  float noise. V3-EXQ-794 ran SD-076's asymmetry at LO=0.6 and HI=0.8 and got
  overconfidence_score = -1.004111904519277 at BOTH, plus calibration_ratio =
  2.7564936387545953 at both, because rv was clamped at a floor above the operating
  point. A genuine dose-response -- INCLUDING A GENUINELY NULL ONE -- gives different
  values at different doses with seed-level variance; agreement to the last bit means the
  quantity saturated before the dose could express itself.
  COST OF NOT HAVING IT: SD-076 was recorded does_not_support, charging a refutation to a
  claim whose lever never moved, and MECH-204's correction was left with no drift to
  correct. Both claims went untested while appearing tested, and it took a full autopsy
  to withdraw the direction.
  SIBLING, NOT DUPLICATE, of inert_arm_knob (c040d28): there the knob never reached a
  live code path so the arms RAN IDENTICALLY; here the knob DID move the dynamics and a
  bound downstream erased the difference. 794's arms are not bit-identical cell-wide, so
  inert_arm_knob does not and should not fire on them.
  Emits dose_levels_separable (bool) + dose_saturation_detail (offenders only, on the
  False verdict). NOT in ALWAYS_CORE_KEYS -- the pre-2026-07-22 corpus cannot carry it.
  POSTURE: record-and-WARN at write, gate at adjudication, same as its sibling -- by
  manifest-write time the compute is spent, and 794's green arms stayed scorable. The
  autopsy's "REFUSE the dose-response criterion" is honoured by emitting the flag for the
  experiment's own scoring to read.
  FALSE-POSITIVE DISCIPLINE (all pinned by contracts): tied INTEGERS never fire
  (n_seeds_overconfident = 0 at both levels is how a count says "no effect"); tied
  strings/bools never fire; exact 0.0/0.0 ties are recorded under zero_ties but do NOT
  flip the verdict (zero is overwhelmingly a not-applicable sentinel). Only a tie between
  two NON-ZERO floats fires -- which different trajectories do not produce by chance.
  Dose identification excludes EVERY fully-varying numeric key, not one: such a key can
  never appear in tied_fields anyway, so the exclusion is lossless and cannot manufacture
  the identity it reports (the reason inert_arm_knob had to reject its analogous
  inference). manifest["dose_key"] declares it explicitly.
  Validation: no EXQ -- it adjudicates nothing and gates no claim; verified by 22
  contracts in tests/contracts/test_dose_saturation_lint.py, including a replay of the
  real 794 per_level block to a firing verdict.
  Source: REE_assembly/evidence/planning/failure_autopsy_V3-EXQ-794_2026-07-22.md sec 6
  item 2.
