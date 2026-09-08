## SD-076: precision.waking_confidence_inflation -- IMPLEMENTED (2026-07-20)
- SD-076: precision.waking_confidence_inflation -- IMPLEMENTED 2026-07-20.
  Design doc: REE_assembly/docs/architecture/sd_076_waking_confidence_inflation.md.
  Modified: E3TrajectorySelector.update_running_variance in
  ree_core/predictors/e3_selector.py (asymmetric EMA).
  Config: E3Config.use_waking_confidence_inflation (default False),
  E3Config.waking_confidence_inflation_asymmetry (default 0.0, range [0,1)),
  E3Config.waking_confidence_rv_floor (default 0.01).
  PROBLEM: the symmetric EMA makes _running_variance a faithful tracker of true
  prediction error, so rv ~= true error BY CONSTRUCTION and the MECH-173
  overconfidence_index is pinned near zero no matter what is ablated (V3-EXQ-774
  measured -0.000148 / -0.000918 on the suppressed arms). MECH-204's corrective
  function presupposes a daytime drift source V3 did not have, so a null on any
  MECH-204 consumer was a TAUTOLOGY, not evidence.
  MECHANISM: good news incorporated fast (alpha * (1 + asym) when error
  improves), bad news slowly (alpha * (1 - asym) when it worsens), so rv settles
  BELOW the true error mean = genuine, directional, correctable overconfidence.
  BIT-IDENTICAL OFF BY CONSTRUCTION, NOT BY ARITHMETIC: the OFF branch evaluates
  the original symmetric expression unchanged rather than re-deriving it at
  asymmetry 0. Pinned by explicit float equality over an 80-step trace, not an
  approximate comparison.
  THE FLOOR IS LOAD-BEARING, NOT HYGIENE: rv feeds an ABSOLUTE commit threshold
  (running_variance < commit_threshold, ARC-016) and current_precision =
  1/(rv + 1e-6). Unbounded downward drift would pin the agent permanently
  committed AND explode precision. Applied only on the inflation path.
  Biological basis: optimism / positive-outcome bias in waking belief updating
  -- the drift the sleep-recalibration literature behind MECH-204 presupposes.
  Functional translation, not a neuromodulator-specific mechanism claim.
  Backward compatible: disabled by default; existing experiments unaffected.
  Phased training required: no. MECH-094: not applicable.
  Smoke: 6/6 PASS. At asymmetry=0.6 on true error mean 0.05,
  overconfidence_index moves -0.164 (OFF, UNDERconfident) -> +0.273 (ON). The
  OFF value reproduces the sign and rough magnitude of 774's measured
  ARM_FULL_SLEEP = -0.2097, which is direct evidence the autopsy diagnosis
  was right.
  SD-069 unaffected: last_instantaneous_pe is captured BEFORE this smoothing.
  HEADROOM REPAIR 2026-07-22 -- READ THIS BEFORE USING THE LEVER. The absolute
  floor above is the WRONG KIND OF QUANTITY, and 0.01 is the wrong value for this
  substrate. V3-EXQ-794 measured the un-inflated operating point at rv = 0.005420
  (ARM_OFF_OFF) and arm_true_error_ref at ~0.0037, so 0.01 sits 1.8x ABOVE the
  operating point: max(0.01, rv) clamps on the first tick inflation bites and
  never releases. rv_final was EXACTLY 0.010000 on all four inflation arms and
  overconfidence_score was bit-identical to 15 significant figures
  (-1.004111904519277) at asymmetry 0.6 AND 0.8. Two doses giving one value is a
  SATURATION signature, not a null -- SD-076 and MECH-204 both went untested, and
  SD-076's recorded does_not_support was withdrawn to non_contributory by
  failure_autopsy_V3-EXQ-794_2026-07-22.
  WHY THE 2026-07-20 SMOKE PASSED 6/6 ANYWAY: it used true error mean 0.05, ~13x
  the substrate's real scale, where 0.01 IS a floor with headroom. An absolute
  constant validated at one scale silently became a clamp at another. THE SIGN IS
  NOT THE BUG (the autopsy's candidate cause (b) is ruled out): inflation drove rv
  DOWN correctly, into a floor sitting ABOVE the OFF arm, so the run's
  inflation_lowers_rv precondition saw rv rise.
  New config, both no-op at default so the ON path stays bit-identical until set:
  E3Config.waking_confidence_rv_floor_relative_frac (default 0.0 = use the
  absolute floor; >0 makes the bound that FRACTION of _wci_symmetric_rv_ref, the
  counterfactual un-inflated rv, so it scales with the substrate's own error
  scale and caps overconfidence at 1 - frac),
  E3Config.waking_confidence_rv_floor_mode ("hard" default = the original clip |
  "soft" = softplus approach) and E3Config.waking_confidence_rv_floor_softness
  (0.25, knee width as a fraction of the effective floor; inert while "hard").
  WHY SOFT MATTERS beyond biology (waking confidence drift is softly bounded):
  the softplus is STRICTLY MONOTONIC, so a residual saturation can only SHRINK a
  dose separation, never collapse it to an exact tie -- a mis-set floor then
  degrades to a small LO/HI gap the dose_saturation lint can see, instead of the
  bit-identical arms that made 794 unadjudicable without an autopsy.
  New state: E3TrajectorySelector._wci_symmetric_rv_ref (exposed as
  .wci_symmetric_rv_ref), a symmetric EMA of the same error at the same alpha,
  advanced ONLY on the inflation path. rv minus it IS the inflation produced.
  Smoke at the MEASURED 794 scale (true error 0.0037), 14/14 PASS: the OLD config
  reproduces the defect exactly (LO == HI == 0.01); the repaired config gives
  rv_final 0.0025377 (LO) vs 0.0021031 (HI), dose-ordered, both genuinely
  overconfident (+0.314 / +0.432). Contracts:
  tests/contracts/test_sd076_rv_floor_headroom.py (17), which pins the DEFECT too
  so a "simplification" back to an absolute hard floor fails there, not in a run.
  Validation experiment: V3-EXQ-794a (same-question re-run of the identical 2x2).
  See MECH-173, MECH-204, ARC-016, Q-042, SD-069.
