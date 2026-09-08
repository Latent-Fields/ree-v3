## mech457_approach_extinction: experiments/_lib approach-drive extinction-on-contact -- IMPLEMENTED (2026-07-25)
- mech457_approach_extinction: experiments/_lib approach-drive extinction-on-contact -- IMPLEMENTED
  2026-07-25. experiments/_lib/mech457_explorer_classes.py (train_a2c) +
  experiments/_lib/mech457_bootstrap_explorer.py (BootstrapExplorerConfig, train_bootstrap_explorer).
  Completes the DRIVE half of competence_floor retention leg 4 (H-consummation-binding). The
  mech457_consummatory_act ENV node (2026-07-25) made contact AFFORD rather than EFFECT
  consumption and emits info["on_consumable_resource"], but NOTHING consumed that signal:
  V3-EXQ-781's appetitive approach primitive (mech.resource_proximity) reads obs_dict, which does
  not carry the flag, and train_a2c never threaded info to the approach hook -- so the drive could
  not extinguish on contact. goal.py's homeostatic drive (drive_ema_alpha/drive_floor/
  per_axis_restoration_fraction) is a DIFFERENT system, does not reference on_consumable_resource,
  and is not imported anywhere in this path (GOAL_DIM=2 here is spatial nav to a target cell).
  Config: BootstrapExplorerConfig.approach_extinguishes_on_contact (bool, default False;
  declared in as_slice()). The env's consummatory_act_enabled is a DIRECT ENV CONSTRUCTOR kwarg
  set by the driver in env_kwargs (NOT a config field -- this config does not build the env).
  Data flow: cfg.approach_extinguishes_on_contact -> train_bootstrap_explorer -> train_a2c: on
  each tick where info["on_consumable_resource"] is True the approach reward is zeroed, so the
  appetitive drive terminates on arrival and hands off to the distinct CONSUME act (the treatment
  arm); 781's non-extinguishing terminal drive is the control.
  HALF-WIRED IS AN ERROR (train_a2c raises): approach_extinguishes_on_contact=True requires (1) an
  approach_drive (use_approach_primitive=True) -- extinction with no drive is the control wearing
  the treatment label; (2) the env built with consummatory_act_enabled=True -- else
  on_consumable_resource is always False and extinction silently never fires.
  Backward compatible: default False -> the extinction branch never fires, byte-identical to the
  pre-change non-extinguishing drive (contract E1 asserts weight-identity; 788 dry-run clean).
  Fingerprint: edits experiments/_lib/**, so pre-change baseline arm fingerprints are correctly
  refused for reuse across the change (expected, not a regression -- same as the consummatory_act
  and retention_probe nodes); as_slice() gains one declared key because an extinguishing and a
  non-extinguishing cell are not interchangeable ARTIFACTS.
  Phased training required: no (no head trained). MECH-094: not applicable (no simulation/replay).
  7 new contracts E1-E6: tests/contracts/test_mech457_approach_extinction.py (default no-op
  byte-identical; extinction fires and changes policy on a consummatory env with proven contact;
  both half-wired guards raise; as_slice declares + defaults False; config-level half-wired via
  train_bootstrap_explorer; ASCII-only). All 7 pass locally (3.72s).
  Companion build (same day): consummatory-aware reference/demonstrator policies (below) -- the
  retention (BC-install) framing of leg 4 needs them so the install can take in the consummatory
  env.
  Validation experiment: V3-EXQ-821 (H-consummation-binding leg 4, retention/BC-install framing)
  queued via /queue-experiment -- the behavioural leg IS the validation.
  MECH-457 stays candidate/v3_pending; INV-088 unchanged; this build promotes and demotes nothing.
  See REE_assembly/docs/architecture/sd_mech457_approach_extinction.md,
  REE_assembly/docs/architecture/sd_mech457_consummatory_act.md, and
  REE_assembly/evidence/planning/mech457_retention_portfolio_2026-07-18.md (leg 4).
