## MECH-307 from_dims() Reachability Repair (2026-08-07)
- MECH-307: affect.anticipatory_conjunction_architecture -- MASTER FLAG WAS UNREACHABLE
  through `REEConfig.from_dims()` from the 2026-05-11 landing until 2026-08-07. The
  substrate itself was correct and contract-verified the whole time; what was broken was
  the only route every consumer uses to ask for it.
  Defect: all TWELVE MECH-307 parameters were `REEConfig` dataclass fields with NO entry in
  the `from_dims()` signature, so each one fell into `**kwargs` and was silently discarded.
  `REEConfig.from_dims(..., use_mech307_conjunction=True)` returned a config with all four
  gaps OFF and raised nothing. This is the `[memory] reference-reeconfig-from-dims-silent-kwargs`
  failure mode: a `REEConfig` knob needs THREE sites and MECH-307 had only site 1.
    site 1  dataclass field on REEConfig             -- was present (config.py:5324)
    site 2  named parameter in from_dims() signature -- WAS MISSING, now added (all twelve)
    site 3  post-cls() re-apply of the __post_init__ resolver, inside from_dims
                                                    -- WAS MISSING, now added
  Site 3 is independently required even once site 2 exists: `from_dims` assigns its fields
  AFTER `cls()`, so `__post_init__` (which holds the OR-only master resolver,
  config.py:5427-5436) has already run against the defaults and the three substrate-side
  sub-flags are never forced True. Same shape as the MECH-090 re-apply and the GAP-3
  sleep-cluster re-apply, both of which already got this right -- MECH-307 was the odd one out.
  Modules:
    ree_core/utils/config.py -- `from_dims()` signature gains the twelve
      (use_mech307_conjunction, use_mech307_split_surprise, use_mech307_schema_multichannel,
      use_mech307_predicted_location_write, use_mech307_signed_pe,
      use_mech307_consumer_conjunction_read, mech307_anticipatory_liking_gain,
      mech307_z_beta_schema_gain, mech307_conjunction_gain,
      mech307_conjunction_wanting_threshold, mech307_conjunction_liking_threshold,
      mech307_conjunction_z_beta_threshold), plus a body assignment block and the resolver
      re-apply. No new config field; no substrate/mechanism change.
  Two placement decisions, both load-bearing and both pinned by contract:
    - The signature entries go AFTER `goal_stream_enabled` and immediately before `**kwargs`,
      so no existing positional argument index moves.
    - The body block goes BEFORE the `goal_stream_enabled` block, so `enable_goal_stream()`
      keeps the precedence it has always had as a bundle preset (it forces the three sub-flags
      plus the consumer read True and pins the gains/thresholds) -- the same ordering the
      MECH-295 block above it relies on. Moving the block after the preset would silently
      start returning bare parameter defaults for every goal-stream driver.
  NOT bit-identical, and that is the point: the 84 experiment drivers that pass
  `use_mech307_conjunction=True` into `from_dims` go from MECH-307 entirely OFF to ON. Prior
  results from those drivers -- including V3-EXQ-603q / 866a / 866b, the scaffolded lineage
  ARC-030's retest is built on -- were measured with MECH-307 OFF and must be read that way.
  Bit-identical for every driver that does not mention MECH-307: the twelve signature defaults
  match the dataclass defaults exactly (pinned by `test_defaults_parity_bit_identical_off`).
  Contract: `tests/contracts/test_mech307_from_dims_wiring.py` (19 tests, all pass). Asserts
  the FACTORY route specifically, each against a direct-construction parity case, so sites 2
  and 3 cannot regress independently -- verified differentially in a throwaway worktree:
  16 fail against pre-fix HEAD, and with site 2 present but site 3 removed exactly the 3
  resolver tests fail. The pre-existing MECH-307 contracts
  (`test_mech307_conjunction_contract.py`, `test_mech307_consumer_conjunction.py`) pass today
  and are sound -- they build a flagless config and `setattr` each SUB-flag afterwards, and
  sub-flag setattr works. The master-flag-through-the-factory route is what was untested,
  i.e. exactly how all 84 drivers ask.
  Out of scope, needs its own triage: `from_dims`'s `**kwargs` still silently swallows an
  unknown flag -- `from_dims(totally_bogus_flag_xyz=True)` raises nothing. That is the general
  mechanism, not a MECH-307 quirk; a reject-or-warn-on-unconsumed-kwargs guard would close the
  whole class but will surface other latent drops.
  Follow-on (deliberately NOT done here): ARC-030's Go/NoGo retest was blocked on this. After
  this repair it still needs a readiness step first -- re-run the 603q/866b configuration with
  MECH-307 genuinely ON and confirm the scaffolded lineage still clears the G0 non-degeneracy
  gate. Only then is the COMBINED-vs-NOGO_ONLY pair (pre-specified in ARC-030's
  `what_would_answer`) worth queuing.
  See MECH-307, MECH-090 (the correct from_dims re-apply precedent, config.py:7347), GAP-3
  sleep-aggregation cluster (the other precedent), MECH-295 (bridge consumer), ARC-030,
  `[memory] reference-reeconfig-from-dims-silent-kwargs`,
  `REE_assembly/evidence/planning/mech307_from_dims_unreachable_2026-08-07.md`.
