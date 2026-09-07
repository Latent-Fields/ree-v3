## MECH-307 Default-Value Recalibration (2026-05-12)
- MECH-307 default tweaks: TWO bridge-config defaults lowered after V3-EXQ-540c
  read-site probe (10x scale, 1087 bridge calls, 34784 candidate-reads) confirmed
  the V3-EXQ-540a / 540b conj_fire_rate=0 across all arms was caused by two
  config defaults sitting above the achievable substrate ceiling under the
  standard env config (CausalGridWorld with resource_respawn_on_consume=True):
    mech295_min_drive_to_fire        0.1 -> 0.01
    mech307_conjunction_z_beta_threshold  0.6 -> 0.3
  V3-EXQ-540c observations driving the change:
    drive_level: max=0.030 mean=0.016 frac>0.1=0.000 across 1087 bridge calls.
      The legacy 0.1 floor was NEVER crossed -- bridge short-circuited at
      mech295_liking_bridge.py:328-329 on every single call.
    z_beta_arousal: max=0.545 mean=0.518 frac>0.1=1.000. The legacy 0.6 floor
      sat above the achievable ceiling; ARM_default at 0.6 would never fire on
      this gate alone even if the drive gate were cleared.
    Predicate components at half-tier thresholds (0.3/0.15/0.3): all_pass=0.9466
      across 34784 candidates -- substrate writes populate read sites cleanly;
      the predicate WOULD have fired on 94.66% of reads if both gates allowed
      it through. At low-tier (0.1/0.05/0.1) and floor-tier (0.01/0.005/0.01)
      all_pass=1.000.
  Both changes are pure default-value adjustments to substrate-side config:
    ree_core/regulators/mech295_liking_bridge.py: MECH295LikingBridgeConfig
      dataclass defaults.
    ree_core/utils/config.py: REEConfig dataclass field defaults +
      REEConfig.from_dims kwarg default.
    ree_core/agent.py: getattr fallback used by REEAgent.__init__ when
      constructing the bridge config from REEConfig.
    Contract test assertions updated to match new defaults:
      tests/contracts/test_mech_295_liking_bridge.py (min_drive_to_fire == 0.01)
      tests/contracts/test_mech307_consumer_conjunction.py (z_beta default == 0.3
        + the C4 low-z_beta-blocks test uses z_beta_arousal=0.1 instead of 0.4
        to remain below the new default).
  Regression: 314/314 contracts + 7/7 preflight PASS with new defaults
    (verified 2026-05-12). The new defaults preserve the bridge's "some unmet
    need" + "some arousal" semantic while letting the bridge fire under
    realistic substrate output magnitudes.
  Backward compat: callers that explicitly set either flag to a custom value
    are unaffected (the defaults only apply when the caller omits the kwarg).
    Old 540a/540b scripts that explicitly set cfg.mech295_min_drive_to_fire=0.1
    on the cfg object after from_dims would still see the legacy behaviour;
    540e is specifically NOT overriding these to test the new defaults.
  Validation experiment: V3-EXQ-540e (3-arm decomposition under new defaults).
    Dry-run smoke 2026-05-12T06:39Z PASS at 6 ep / 1 seed: ARM_2_full
    conj_fire_rate=0.155 (>= 0.10 floor cleared at dry-run scale; ARM_0_off and
    ARM_1_split_only correctly zero). Full-scale run on DLAPTOP-4.local
    produces the definitive manifest at 3 seeds x 70 ep.
  Deferred follow-on (separate session): Option-b semantic fix at the bridge
    consumer-read site (line 343 of mech295_liking_bridge.py reads v[:, 3] =
    VALENCE_SURPRISE -- the LEGACY unsigned-magnitude channel -- rather than
    v[:, 4] = VALENCE_POSITIVE_SURPRISE under Option-b semantics; (v_s > 0.0)
    therefore loses its sign-filter intent and falsely satisfies on
    harm-paired magnitudes). Not a behavioural blocker (the v_s > 0 gate is
    cleared by Option-b's magnitude write anyway), but is a design-doc
    fidelity bug worth landing once 540e PASS confirms the architecture.
  See MECH-307, MECH-295 (bridge config owner), V3-EXQ-540c (probe diagnosis),
    V3-EXQ-540e (default-fix validation), goal_pipeline:GAP-1 (closure plan).
