## mode-governance-engagement: affinity-input saturating squash bounding operator (2026-09-19)

- mode-governance-engagement item (1): `cingulate.salience_coordinator.affinity_bound_operator` -- IMPLEMENTED 2026-09-19.
  `ree_core/cingulate/salience_coordinator.py` (`bound_affinity_input()`,
  `AFFINITY_BOUND_CLAMP` / `AFFINITY_BOUND_SQUASH`, `SalienceCoordinator.tick()`),
  wired through `ree_core/utils/config.py` and `ree_core/agent.py`.

  Config: `SalienceCoordinatorConfig.affinity_bound_mode` (default `"clamp"`) +
  `SalienceCoordinatorConfig.affinity_squash_sigma` (default `None`, REQUIRED when
  the squash is selected). Reachable from `REEConfig` as
  `salience_affinity_bound_mode` / `salience_affinity_squash_sigma` at all three
  sites (dataclass field, `from_dims()` signature, post-`cls()` re-apply -- the
  MECH-307 `from_dims`-swallows-unknown-kwargs trap), and from `REEAgent.__init__`
  into the live `SalienceCoordinatorConfig`.

  Data flow: `affinity_weights` input signal raw value -> `bound_affinity_input(value,
  affinity_input_cap, mode, sigma)` -> per-mode weight -> affinity logit -> softmax
  over operating modes. Unchanged: `salience_weights` / `salience_aggregate` are NOT
  bounded (urgency magnitude is deliberately unbounded -- "how loud is the alarm" vs
  "which mode does this argue for").

  Operator (user decision 2026-09-19T04:18:29Z, real AskUserQuestion, Orchestrator
  decision lane `orchestrate-20260918-1840-cloud4`, answering
  `chip-20260919-salience-bounding-operator-choice`): per-signal, SIGN-PRESERVING
  SATURATING SQUASH `cap * x / (sigma + |x|)`, applied exactly where the box clamp
  was. Option A of three. NOT pooled divisive normalisation (B) and NOT
  dynamic/leaky normalisation (C).

  WHY: the box clamp maps every input above the cap to the identical output. On the
  464d/467d substrate `dacc_pe ~16-17` sat permanently at every tested cap, and
  V3-EXQ-935a measured the signature directly -- `ext_margin_mean` LINEAR in cap at
  R^2 0.9996-0.9999, i.e. the cap value and not the signal is what the softmax sees.
  That destroys the graded magnitude MECH-259's threshold test needs and is the
  discontinuous-derivative source the substrate_queue entry names for the
  `<= 1-grid-step` crossing width. Measured on the real coordinator at cap 2.0,
  sigma 1.0: clamp gives `internal_planning = 0.534446645389` for BOTH `pe=16` and
  `pe=17`; the squash gives `0.511193048516` and `0.512492618943`.

  WHY NOT B/C: the commissioned lit pull
  (`REE_assembly/evidence/literature/targeted_review_salience_gain_normalisation`,
  6 entries, 2026-09-19) licenses "saturating beats truncating" as the operator
  FAMILY (Carandini & Heeger 2012 -- nothing biology reaches for here is a
  truncation) but explicitly declines to license the POOLED divisive form at this
  site: pooling makes each mode's probability depend on the magnitudes arguing for
  the others (Louie/Khaw/Glimcher 2013), so adding MECH-261's anticipated fifth mode
  (`parallel_goal_deliberation`, SD-033e) would silently shift all four existing
  modes across the whole occupancy corpus; and Cohen 2019 identifies input-circuit
  ASYMMETRY -- which SD-032a's `affinity_weights` map has by construction -- as the
  configuration where normalisation turns pathological. C (Louie 2014 dynamic
  normalisation) was rejected because it reaches into MECH-266's two-threshold form.

  Backward compatible: `affinity_input_cap = None` is a no-op whatever the selector
  says, and with a cap set the selector's DEFAULT remains the legacy box clamp -- the
  literal pre-2026-09-19 expression `max(-cap, min(cap, value))` is still the code
  that runs. The V3-EXQ-934 cap-sweep baseline (and 935 / 935a) stay bit-identically
  reproducible, which is the substrate_queue entry's own stated requirement.

  **`affinity_squash_sigma` HAS NO DEFAULT AND MUST NOT BE GIVEN ONE HERE.** Neither
  the substrate_queue entry nor the lit pull fixes sigma or its relation to cap;
  Carandini & Heeger record that the biological semi-saturation constant is itself
  ADAPTIVE and that "a fixed saturating operator with a fixed sigma discards the
  adaptation the biology shows and may reintroduce the same range problem at a
  different operating point". Selecting the squash without an explicit positive
  sigma raises `ValueError` -- at the operator and on the first live `tick()`.
  The choice is an OPEN governance decision:
  `chip-20260919-sd032a-squash-sigma-choice` (kind `decision`). Note that sigma also
  sets the small-signal gain: `d/dx` at the origin is `cap/sigma`, so `sigma == cap`
  reproduces the clamp's unit slope in its linear region while `sigma < cap`
  AMPLIFIES small affinity signals (the Ohshiro 2011 inverse-effectiveness property
  the lit pull flags as a behavioural change the entry's title does not mention).

  Biological basis: divisive/saturating gain control as a canonical neural
  computation (Carandini & Heeger 2012); value gain control in LIP (Louie et al.
  2011). Both are mapped with explicit caveats in the lit-pull records -- they
  license the FAMILY, not this exact operator at this exact site.

  Phased training required: no (the coordinator is pure arithmetic, non-trainable,
  no gradient flow).

  MECH-094: not applicable -- the coordinator does not author replay content.

  Validation experiment: NOT queued by this landing (deliberate; the chip scoped
  this session to item (1) build only and forbade queueing). The entry's own named
  validations are `v3_exq_464d_mech266_competing_goals_behavioural` and
  `v3_exq_467d_mech266_mode_stickiness_behavioural`, both of which self-routed
  `substrate_not_ready_requeue` on the readiness gate; the successor a cap/operator
  sweep should import is `experiments/_lib/regime_occupancy_gate.py`. A validation
  run remains BLOCKED on the sigma decision above.

  Contracts: `tests/contracts/test_salience_affinity_bound_operator.py` (10 tests --
  the landing commit's message says 14, which is wrong; this file is the correct count --
  C1 bit-identical default, C2 bounded strictly inside (-cap, cap), C3 odd /
  sign-preserving, C4 strictly monotone, C5 derivative continuous across the old
  clamp boundary where the clamp's jumps 1 -> 0, C6 sigma required and undefaulted,
  C7 the at-cap degeneracy is gone, C8 `from_dims` reachability).

  Items (2) commitment-term grading (`agent.py` `_et_commit` boolean latch) and
  (3) production default remain OPEN on the substrate_queue entry. GFLAG-0311's
  title/status discrepancy on this entry is cited, not resolved -- it stays with
  `/governance`.

  See MECH-266, MECH-259, SD-032a, MECH-261; substrate_queue entry
  `mode-governance-engagement`; `failure_autopsy_V3-EXQ-935a_2026-09-16`.
