## mode-governance-engagement: affinity-input saturating squash bounding operator (2026-09-19)

- mode-governance-engagement item (1): `cingulate.salience_coordinator.affinity_bound_operator` -- IMPLEMENTED 2026-09-19.
  `ree_core/cingulate/salience_coordinator.py` (`bound_affinity_input()`,
  `AFFINITY_BOUND_CLAMP` / `AFFINITY_BOUND_SQUASH`, `SalienceCoordinator.tick()`),
  wired through `ree_core/utils/config.py` and `ree_core/agent.py`.

  Config: `SalienceCoordinatorConfig.affinity_bound_mode` (default `"clamp"`) +
  `SalienceCoordinatorConfig.affinity_squash_sigma` (default `None`, which MEANS
  `sigma = affinity_input_cap`; an explicit positive value overrides).
  Reachable from `REEConfig` as
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
  `<= 1-grid-step` crossing width. Measured on the real coordinator at cap 2.0 with
  the DEFAULT sigma (= cap): clamp gives `internal_planning = 0.534446645389` for
  BOTH `pe=16` and `pe=17`; the squash gives `0.490303890020` and `0.492647984672`.
  (The same measurement at the earlier fixture sigma 1.0 gave `0.511193048516` and
  `0.512492618943` -- recorded because the first landing quoted those.)

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

  **SIGMA RULE (user decision 2026-09-19T09:45Z, OPTION 1 of the decision chip
  `chip-20260919-sd032a-squash-sigma-choice`): `sigma` DEFAULTS TO THE CONFIGURED
  CAP.** `affinity_squash_sigma = None` means `sigma = affinity_input_cap`; an
  explicit positive value overrides it (that is the knob a calibration sweep
  varies), and a non-positive explicit value raises `ValueError`, at the operator
  and on the first live `tick()`.

  Why `sigma = cap` and not some other number: sigma sets the SMALL-SIGNAL GAIN as
  well as where saturation bites -- `d/dx` at the origin is `cap/sigma`. At
  `sigma == cap` that slope is exactly 1, so every SUB-cap signal passes through
  precisely as the legacy box clamp passes it and the only behavioural change is at
  the top end, where the clamp was degenerate. That is what lets a validation sweep
  attribute an effect to the operator change rather than to a simultaneous
  small-signal gain change. `sigma < cap` instead AMPLIFIES small affinity signals
  (the Ohshiro 2011 inverse-effectiveness property the lit pull flags as a
  behavioural change the entry's title does not mention) AND recovers LESS top-end
  separation, not more. It is also the property the queue entry's own example
  operator `cap*tanh(x/cap)` has.

  CONSIDERED AND NOT TAKEN -- option 3, an ADAPTIVE sigma tied to the signal's own
  running scale (e.g. an EMA of `|x|`), which is the form the biology actually
  shows. Carandini & Heeger record that the semi-saturation constant is itself
  adaptive and warn that "a fixed saturating operator with a fixed sigma discards
  the adaptation the biology shows and may reintroduce the same range problem at a
  different operating point". That warning is ACCEPTED here rather than answered:
  the adaptive form adds a fitted time constant to a coordinator whose occupancy
  behaviour is itself under investigation (and Louie 2014 notes REE's tick has no
  principled timescale to inherit), so **if it is ever wanted it is a NEW
  substrate_queue item, not a value change to this one.**

  Biological basis: divisive/saturating gain control as a canonical neural
  computation (Carandini & Heeger 2012); value gain control in LIP (Louie et al.
  2011). Both are mapped with explicit caveats in the lit-pull records -- they
  license the FAMILY, not this exact operator at this exact site.

  Phased training required: no (the coordinator is pure arithmetic, non-trainable,
  no gradient flow).

  MECH-094: not applicable -- the coordinator does not author replay content.

  Validation experiment: NOT queued by this landing or by the sigma landing
  (deliberate -- both were scoped to the build and forbade queueing). **With sigma
  now fixed, the run that is UNBLOCKED is the cap/operator sweep in the
  V3-EXQ-934 / 935a lineage** -- a successor to
  `experiments/v3_exq_934_mech266_cap_sweep_mode_occupancy.py` sweeping the cap with
  `salience_affinity_bound_mode="squash"` against the clamp arm, which is what can
  now show whether a bounded-but-graded input admits the mixed regime the
  bang-bang clamp could not. It must import
  `experiments/_lib/regime_occupancy_gate.py` rather than re-deriving `min()`. The
  entry's two originally-named validations,
  `v3_exq_464d_mech266_competing_goals_behavioural` and
  `v3_exq_467d_mech266_mode_stickiness_behavioural`, remain separately gated on the
  Stage-H nav-competence dependency that made them self-route
  `substrate_not_ready_requeue`, which this landing does not touch.

  Contracts: `tests/contracts/test_salience_affinity_bound_operator.py` (13 tests --
  the FIRST landing commit's message says 14 and the file then had 10; this line is
  the current count -- C1 bit-identical default, C2 bounded strictly inside
  (-cap, cap), C3 odd / sign-preserving, C4 strictly monotone, C5 derivative
  continuous across the old clamp boundary where the clamp's jumps 1 -> 0, C6 sigma
  defaults to the cap / slope 1 at the origin / explicit override / bad explicit
  sigma still raises / live-tick equivalents, C7 the at-cap degeneracy is gone,
  C8 `from_dims` reachability).

  Items (2) commitment-term grading (`agent.py` `_et_commit` boolean latch) and
  (3) production default remain OPEN on the substrate_queue entry. GFLAG-0311's
  title/status discrepancy on this entry is cited, not resolved -- it stays with
  `/governance`.

  See MECH-266, MECH-259, SD-032a, MECH-261; substrate_queue entry
  `mode-governance-engagement`; `failure_autopsy_V3-EXQ-935a_2026-09-16`.
