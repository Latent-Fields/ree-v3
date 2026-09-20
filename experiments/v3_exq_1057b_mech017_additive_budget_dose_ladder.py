#!/opt/local/bin/python3
"""
V3-EXQ-1057b -- MECH-017 reality consolidation: a DOSE LADDER on the FINAL
consolidation pass. Two letters have now circled the additive-budget construction
and established that PASS ORDER decides which way the arm is displaced. This run
stops asking whether the additive arm "works" and measures the DOSE-RESPONSE of
the last pass instead: after a FIXED pass, run k in {0, 3, 6, 12, 24} steps of the
OTHER window, and trace early and late held-out MSE against k, in BOTH orders,
with ARM_C as a plotted reference.

SLEEP DRIVER: N/A -- no SleepLoopManager is built (use_sleep_loop / sws_enabled /
              rem_enabled / use_sleep_aggregation_cluster are all left at their
              default False). The manipulation is the MECH-423 R3
              CrossModuleConsolidator pass called DIRECTLY (the same call shape
              V3-EXQ-680e validated), deliberately outside a sleep cycle -- see
              WHY NO SLEEP CYCLE below. Unchanged from V3-EXQ-1057/1057a.

RED-TEAM (Step 4.5): see THIS RUN'S OWN RED-TEAM below.

PROVENANCE -- THIS DESIGN WAS RATIFIED, NOT INVENTED HERE
----------------------------------------------------------
Source: the CONFIRMED cluster autopsy
REE_assembly/evidence/planning/failure_autopsy_V3-EXQ-1057-1057a-cluster_2026-09-20.json
(status `confirmed`, red_team CONTESTED/opus with disposition_survived true),
ratified by /governance cycle governance-20260920 (REE_assembly 108b156192).

Its fanout_recommendation (GOV-FANOUT-1, is_discrimination true) names three live
hypotheses and orders the probes:

  H-intrinsic    -- the recency cost is intrinsic to training on remote traces,
                    at any budget.
  H-reallocation -- the recency cost is what is left when a fixed budget is spent
                    away from the recent window.
  H-schedule     -- on a blocked schedule the displacement is set by pass ORDER,
                    so "cost" and "benefit" are schedule artefacts. OPEN, and
                    explicitly NOT established.

  probe 1 (this run, axis `measurement`, bears on ALL THREE):
    "FIRST: dose ladder on the final pass -- after a fixed whole-buffer pass run
     k in {0, 3, 6, 12, 24} recent-window steps (and the mirror), tracing early
     and late MSE against k, with C as a plotted reference. It can explain why the
     additive budget buys early retention only. Null: both curves flat in k."

  probe 2 (NOT queued here, OWED -- see D3 IS OWED below).

routing_detail.successor: "one run, V3-EXQ-1057b, the LAST letter: dose ladder on
the final pass first". The Step-8 gate was held interactively with the user on
2026-09-20 (AskUserQuestion) and the user selected "One last letter, 1057b
(Recommended)". THE ID IS THEREFORE 1057b BY HUMAN DECISION, not by this session's
reading of the EXQ versioning rule -- which, left to itself, would have argued for
a new number (the dose ladder asks a different measurement question than the
additive arm did). A ratified human choice outranks that heuristic; it is recorded
here so the deviation is visible rather than silent.

WHAT THE TWO PREDECESSORS ESTABLISHED (the prior, from recorded manifests)
---------------------------------------------------------------------------
V3-EXQ-1057 (whole-buffer THEN recent-window, "D1"): the additive arm failed its
own `additive_arm_retains_replay_gain` precondition 0/5. On the EARLY probes D1
landed at the NO-TRAINING control -- worse than BOTH parents on 5/5 seeds.
V3-EXQ-1057a (both orders as explicit arms): the mirror D2 (recent-window THEN
whole-buffer) KEPT replay's early gain 5/5 and was the best early arm of five, but
retained the recency benefit on only 2/5.

  recorded e1_holdout_mse_early, mean over n=5 seeds (lower is better)
    arm                                   mean        sd       source manifest
    ARM_A  (whole-buffer only)          8.02e-05   2.14e-05   1057
    ARM_D2 (recent THEN whole)          7.48e-05   1.52e-05   1057a
    ARM_C  (no extra training)          1.75e-04   5.01e-05   1057
    ARM_D1 (whole THEN recent)          1.74e-04   4.86e-05   1057

So neither order delivers both parents' benefits, and the displacement direction is
set by WHICH PASS RUNS LAST: with the recent-window pass last (D1) early fidelity
collapses onto the untrained control; with the whole-buffer pass last (D2) it is
preserved and slightly improved. The autopsy's structural_property records exactly
this and leaves the MECHANISM open -- the first draft's "fresh Adam per call"
account was REFUTED in-run (wake training is 18 such calls and accumulates).

WHAT THIS RUN ADDS, AND WHY IT IS NOT A THIRD LETTER OF THE SAME CONSTRUCTION
------------------------------------------------------------------------------
The autopsy explicitly REFUSED "a third sequential-pass letter". This run is not
one. 1057 and 1057a both asked a YES/NO question about a single fixed dose (2 x
CMC_STEPS, split 24/24 at every consolidation point) and were adjudicated on a
retention gate. This run holds the construction fixed and varies ONE scalar -- the
step count of the FINAL pass -- across five levels, and is adjudicated on the SHAPE
of the resulting curve. That is an `axis: measurement` probe, which is what the
ratified fanout ordered first.

THE MANIPULATION, EXACTLY
--------------------------
There are FOUR consolidation points in a life (2 in the early block, 2 in the late
block; CONSOLIDATE_EVERY = 3 episodes, recorded as n_consolidation_passes 4.0 in
the 1057 manifest). At the first THREE, every ladder arm runs the full, unchanged
additive construction for its order: both passes at CMC_STEPS = 24.

At the FOURTH -- the FINAL consolidation point, the last weight-moving event before
the late probes are captured -- the FIXED pass still runs at CMC_STEPS and the
DOSED pass runs at k steps, k in {0, 3, 6, 12, 24}:

  ORDER_WR ("whole_then_recent", the D1 lineage)
      non-final points : whole-buffer(24) -> recent-window(24)
      FINAL point      : whole-buffer(24) -> recent-window(k)
  ORDER_RW ("recent_then_whole", the D2 lineage, THE MIRROR)
      non-final points : recent-window(24) -> whole-buffer(24)
      FINAL point      : recent-window(24) -> whole-buffer(k)

In BOTH orders the DOSED pass is the LAST one. That is forced, not chosen: the
autopsy's sketch doses the pass that follows a "fixed" pass, and the whole point of
scoping to the final point is to titrate the last writer. k = 0 means that pass is
not run at all at the final point (`consolidate()` short-circuits at steps <= 0,
cross_module_consolidation.py:150, so k=0 is a genuine tested no-op, not a special
case -- but this driver skips the call outright and records the skip explicitly, so
the cmc_records ledger cannot be misread).

WHICH LADDER ENDPOINTS ARE ANCHORED TO ALREADY-MEASURED ARMS -- AND WHICH IS NOT
----------------------------------------------------------------------------------
This is stated explicitly because the pre-flight that authorised this run got it
half wrong, and the error is load-bearing for how the result must be read.

  ANCHORED. k = 24 reproduces the predecessor arm EXACTLY, in both orders: all four
  points run both passes at 24, which IS V3-EXQ-1057a's ARM_D1 / ARM_D2
  construction, bit-for-bit in config. So ARM_D1_K24 should reproduce 1.74e-04 and
  ARM_D2_K24 should reproduce 7.48e-05 on this machine_class, and a large departure
  is a harness-drift finding in its own right.

  NOT ANCHORED. k = 0 does NOT reproduce ARM_A. The pre-flight asserted "k=0 (no
  recent-window pass at all) is exactly ARM_A's construction (whole-buffer pass
  only)". That is FALSE under final-pass-only scoping, which the same pre-flight
  and the autopsy sketch both require: ARM_D1_K00 still runs the recent-window pass
  at the first THREE points and only omits it at the fourth. It is a new
  construction that has never been measured.

  CONSEQUENCE, stated rather than papered over: the pre-flight's non-degeneracy
  argument -- "the k=0 and k=24 endpoints already differ by ~9.4e-5 ... a real,
  resolvable gap" -- does NOT transfer to this design as written, because one of
  the two endpoints it cites is not an endpoint of this ladder. The manipulation
  here is ONE of four consolidation points, not all four, so the a-priori expected
  effect is SMALLER than that recorded gap. ARM_A and ARM_B are therefore carried
  in this run (see below) so the ladder is read against in-run anchors rather than
  against a cross-design analogy.

WHY THE FINAL POINT IS NEVERTHELESS THE RIGHT PLACE TO TITRATE
----------------------------------------------------------------
The recorded D1-vs-D2 swing is a LAST-WRITER signature: the arms are identical in
budget, window, lr and schedule and differ only in which pass runs last, and the
early-probe DV moves 2.3x between them (7.48e-05 vs 1.74e-04). If the displacement
is produced by the last pass, then titrating the last pass alone must move the
curve, and the curve's shape says HOW: a step at small k means a few steps of the
wrong window are enough to erase the other pass; a gradual ramp means the two
passes genuinely trade off. If instead the displacement accumulates across all four
points, the final-pass-only ladder comes out FLAT -- and that is an informative
null, because it falsifies the last-writer reading that both predecessors' results
suggest.

THE PREDICTION THIS DESIGN MAKES (so a null is not a shrug)
-------------------------------------------------------------
Under the last-writer reading: the ORDER_WR early curve RISES with k (toward the
untrained control, as the recent-window pass increasingly erases the whole-buffer
pass) while the ORDER_RW early curve stays roughly FLAT and low (its last pass is
the whole-buffer one, which is the pass that preserves early fidelity). That
asymmetry between the two orders is itself a discriminating observable, and it is
reported as `order_asymmetry_*` whether or not the load-bearing criterion fires.
No threshold in this script was chosen with any knowledge of the intermediate
k values; only k=24 has ever been measured, and only as a whole-life dose.

WHAT THE LADDER BUYS EACH HYPOTHESIS (the Step 2.5b design audit of the leg's null)
------------------------------------------------------------------------------------
  H-schedule     -- a non-flat curve in ONE order and a flat curve in the MIRROR is
                    the schedule signature: the dose matters only through which
                    window is written last. SUPPORTED by asymmetry.
  H-reallocation -- if the cost is budget being spent away from the recent window,
                    both curves move with k in the SAME direction (more total steps
                    of the non-recent window = more recency cost), since k adds
                    budget rather than moving it. SUPPORTED by symmetric movement.
  H-intrinsic    -- if the cost is intrinsic to touching remote traces at all, the
                    LATE curve degrades with k in ORDER_RW (where k IS whole-buffer
                    steps) and is insensitive to k in ORDER_WR (where k is
                    recent-window steps). SUPPORTED by a late-leg order asymmetry
                    opposite to the early-leg one.
  ALL THREE FLAT -- the null. It is NOT vacuous: it falsifies the last-writer
                    reading of the 1057/1057a cluster and says the displacement is
                    a property of the whole four-point schedule, not of its last
                    writer. That routes the next question to the schedule as a
                    whole (or to D3), which is a real routing decision.
  VERDICT ALIASING, checked: "curve flat because the dose does nothing" and "curve
                    flat because the DV is pinned" are separated by the
                    `dose_reaches_dv` precondition (below), which requires the k=0
                    and k=24 cells to differ AT ALL, and by `probe_target_variance`.

D3 IS OWED, NOT DONE -- AND IT IS NOT BUILDABLE AS SKETCHED
-------------------------------------------------------------
The autopsy's probe 2 (interleaved arm D3: "alternate single whole-buffer and
recent-window steps inside ONE optimisation") is NOT queued here, per the chip's
own instruction to "queue the ladder alone and report D3 as owed", and per the
autopsy's own note that the sketch is under-specified. Confirmed against source:
`CrossModuleConsolidator.consolidate()` (cross_module_consolidation.py:100-137)
takes ONE `schedule` for the whole call, and its "interleaved" mode interleaves
MODULES (e1/e2) within a trace, not DATA WINDOWS across passes (lines 185-201).
There is no per-step `window` argument, and no mixture-buffer utility exists
anywhere under ree_core/ or experiments/_lib/.

AND THE CHEAP VERSION IS WRONG -- recorded so nobody builds it by accident: a
driver-side loop calling `_consolidate(..., n_steps=1)` alternately would NOT
implement the sketch, because each call constructs a fresh per-module Adam
(cross_module_consolidation.py:155-162), so it would cold-start optimiser momentum
on every single step rather than interleave within one optimisation. D3 needs a
real substrate change (a `consolidate()` that accepts a per-step window, or a
driver-built mixture buffer) and should be routed to /implement-substrate first.

THE STOP RULE, PRE-REGISTERED (from the autopsy's fanout note)
----------------------------------------------------------------
"If the interleaved arm ALSO fails both retention gates, stop: the additive-budget
question is not answerable at this dose on this consolidator and the honest record
is V3-EXQ-1048's mixed reading." That rule is attached to the INTERLEAVED arm, not
to this ladder, and this run does not discharge it. What this run does is supply
the retention gates' DOSE-RESPONSE so that, when D3 eventually runs, "fails both
retention gates" can be read against a measured curve instead of a single point.

WHY THE RETENTION GATES ARE REPORTED HERE AND NEVER GATE (the key design call)
--------------------------------------------------------------------------------
routing_detail says "Both retention gates kept". They ARE kept -- computed, per
arm, per k, and reported as the ladder's headline curves. They are NOT whole-run
readiness preconditions, and making them so would destroy the run:

  * `additive_arm_retains_replay_gain` was measured at 0/5 on the D1 order
    (V3-EXQ-1057). ARM_D1_K24 reproduces that order exactly, so an AND'd whole-run
    gate on it is a pre-registered guaranteed failure -- the run would self-route
    substrate_not_ready_requeue before reporting a single point of the ladder.
  * At k = 0 the gates are structurally not meaningful: there is no full additive
    dose at the final point to retain anything with.
  * Retention LOSS at high k is the PHENOMENON this ladder measures. Gating on it
    would gate away the dependent variable.

This is not a judgement call invented here; it is the disposition CLAUDE.md's
Step-3 multi-arm rule mandates ("condition every precondition on the regimes it is
meaningful for, and never AND the gate whole-run"; the remedy is scoping, never
lowering a threshold), and it is the EXACT precedent V3-EXQ-1057a already set in
this same lineage, in its own combination_rule: "The corresponding gain-retention
figure for ARM_D1 is REPORTED and never gates: V3-EXQ-1057 measured that order at
0/5, so gating on it would void ARM_D2 and destroy the comparison this run exists
to make." This run applies that same established handling across the whole ladder.

WHY ARM_A, ARM_B AND ARM_C ARE CARRIED
----------------------------------------
ARM_C is required by the sketch ("with C as a plotted reference"). ARM_A
(whole-buffer only) and ARM_B (recent-window only) are carried because the
retention gates are DEFINED against them -- replay-gain is measured against B,
recency-benefit against A -- so without them "both retention gates kept" is not
computable at all. They also anchor the ladder in-run, which the k=0 endpoint
cannot do (see above), and they pin the harness: all 15 A/B/C cells of V3-EXQ-1057
came back BIT-IDENTICAL to V3-EXQ-1048's on this machine_class, so a departure here
is drift, not noise. They are NOT independent replication and are recorded
load_bearing:false, exactly as 1057a recorded them.

WHY THIS RUN IS A `diagnostic` AND ITS PREDECESSORS WERE `evidence`
--------------------------------------------------------------------
Flagged prominently because it is a DEPARTURE from the two runs this one
succeeds, and because it changes how governance weights the result.

V3-EXQ-1057 and 1057a each asked a YES/NO question whose answer would have moved
MECH-017: "does the recency deficit disappear at additive budget?" -- PASS would
have said the deficit was a reallocation artefact, FAIL that it is intrinsic to
replay. Those are claim verdicts, so `evidence` was right for them.

This run's load-bearing criterion is the SHAPE of a dose curve on a consolidation
SCHEDULE. Neither its PASS nor its FAIL makes MECH-017's "replay counters
forgetting at no cost to recency" more or less likely: a non-flat curve says the
last writer produces the displacement, a flat curve says the whole four-point
schedule does, and MECH-017 is untouched either way. Scoring a curve-shape
criterion into MECH-017's confidence or conflict ratio would corrupt it -- exactly
what this skill's claim_ids accuracy rule warns against.

This is not a free choice made to be safe. The ratified autopsy's own `debt_class`
for this node is "complex (probe-gated) / puzzle (known rules)", and CLAUDE.md's
work-graph debt vocabulary states that a `diagnostic` spike's job is to convert a
`complex (probe-gated)` node into `complicated (buildable)` backlog. That is
precisely this run's job: its result routes the next decision -- build D3 via
/implement-substrate, or accept V3-EXQ-1048's mixed reading as the record, which
is the autopsy's own stop rule. The classification follows from the ratified
artifact plus the standing vocabulary, not from this session's preference.

CONSEQUENCES, so nothing is hidden: (1) this run is excluded from governance
confidence/conflict scoring and appears in the evidence record as context only;
(2) it therefore carries the diagnostic adjudication machinery in full --
`interpretation.preconditions[]` with measured/threshold on every entry,
`criteria_non_degenerate{}`, a `load_bearing` tag, and a below-floor self-route to
`substrate_not_ready_requeue` rather than to any substrate-verdict label; (3)
every branch of the verdict grid records `evidence_direction: non_contributory`,
which is a design property and not an oversight -- the discriminating content
lives in `label` and in the machine-readable `hypothesis_verdict` block.
IF A REVIEWER DISAGREES, this is a one-line change to EXPERIMENT_PURPOSE; the
criteria, preconditions and readouts are unaffected by it.

THE PRE-REGISTERED CRITERION, AND WHERE ITS CONSTANTS CAME FROM
-----------------------------------------------------------------
NOTHING NEW WAS INVENTED. The null is the autopsy's own: "both curves flat in k".
Its negation is a disjunction, so the load-bearing criterion is:

  C1_dose_response_detected (LOAD-BEARING): at least ONE of the FOUR ladder legs
  (2 orders x {early, late} probe stratum) is NON-FLAT, where a leg is non-flat iff
  the per-seed relative endpoint contrast
        rel_delta(seed) = (mse[k=24] - mse[k=0]) / mse[k=0]
  exceeds FLAT_BAND in magnitude WITH A CONSISTENT SIGN on at least
  SIGN_CONSISTENCY_REQUIRED of the seeds.

  FLAT_BAND = LATE_TOL = 0.10 -- the lineage's existing pre-registered tolerance,
  reused unchanged (V3-EXQ-1048 C2 -> 1057 C4 -> 1057a C4 -> here). It is 12.5x this
  harness's own declared 0.8% probe-time noise band (see LIMIT 3 below), and the
  recorded cross-arm gaps it must resolve were 7-32x that band.
  SIGN_CONSISTENCY_REQUIRED = 4 of 5 -- unchanged from 1048/1057/1057a.
  The sign-consistency requirement (max of the two directional counts, not the
  count of |rel_delta| > band) is what stops symmetric seed noise reading as a
  dose response.

  It is a DISJUNCTION because the null is a CONJUNCTION -- that is forced by the
  autopsy's wording, not chosen for sensitivity. All four legs are reported
  individually (`leg_*_nonflat`, counts and signs), so a reader can see exactly
  which leg fired and is never asked to trust the OR.

  SHAPE is the scientific payload and is deliberately NOT GATED: per-leg Spearman
  rho of mse against k (per seed), the mean curve at every k, and k_half (the
  smallest k at which the mean curve has covered >= 50% of its k=0 -> k=24 change)
  are all recorded as non-load-bearing readouts. No floor is set on them because no
  prior distribution exists to set one from -- inventing a monotonicity threshold
  here would be an unpre-registered constant. This is the same disposition 1057a
  took for its late-probe skill measurement (its red-team F4): MEASURE now, let a
  successor set a floor once there is a distribution.

DV-SYMMETRY INVARIANCE (per arm, per the mandatory declaration)
-----------------------------------------------------------------
Every arm's DV is a held-out MSE -- F.mse_loss(E1.predict_long_horizon(...),
targets) on a FIXED, frozen target tensor -- i.e. a real-valued function of the
E1/E2 WEIGHTS evaluated at fixed inputs.
  - ARM_A / ARM_B: unchanged from 1057/1057a. The manipulation is WHICH TRACES the
    gradient steps are computed on; it changes the gradient direction and hence the
    weights. MSE at fixed inputs is not invariant under a change of weights.
  - ARM_C: the manipulation is the ABSENCE of those steps -- a code-path gate, not a
    value transform of the DV.
  - EVERY LADDER ARM (both orders, every k): the manipulation is the NUMBER of
    gradient steps taken by the final pass. It is emphatically NOT a broadcast
    additive constant on the DV (nothing is added to the measured MSE; steps are
    added to the training budget, and the two are related through the optimizer,
    not by arithmetic), NOT a monotone rescaling (nothing rescales the DV), and NOT
    a permutation of interchangeable units (the probe windows are a fixed ordered
    set, and the manipulation acts on the TRAINING draw, not on the probe set).
    None of the three invariance classes applies. The DV is not a rank/argmax
    statistic, so the monotone-transform class is inapplicable throughout.
  - The LADDER STATISTIC itself (rel_delta between two k levels) is a difference of
    two MSEs divided by one of them. A broadcast constant added to every arm's DV
    WOULD partially cancel in the numerator -- but no such constant exists here (the
    DV is not an aggregate over candidates), and the separate `dose_reaches_dv`
    precondition requires the two endpoint cells to differ at all.

WHAT IS MANIPULATED, EXACTLY (attribution) -- unchanged from V3-EXQ-1057a
--------------------------------------------------------------------------
All arms run the SAME live substrate function
(`CrossModuleConsolidator.consolidate(module_losses={e1: agent.compute_prediction_loss,
e2: agent.compute_e2_loss}, ...)`, ree_core/sleep/cross_module_consolidation.py),
with the SAME schedule and the SAME lr. The only things that differ across arms are
(a) which slice of agent._{self,world,action}_experience_buffer /
agent._e2_transition_buffer is visible to the loss closures for the duration of each
call -- the whole buffer or its last RECENT_WINDOW entries -- (b) the ORDER of the
two calls at a consolidation point, and (c) the STEP COUNT of the final call.

Waking training is IDENTICAL in every arm: after every training episode each arm
runs WAKE_STEPS of the same pass restricted to THAT episode's entries (the online
stream). This is what produces the recency pressure the claim's premise needs; it
is not part of the contrast.

WHY NO SLEEP CYCLE (and why that is the stronger design) -- unchanged
-----------------------------------------------------------------------
Routing the manipulation through `SleepLoopManager.force_cycle()` would make the
buffer narrowing also visible to every OTHER buffer reader inside `_run_cycle` --
`offline_integration()`'s `e1.integrate_experience`, the SWS schema pass, REM
attribution, the replay sampler -- so an arm difference would no longer be
attributable to the consolidation pass. The sleep-cycle WIRING of this exact
consolidator is separately validated by V3-EXQ-1026 (PASS, 2026-09-14). This run
isolates the consolidation CONTENT; 1026 owns the wiring.

THREE SUBSTRATE LIMITS, DECLARED UP FRONT (not worked around) -- unchanged
----------------------------------------------------------------------------
1. MECH-017's `what_would_answer` names "held-out one-step E2 world_forward error"
   as a co-primary readout. It is STRUCTURALLY UNAVAILABLE: open substrate_queue
   entry `e2-world-forward-sleep-trainer` (status pending_implementation, severity
   degrading, substrate_paths ree_core/sleep/cross_module_consolidation.py +
   phase_manager.py) records that `consolidate()` updates only E2's SELF-forward
   head and that `E2.world_forward` has no trainer anywhere in ree_core -- so a
   world_forward DV is identically 0.0 in EVERY arm by construction. This run does
   NOT attempt it. The E2 readout here is E2 SELF-forward (`predict_next_self`),
   recorded as a SECONDARY, explicitly NON-load-bearing diagnostic. The
   load-bearing verdict rests entirely on the E1 leg, which MECH-017 names as its
   other primary readout and which IS trained by this pass.
2. The open CORRUPTING entry `contextmemory-write-path-addressing-degeneracy`
   (substrate_paths ree_core/predictors/e1_deep.py::ContextMemory.write) is
   OFF-PATH here, not argued away: `ContextMemory.write()` is reached only from
   `E1Deep.update_from_observation`, gated on `E1Config.sd016_writepath_mode`,
   which this run leaves at its default "off". A precondition asserts the mode is
   "off" at runtime in every cell rather than trusting the default.
3. PROBE-TIME LSTM DROPOUT IS LIVE -- declared, measured, and deliberately NOT
   changed here. `ree_core/predictors/e1_deep.py:648` builds the transition LSTM
   with `dropout=0.1 if self.config.num_layers > 1 else 0`, and `num_layers`
   defaults to 3. Nothing in this driver or in agent.py calls `.eval()`, and
   `torch.no_grad()` does NOT disable dropout. So the held-out probe evaluation
   samples a dropout mask, and because arms consume different amounts of global RNG
   the mask differs per arm. V3-EXQ-1057a's red-team measured the resulting spread
   at 0.81% on a toy-scale trained agent (0.03% on untrained synthetic latents),
   bit-stable after `e1.eval()`. NOT FIXED HERE, on purpose: calling `.eval()` would
   change the DV's distribution relative to 1048/1057/1057a, whose recorded
   manifests this design reads as its prior and whose A/B/C cells it reproduces
   bit-identically. THIS IS THE 0.8% NOISE BAND the FLAT_BAND constant is set
   against, and it is why FLAT_BAND is 12.5x it rather than tight to it.

WHY THE ENCODER IS FROZEN (deliberate, load-bearing) -- unchanged
-------------------------------------------------------------------
Nothing in this driver trains `agent.latent_stack`: `CrossModuleConsolidator` builds
optimizers over `agent.e1.parameters()` / `agent.e2.parameters()` only. That is the
design, not an SD-070 omission (`_train_all_on_agent` / `zworld_p0_episodes` are not
used here at all): a trained encoder would make z_self/z_world ARM-DEPENDENT, and
the design requires a FIXED probe set shared by all arms. With the encoder frozen
and the action stream model-independent, every arm sees bit-identical latents --
asserted, not assumed, by the `probe_set_identical_across_arms` precondition, which
hashes each cell's probe tensors and requires all arms of a seed to agree exactly.
`alpha_world` / `alpha_self` are raised to 0.9 (from the 0.3 default) so z_world
tracks the observation instead of being heavily EMA-smoothed.

This IS the mandatory phased-training discipline in its strictest form, not an
exemption: P0 = the WARMUP_EPISODES encoder-only rollout (no downstream loss);
P1 = every E1/E2 gradient step in this run, taken over buffer entries stored as
`latent_state.z_*.detach().clone()` against a permanently frozen encoder -- so the
heads never chase a moving latent target (the EXQ-166b/c/d, EXQ-085l, EXQ-194
failure mode); P2 = the held-out probe evaluation, under `torch.no_grad()`.

MODEL-INDEPENDENT BEHAVIOUR (commitment-free by construction) -- unchanged
----------------------------------------------------------------------------
Actions are drawn from a per-cell `torch.Generator(seed)`, never from
`agent.select_action`. This makes the experience stream identical across arms (so
the arms cannot diverge behaviourally and the probe set really is fixed), and keeps
the run entirely clear of the E3/F-dominance committed-selection layer -- there is
no action-commitment DV here, so the known conversion ceiling can neither
manufacture nor mask this result.

NON-DEGENERACY PRECONDITIONS
------------------------------
The claim's four, unchanged in form from 1057a, plus the instrument gates, plus two
new ones specific to the ladder. ALL are whole-run gates EXCEPT where noted, and
none of them is a retention gate (see above):
 (i)   `forgetting_present_in_control` -- ARM_C's WITHIN-STRATUM early-probe
       degradation must clear FORGETTING_FLOOR. Read off an arm no criterion uses.
 (ii)  `replay_covers_early_regime` -- ARM_A's early-regime buffer share at the
       LATE-block points must clear REPLAY_EARLY_SHARE_FLOOR.
 (iii) `consolidation_updated_e1` / `consolidation_updated_e2` -- the pass must
       actually move the modules the DV reads.
 (iv)  `early_late_state_divergence` -- Jaccard distance between early- and
       late-regime occupied grid cells, floored.
Instrument gates: `replay_window_separation`, `manipulation_reaches_dv`,
`e1_has_skill_over_persistence_compared_arms`, `gradient_budget_matched`,
`probe_set_identical_across_arms`, `probe_target_variance`,
`sd016_writepath_mode_off` -- all unchanged from 1057a.
NEW, and specific to this design:
 (v)   `ladder_budget_is_pre_registered_dose` -- every ladder cell's realized
       CONSOLIDATION E1 GRADIENT STEPS must equal (n_points-1)*2*CMC_STEPS +
       CMC_STEPS + k exactly (worst absolute error over all ladder cells, <= 0.5).
       If the dose is not what was pre-registered, the abscissa of the entire
       ladder is wrong, which is an INSTRUMENT failure and not evidence about
       MECH-017. Counterpart of 1057a's `additive_budget_is_a_plus_b`. NOTE THE
       UNITS: it reads `updates_e1`, not `n_updates` -- the latter counts one
       update per MODULE per step (e1 and e2) and is therefore 2x the step count.
       The first smoke of this script failed on exactly that confusion.
 (vi)  `dose_reaches_dv` -- in BOTH orders and on EVERY seed, the k=0 and k=24
       cells must produce DIFFERENT early-probe E1 MSE. A bit-identical pair is a
       dead manipulation, and would otherwise be scored as a clean "flat in k"
       null on no manipulation at all. Same form and same threshold as
       `manipulation_reaches_dv`, re-pointed at the ladder endpoints; takes the
       WORSE of the two orders.
Any unmet precondition routes the whole run to `substrate_not_ready_requeue` with
`non_degenerate: false` -- never to a MECH-017 verdict.

THIS RUN'S OWN RED-TEAM (Step 4.5)
------------------------------------
RED_TEAM_PLACEHOLDER

Run with:
  /opt/local/bin/python3 experiments/v3_exq_1057b_mech017_additive_budget_dose_ladder.py --dry-run
  /opt/local/bin/python3 experiments/v3_exq_1057b_mech017_additive_budget_dose_ladder.py
"""
from __future__ import annotations

import argparse
import hashlib
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.sleep.cross_module_consolidation import (
    CrossModuleConsolidator,
    CrossModuleConsolidatorConfig,
)
from ree_core.utils.config import REEConfig
from experiment_protocol import emit_outcome
from experiments._lib.arm_fingerprint import arm_cell
from experiments._metrics import check_degeneracy, p0_readiness_gate, P0NotReady
from experiments.pack_writer import write_flat_manifest


EXPERIMENT_TYPE = "v3_exq_1057b_mech017_additive_budget_dose_ladder"
QUEUE_ID = "V3-EXQ-1057b"
CLAIM_IDS: List[str] = ["MECH-017"]
# DIAGNOSTIC, deliberately and UNLIKE its predecessors (V3-EXQ-1057/1057a were
# both `evidence`). See WHY THIS RUN IS A DIAGNOSTIC in the docstring. One line
# to reverse if a reviewer disagrees.
EXPERIMENT_PURPOSE = "diagnostic"
SUPERSEDES = "V3-EXQ-1057a"

# --- reference arms (carried; see WHY ARM_A, ARM_B AND ARM_C ARE CARRIED) ----
ARM_A = "ARM_A_OFFLINE_REPLAY"           # whole-buffer pass only
ARM_B = "ARM_B_BUDGET_MATCHED_RECENT"    # recent-window pass only
ARM_C = "ARM_C_NO_EXTRA_TRAINING"        # the plotted reference the sketch asks for
REFERENCE_ARMS = (ARM_A, ARM_B, ARM_C)

# --- the ladder ---------------------------------------------------------------
# Two orders. In BOTH, the DOSED pass is the LAST one at the final consolidation
# point; the FIXED pass always runs at CMC_STEPS. See THE MANIPULATION, EXACTLY.
ORDER_WR = "whole_then_recent"   # V3-EXQ-1057's D1 lineage; dosed pass = recent-window
ORDER_RW = "recent_then_whole"   # V3-EXQ-1057a's D2 lineage; dosed pass = whole-buffer
ORDERS = (ORDER_WR, ORDER_RW)
ORDER_TAG = {ORDER_WR: "D1", ORDER_RW: "D2"}

# The ratified ladder, verbatim from the autopsy's fanout_recommendation sketch.
K_LADDER = (0, 3, 6, 12, 24)


_LADDER_ARM_RE = re.compile(r"^ARM_(D1|D2)_K(\d+)$")


def _ladder_arm(order: str, k: int) -> str:
    """Canonical arm name for a ladder point, e.g. ARM_D1_K06."""
    return f"ARM_{ORDER_TAG[order]}_K{int(k):02d}"


def _parse_ladder_arm(arm: str) -> Optional[Tuple[str, int]]:
    """(order, k) for a ladder arm, or None for a reference arm.

    Parsed STRUCTURALLY from the name, never by enumerating K_LADDER. The
    enumerating form silently failed for any ladder whose k values are not the
    module-level constant -- which is exactly the --dry-run ladder -- so every
    dry-run ladder arm fell through to the reference-arm branch and the smoke
    died on a window_label KeyError. Caught by the first smoke of this script.
    """
    m = _LADDER_ARM_RE.match(arm)
    if not m:
        return None
    tag, k = m.group(1), int(m.group(2))
    for order in ORDERS:
        if ORDER_TAG[order] == tag:
            return order, k
    return None


LADDER_ARMS = tuple(_ladder_arm(o, k) for o in ORDERS for k in K_LADDER)
ARMS = REFERENCE_ARMS + LADDER_ARMS

# Seed 44 is deliberately absent (recurring per-seed instability on reef-config
# envs, EXQ-539/540, V3-EXQ-538a); 45 is the sanctioned substitute. Unchanged
# from V3-EXQ-1048/1057/1057a, which is what makes the A/B/C cells comparable.
SEEDS = (42, 123, 456, 2026, 45)

# ---------------------------------------------------------------------------
# Life geometry. Total buffered steps = TRAIN_EPISODES * EPISODE_STEPS = 420,
# comfortably under the 1000-entry FIFO cap on _world_experience_buffer /
# _e2_transition_buffer (agent.py) -- so the WHOLE life stays resident and
# precondition (ii) is structurally satisfiable rather than capped away.
# ---------------------------------------------------------------------------
GRID_SIZE = 9
N_HAZARDS = 1
N_RESOURCES = 1
SELF_DIM = 16
WORLD_DIM = 16
ALPHA = 0.9  # alpha_world / alpha_self; see docstring (0.3 default degenerates)

EPISODE_STEPS = 30
WARMUP_EPISODES = 6      # early regime, trains, no consolidation point
EARLY_EPISODES = 6       # early regime, trains, consolidation at 3 and 6
LATE_EPISODES = 6        # late  regime, trains, consolidation at 3 and 6
TRAIN_EPISODES = WARMUP_EPISODES + EARLY_EPISODES + LATE_EPISODES   # = 18 (the ep N/M denominator)
# Warmup and per-episode waking budget were raised (2 -> 6 episodes, 16 -> 24 steps)
# after a 3-seed full-scale measurement showed E1 sitting BELOW a do-nothing
# persistence predictor on some seeds -- an under-trained world model, which makes a
# "fidelity" reading hard to interpret even though the A-vs-B ordering was already
# 3/3 consistent. The lever is training strength, not a relaxed gate.
# Life length stays inside the 1000-entry buffer FIFO cap: 18 * 30 = 540.
PROBE_EPISODES = 4       # per stratum; HELD OUT (buffers truncated after capture)

CONSOLIDATE_EVERY = 3    # episodes, within the early and late blocks
RECENT_WINDOW = CONSOLIDATE_EVERY * EPISODE_STEPS   # = 90: arm B's visible slice
# NOTE: the LIVE window is always recomputed from the episode length actually in
# use (`_recent_window()`), never read off this constant. Holding it fixed while
# --dry-run shrinks the episode makes the window exceed the whole buffer, and arm
# B silently collapses onto arm A -- which is exactly what the first smoke of this
# script did (A and B bit-identical). The `replay_window_separation` and
# `manipulation_reaches_dv` preconditions below now fail that state outright.

CMC_STEPS = 24           # consolidation steps per point (A and B; C gets 0)
WAKE_STEPS = 24          # per-episode online steps (ALL arms, identical)
# Two learning rates, deliberately different and for different jobs.
# WAKE_LR drives the recency pressure the claim's premise needs (forgetting of
# early states must actually happen, or nothing is there for replay to counter).
# CMC_LR is the OFFLINE consolidation step, deliberately gentler: at 1e-3 the
# consolidation pass overshot so hard that BOTH arms A and B ended up WORSE than a
# do-nothing persistence predictor on the early probes (measured -0.36 to -0.77
# skill across three independent runs), which is the state the red-team's Finding 2
# calls uninterpretable. A and B share this value exactly -- it is not part of the
# contrast, and the gradient-budget match is enforced separately.
WAKE_LR = 1e-3
CMC_LR = 2e-4
CMC_BATCH = 16

# Two spatially separated regimes -> early and late states genuinely differ.
EARLY_START = (2, 2)
LATE_START = (6, 6)
HAZARDS = [(4, 4)]
RESOURCES = [(4, 2)]
# action indices: 0 up, 1 down, 2 left, 3 right (CausalGridWorldV2.ACTIONS)
EARLY_ACTION_P = (0.35, 0.15, 0.35, 0.15)   # up/left biased
LATE_ACTION_P = (0.15, 0.35, 0.15, 0.35)    # down/right biased

# ---------------------------------------------------------------------------
# PRE-REGISTERED thresholds (constants; never derived from this run's own stats)
# ---------------------------------------------------------------------------
SIGN_CONSISTENCY_REQUIRED = 4        # of len(SEEDS) = 5
LATE_TOL = 0.10                      # A may be up to 10% worse than B on LATE probes
MIN_REL_GAIN_EARLY = 0.05            # mean (B-A)/B on EARLY probes must clear 5%
FORGETTING_FLOOR = 1.10              # precondition (i): C_early / C_late
REPLAY_EARLY_SHARE_FLOOR = 0.20      # precondition (ii)
UPDATES_FLOOR = 1.0                  # precondition (iii)
STATE_DIVERGENCE_FLOOR = 0.20        # precondition (iv), Jaccard distance
PROBE_TARGET_VAR_FLOOR = 1e-4        # instrument gate: probe targets must move

MIN_WINDOW_SEPARATION = 0.20         # A's early-regime share minus B's
PERSISTENCE_SKILL_FLOOR = 0.0        # E1 must beat a do-nothing forward model

PROBE_WINDOW_STRIDE = 2   # denser windows -> lower DV variance
EPS = 1e-12

# `replay_window_separation`'s 0.20 floor is guaranteed by ARITHMETIC at design
# time, not hoped for -- which is what this exemption is for. At every LATE-block
# consolidation point arm B's visible slice is the last CONSOLIDATE_EVERY (3)
# episodes, all of which are late-regime by construction, so B's early-regime
# share is exactly 0.000. Arm A sees the whole buffer, whose early-regime prefix is
# (WARMUP_EPISODES + EARLY_EPISODES) * EPISODE_STEPS = 8 * 30 = 240 steps out of at
# most TRAIN_EPISODES * EPISODE_STEPS = 420, i.e. a share of at least 240/420 =
# 0.571 (and 240/330 = 0.727 at the first late point). Minimum separation is
# therefore >= 0.571, ~2.9x the floor, before any learning happens. The separate
# `manipulation_reaches_dv` precondition supplies the DV-side headroom check the
# same warning asks for: it fails the run when A and B produce a bit-identical DV.
CRITERION_ACHIEVABLE_RANGE_EXEMPT = (
    "replay_window_separation's floor is established by construction: arm B's "
    "late-block window is all-late-regime (share 0.000) and arm A's whole-buffer "
    "share is >= 240/420 = 0.571, so separation >= 0.571 vs a 0.20 floor. DV-side "
    "headroom is covered by the manipulation_reaches_dv precondition."
)


def _recent_window(episode_steps: int) -> int:
    """Arm B's visible slice, in buffered steps, for the episode length in use."""
    return CONSOLIDATE_EVERY * int(episode_steps)

# --- NEW, ladder-specific pre-registered constant ---------------------------
# The "flat in k" band. NOT a new constant: it is LATE_TOL, the lineage's own
# pre-registered tolerance, reused unchanged (V3-EXQ-1048 C2 -> 1057 C4 -> 1057a
# C4 -> here). Bound to the same name so the reuse is explicit in the code rather
# than only in the docstring, and so no second numeric literal can drift from it.
# 12.5x this harness's declared 0.8% probe-time dropout noise band (LIMIT 3).
FLAT_BAND = LATE_TOL

# Total extra gradient steps a ladder cell must spend, by construction:
# (N_POINTS - 1) points at 2 x CMC_STEPS, then the FINAL point at CMC_STEPS + k.
# Asserted per cell by the `ladder_budget_is_pre_registered_dose` precondition.
# UNITS, and the trap the first smoke caught: `total_extra_gradient_steps` sums
# CrossModuleConsolidator's `n_updates`, which counts one update PER MODULE per
# step (e1 AND e2), so it is 2x the step count. The pre-registered dose is a STEP
# count, so this is compared against `updates_e1` -- the pass's own per-step E1
# counter, which equals the step count exactly (verified: n_steps=k -> updates_e1
# =k) and is the module the load-bearing DV reads. Comparing against
# total_extra_gradient_steps instead reported 56 vs an expected 28.
def _expected_ladder_budget(n_points: int, cmc_steps: int, k: int) -> float:
    """Pre-registered CONSOLIDATION E1 GRADIENT STEPS for a ladder cell."""
    return float((n_points - 1) * 2 * cmc_steps + cmc_steps + k)



# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _to_batched(x, device) -> torch.Tensor:
    if not torch.is_tensor(x):
        x = torch.tensor(x, dtype=torch.float32, device=device)
    else:
        x = x.to(device)
    if x.dim() == 1:
        x = x.unsqueeze(0)
    return x


def _make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        seed=seed,
        size=GRID_SIZE,
        num_hazards=N_HAZARDS,
        num_resources=N_RESOURCES,
        use_proxy_fields=True,
    )


def _make_agent(env: CausalGridWorldV2) -> REEAgent:
    """IDENTICAL config in every arm -- the arms differ only in the driver's
    consolidation call, never in agent construction."""
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=SELF_DIM,
        world_dim=WORLD_DIM,
        alpha_world=ALPHA,
        alpha_self=ALPHA,
    )
    return REEAgent(cfg)


def _mean(xs: Sequence[float]) -> float:
    xs = list(xs)
    return float(sum(xs) / len(xs)) if xs else float("nan")


def _tensor_digest(chunks: Sequence[torch.Tensor]) -> str:
    h = hashlib.sha256()
    for t in chunks:
        h.update(t.detach().to(torch.float32).cpu().numpy().tobytes())
    return h.hexdigest()[:16]


class _BufferWindow:
    """Temporarily narrow the four replay buffers the consolidation losses read.

    Restores the ORIGINAL list objects on exit (the slices are new lists holding
    the same tensor references, so nothing is copied or lost). ``window=None``
    leaves everything untouched -- that is arm A's whole-life case, and it goes
    through this same object so A and B differ by the slice bound alone.
    """

    _NAMES = (
        "_self_experience_buffer",
        "_world_experience_buffer",
        "_action_experience_buffer",
        "_e2_transition_buffer",
    )

    def __init__(self, agent: REEAgent, window: Optional[int]):
        self.agent = agent
        self.window = window
        self._saved: Dict[str, list] = {}

    def __enter__(self) -> "_BufferWindow":
        if self.window is None:
            return self
        for name in self._NAMES:
            buf = getattr(self.agent, name)
            self._saved[name] = buf
            setattr(self.agent, name, buf[-self.window:])
        return self

    def __exit__(self, *exc) -> bool:
        for name, buf in self._saved.items():
            setattr(self.agent, name, buf)
        self._saved.clear()
        return False


def _consolidate(
    agent: REEAgent,
    consolidator: CrossModuleConsolidator,
    n_steps: int,
    window: Optional[int],
    lr: float,
) -> Dict[str, float]:
    """One live MECH-423 R3 consolidation pass over the windowed buffers."""
    with _BufferWindow(agent, window):
        return consolidator.consolidate(
            module_losses={
                "e1": lambda: agent.compute_prediction_loss(),
                "e2": lambda: agent.compute_e2_loss(batch_size=CMC_BATCH),
            },
            module_params={
                "e1": list(agent.e1.parameters()),
                "e2": list(agent.e2.parameters()),
            },
            n_steps=n_steps,
            schedule="interleaved",
            lr=lr,
            simulation_mode=False,
        )


def _early_share(regime_flags: List[int], window: Optional[int]) -> float:
    """Share of the buffer slice visible to the pass that is EARLY-regime.

    ``regime_flags`` is one entry per buffered step, 1 for the early regime
    (warmup + early block), 0 for the late block -- kept in lockstep with
    ``_world_experience_buffer`` by the training loop.
    """
    view = regime_flags if window is None else regime_flags[-window:]
    if not view:
        return 0.0
    return float(sum(view)) / float(len(view))


# ---------------------------------------------------------------------------
# probe capture + held-out readouts
# ---------------------------------------------------------------------------
def _e1_probe_mse(agent: REEAgent, episodes: List[Dict]) -> Tuple[float, float, float]:
    """Held-out MULTI-STEP E1 error, mirroring compute_prediction_loss exactly.

    Returns (mean MSE over windows, variance of the concatenated targets, mean
    PERSISTENCE-baseline MSE over the same windows). The persistence baseline is
    the do-nothing forward model -- predict the window's initial state for every
    horizon step -- and exists so a below-baseline E1 is caught as an instrument
    failure rather than read as a fidelity result.
    """
    horizon = int(agent.e1.config.prediction_horizon)
    action_cond = bool(getattr(agent.e1.config, "action_conditioned_transition", False))
    saved_hidden = agent.e1._hidden_state
    losses: List[float] = []
    persistence: List[float] = []
    target_chunks: List[torch.Tensor] = []
    with torch.no_grad():
        for ep in episodes:
            combined = ep["combined"]              # [1, T, total_dim]
            actions = ep["actions"]                # [1, T, action_dim] or None
            total = int(combined.shape[1])
            start = 0
            while start + 2 <= total:
                end = min(start + horizon + 1, total)
                if end - start < 2:
                    break
                seq = combined[:, start:end, :]
                agent.e1.reset_hidden_state()
                initial = seq[:, 0, :]
                horizon_len = int(seq.shape[1]) - 1
                acts = None
                if action_cond and actions is not None:
                    # Same +1 offset compute_prediction_loss uses: the action that
                    # carries state_i -> state_{i+1} is recorded alongside state_{i+1}.
                    acts = actions[:, start + 1:end, :]
                    if int(acts.shape[1]) != horizon_len:
                        acts = None
                preds = agent.e1.predict_long_horizon(
                    initial, horizon=horizon_len, actions=acts
                )
                targets = seq[:, 1:, :]
                losses.append(
                    float(F.mse_loss(preds[:, :targets.shape[1], :], targets).item())
                )
                persistence.append(float(F.mse_loss(
                    initial.unsqueeze(1).expand_as(targets), targets
                ).item()))
                target_chunks.append(targets.reshape(-1))
                if end >= total:
                    break
                start += PROBE_WINDOW_STRIDE
    agent.e1._hidden_state = saved_hidden
    if not losses:
        return float("nan"), 0.0, float("nan")
    var = float(torch.cat(target_chunks).var(unbiased=False).item()) if target_chunks else 0.0
    return _mean(losses), var, _mean(persistence)


def _e2_probe_mse(agent: REEAgent, episodes: List[Dict]) -> float:
    """Held-out ONE-STEP E2 SELF-forward error (predict_next_self).

    NOT `E2.world_forward` -- that head has no trainer anywhere in ree_core
    (substrate_queue `e2-world-forward-sleep-trainer`), so a world_forward DV
    would be identically 0.0 in every arm. Secondary / non-load-bearing.
    """
    zs, acts, zs1 = [], [], []
    for ep in episodes:
        for z_t, a_t, z_t1 in ep["transitions"]:
            zs.append(z_t)
            acts.append(a_t)
            zs1.append(z_t1)
    if not zs:
        return float("nan")
    with torch.no_grad():
        pred = agent.e2.predict_next_self(torch.cat(zs, dim=0), torch.cat(acts, dim=0))
        return float(F.mse_loss(pred, torch.cat(zs1, dim=0)).item())



# ---------------------------------------------------------------------------
# one (arm, seed) cell
# ---------------------------------------------------------------------------
def run_cell(
    arm: str,
    seed: int,
    warmup_eps: int,
    early_eps: int,
    late_eps: int,
    probe_eps: int,
    episode_steps: int,
    cmc_steps: int,
) -> Dict:
    print(f"Seed {seed} Condition {arm}", flush=True)

    ladder = _parse_ladder_arm(arm)
    order = ladder[0] if ladder else None
    # k_final is the FINAL point's dosed-pass step count. For the reference arms
    # it is unused (they run one pass, or none, at every point).
    k_final = ladder[1] if ladder else None

    env = _make_env(seed)
    agent = _make_agent(env)
    device = agent.device
    consolidator = CrossModuleConsolidator(
        CrossModuleConsolidatorConfig(schedule="interleaved", n_steps=cmc_steps, lr=CMC_LR)
        # lr is overridden per call by _consolidate(); this default is never used.
    )
    # Model-INDEPENDENT action stream: identical in every arm of this seed.
    rng = torch.Generator(device="cpu").manual_seed(seed)

    recent_window = _recent_window(episode_steps)
    train_total = warmup_eps + early_eps + late_eps
    regime_flags: List[int] = []          # 1 per buffered step: 1 = early regime
    occupancy = {"early": set(), "late": set()}
    cmc_records: List[Dict] = []
    ep_index = 0
    def _run_episode(regime: str, capture: bool) -> Optional[Dict]:
        """One 30-step episode. ``capture=True`` holds the episode OUT of the
        training buffers (they are truncated back afterwards) and returns its
        probe tensors instead."""
        nonlocal ep_index
        pre_lens = {
            name: len(getattr(agent, name))
            for name in _BufferWindow._NAMES
        }
        pre_flags = len(regime_flags)

        start_pos = EARLY_START if regime == "early" else LATE_START
        probs = torch.tensor(
            EARLY_ACTION_P if regime == "early" else LATE_ACTION_P, dtype=torch.float32
        )
        _, obs_dict = env.reset_to(start_pos, HAZARDS, RESOURCES)
        agent.e1.reset_hidden_state()

        prev_action: Optional[torch.Tensor] = None
        combined_rows: List[torch.Tensor] = []
        action_rows: List[torch.Tensor] = []
        transitions: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

        for _step in range(episode_steps):
            obs_body = _to_batched(obs_dict["body_state"], device)
            obs_world = _to_batched(obs_dict["world_state"], device)
            obs_harm = obs_dict.get("harm_obs", None)
            if obs_harm is not None:
                obs_harm = _to_batched(obs_harm, device)

            prev_latent = agent._current_latent
            prev_z_self = (
                prev_latent.z_self.detach().clone() if prev_latent is not None else None
            )
            # The action that CARRIED z_{t-1} -> z_t is the one sampled on the
            # PREVIOUS step, not the one about to be sampled now. Red-team minor
            # note (a): several landed harnesses key E2 on `act_prev`
            # (_lib/allon_training.py:395); keying on the not-yet-executed action
            # trains E2 on a mapping that never happened. Probe and training use
            # this same corrected convention, so the held-out E2 readout stays a
            # fair measure of what E2 was trained to do.
            caused_by = prev_action

            latent = agent.sense(obs_body, obs_world, obs_harm=obs_harm)
            ticks = agent.clock.advance()
            if ticks.get("e1_tick", False):
                agent._e1_tick(latent)
            if not capture:
                regime_flags.append(1 if regime == "early" else 0)

            occupancy[regime].add(tuple(env.get_agent_position()))

            action_idx = int(torch.multinomial(probs, 1, generator=rng).item())
            action = torch.zeros(1, env.action_dim, device=device)
            action[0, action_idx] = 1.0

            if capture:
                # The probe transition triple uses the SAME (prev_z_self, action,
                # z_self) convention as the record_transition() call below, i.e.
                # exactly what compute_e2_loss trains on. Held-out probe and
                # training data must share the convention or the E2 readout is
                # not a fair held-out measure of what E2 was trained to do.
                combined_rows.append(
                    torch.cat(
                        [latent.z_self.detach().squeeze(0), latent.z_world.detach().squeeze(0)]
                    )
                )
                action_rows.append(agent._e1_action_one_hot().detach().reshape(1, -1))
                if prev_z_self is not None and caused_by is not None:
                    transitions.append(
                        (prev_z_self, caused_by.clone(), latent.z_self.detach().clone())
                    )
            if prev_z_self is not None and caused_by is not None:
                agent.record_transition(prev_z_self, caused_by, latent.z_self.detach())
            prev_action = action

            _, harm_signal, done, _info, obs_dict = env.step(action)
            agent.update_residue(float(harm_signal) if float(harm_signal) < 0 else 0.0)
            if done:
                _, obs_dict = env.reset_to(start_pos, HAZARDS, RESOURCES)
                agent.e1.reset_hidden_state()

        if capture:
            # HOLD OUT: truncate every buffer back to its pre-episode length so
            # nothing from a probe episode can be trained on or replayed.
            for name, n in pre_lens.items():
                del getattr(agent, name)[n:]
            del regime_flags[pre_flags:]
            acts_t = torch.stack([a.reshape(-1) for a in action_rows]).unsqueeze(0).to(device)
            return {
                "combined": torch.stack(combined_rows).unsqueeze(0),
                "actions": acts_t,
                "transitions": transitions,
            }

        ep_index += 1
        print(f"  [train] {arm} seed={seed} ep {ep_index}/{train_total}", flush=True)
        # Waking training -- IDENTICAL in all arms: budgeted steps over THIS
        # episode's entries only (the online stream, no replay).
        _consolidate(agent, consolidator, WAKE_STEPS, window=episode_steps, lr=WAKE_LR)
        return None


    def _one_pass(block: str, label: str, window: Optional[int],
                  n_steps: int, is_final: bool, dosed: bool) -> None:
        """One consolidation pass, recorded.

        `window=None` is the whole-life buffer. `n_steps` is explicit rather than
        read off the outer `cmc_steps`: that single shared variable is exactly what
        V3-EXQ-1057/1057a used for BOTH passes, and threading it per pass is the
        one structural change the ladder needs.
        """
        share = _early_share(regime_flags, window)
        metrics = _consolidate(agent, consolidator, n_steps, window, lr=CMC_LR)
        cmc_records.append({
            "block": block,
            "pass": label,
            "skipped": False,
            "is_final_point": bool(is_final),
            "is_dosed_pass": bool(dosed),
            "requested_steps": float(n_steps),
            "early_regime_share": share,
            "n_updates": float(metrics.get("n_updates", 0.0)),
            "updates_e1": float(metrics.get("updates_e1", 0.0)),
            "updates_e2": float(metrics.get("updates_e2", 0.0)),
            "cross_module_replay_share": float(metrics.get("cross_module_replay_share", 0.0)),
        })

    def _consolidation_point(block: str, is_final: bool) -> None:
        """The manipulation.

        ARM_C skips consolidation entirely. ARM_A and ARM_B run their single pass
        at CMC_STEPS at every point, exactly as in V3-EXQ-1048/1057/1057a --
        `is_final` does not touch them, which is what keeps them bit-comparable
        with the recorded manifests.

        A LADDER arm runs BOTH passes at every point, in its order, as two
        INDEPENDENT sequential `consolidator.consolidate(...)` calls -- never one
        accumulated backward. `_consolidate` wraps each call in its own
        `_BufferWindow`, so each pass has fully exited the context manager (buffers
        restored) before the next begins; neither pass can inherit the other's
        narrowing, in either order.

        THE DOSE. At the first (N_POINTS - 1) points both passes run at CMC_STEPS,
        i.e. the full unchanged additive construction for that order. At the FINAL
        point the FIXED pass still runs at CMC_STEPS and the LAST (dosed) pass runs
        at `k_final` steps. k_final == 0 means the dosed pass is not run at all;
        the skip is RECORDED rather than silently omitted, so the cmc_records
        ledger and the budget precondition can both see it.
        """
        if arm == ARM_C:
            cmc_records.append({"block": block, "pass": "none", "skipped": True,
                                "is_final_point": bool(is_final),
                                "is_dosed_pass": False,
                                "requested_steps": 0.0, "n_updates": 0.0})
            return
        if order is None:
            # ARM_A / ARM_B -- one pass, full budget, every point. Unchanged.
            _one_pass(block,
                      "whole_buffer" if arm == ARM_A else "recent_window",
                      None if arm == ARM_A else recent_window,
                      cmc_steps, is_final, dosed=False)
            return

        # --- ladder arms --------------------------------------------------
        if order == ORDER_WR:
            fixed_label, fixed_window = "whole_buffer", None
            dosed_label, dosed_window = "recent_window", recent_window
        else:
            fixed_label, fixed_window = "recent_window", recent_window
            dosed_label, dosed_window = "whole_buffer", None

        _one_pass(block, fixed_label, fixed_window, cmc_steps, is_final, dosed=False)

        dose = int(k_final) if is_final else int(cmc_steps)
        if dose > 0:
            _one_pass(block, dosed_label, dosed_window, dose, is_final, dosed=True)
        else:
            # k = 0 at the final point: the dosed pass is not run. Recorded
            # explicitly so "no entry" can never be confused with "not reached".
            cmc_records.append({
                "block": block, "pass": dosed_label + "_dose_zero", "skipped": True,
                "is_final_point": bool(is_final), "is_dosed_pass": True,
                "requested_steps": 0.0, "n_updates": 0.0,
            })

    # ---- how many consolidation points this life has ----------------------
    # Needed by the budget precondition; computed from the same arithmetic the
    # loops below use, never hardcoded (a --dry-run life has fewer points).
    n_points = (early_eps // CONSOLIDATE_EVERY) + (late_eps // CONSOLIDATE_EVERY)
    late_points_total = late_eps // CONSOLIDATE_EVERY
    late_point_index = 0

    # ---- P0: warmup (early regime) ----------------------------------------
    for _ in range(warmup_eps):
        _run_episode("early", capture=False)

    # ---- EARLY block ------------------------------------------------------
    for i in range(early_eps):
        _run_episode("early", capture=False)
        if (i + 1) % CONSOLIDATE_EVERY == 0:
            _consolidation_point("early", is_final=False)

    # ---- EARLY probe capture (held out) -----------------------------------
    early_probes = [_run_episode("early", capture=True) for _ in range(probe_eps)]
    # Held-out EARLY error measured NOW, while the early states are still the
    # recent ones. Re-measured at the end of the life; the ratio of the two is a
    # WITHIN-STRATUM forgetting measure, which an early-vs-late comparison is not.
    e1_early_at_capture, _, _ = _e1_probe_mse(agent, early_probes)

    # ---- LATE block -------------------------------------------------------
    for i in range(late_eps):
        _run_episode("late", capture=False)
        if (i + 1) % CONSOLIDATE_EVERY == 0:
            late_point_index += 1
            # THE FINAL CONSOLIDATION POINT: the last weight-moving event before
            # the late probes are captured. This is the point the ladder doses.
            _consolidation_point("late", is_final=(late_point_index == late_points_total))

    # ---- LATE probe capture (held out) ------------------------------------
    late_probes = [_run_episode("late", capture=True) for _ in range(probe_eps)]

    # ---- P2: held-out readouts -------------------------------------------
    e1_early, var_early, pers_early = _e1_probe_mse(agent, early_probes)
    e1_late, var_late, pers_late = _e1_probe_mse(agent, late_probes)
    skill_early = 1.0 - (e1_early / max(pers_early, EPS))
    skill_late = 1.0 - (e1_late / max(pers_late, EPS))
    forgetting_ratio = e1_early / max(e1_early_at_capture, EPS)
    e2_early = _e2_probe_mse(agent, early_probes)
    e2_late = _e2_probe_mse(agent, late_probes)

    late_points = [r for r in cmc_records if r["block"] == "late" and not r["skipped"]]
    replay_early_share = (
        _mean([r["early_regime_share"] for r in late_points]) if late_points else 0.0
    )
    # A ladder arm runs TWO passes per point, so the cell-level mean above blends
    # them. Report the WHOLE-BUFFER pass separately -- that is the replay
    # component, and it is what makes ladder coverage comparable with ARM_A's.
    # Descriptive only: no precondition or criterion reads it
    # (replay_covers_early_regime is ARM_A-only; replay_window_separation is
    # ARM_A minus ARM_B; both unchanged from 1057a).
    _wb_late = [r for r in late_points if r.get("pass") == "whole_buffer"]
    replay_early_share_whole_buffer = (
        _mean([r["early_regime_share"] for r in _wb_late]) if _wb_late else 0.0
    )

    if order is None:
        window_label = {ARM_A: "ALL", ARM_B: str(recent_window),
                        ARM_C: "none"}.get(arm, "UNKNOWN_ARM")
    elif order == ORDER_WR:
        window_label = f"ALL->{recent_window} (final ALL->{recent_window}@k={k_final})"
    else:
        window_label = f"{recent_window}->ALL (final {recent_window}->ALL@k={k_final})"

    updates_e1 = sum(float(r.get("updates_e1", 0.0)) for r in cmc_records)
    updates_e2 = sum(float(r.get("updates_e2", 0.0)) for r in cmc_records)
    total_extra_steps = sum(float(r.get("n_updates", 0.0)) for r in cmc_records)
    # The dose actually SPENT at the final point's dosed pass, read back off the
    # consolidator's own counter rather than from k_final. This is what makes
    # the budget precondition a measurement and not a restatement of the config.
    _final_dosed = [r for r in cmc_records
                    if r.get("is_final_point") and r.get("is_dosed_pass")]
    # E1 steps, not n_updates -- see the UNITS note on _expected_ladder_budget.
    realized_final_dose = sum(float(r.get("updates_e1", 0.0)) for r in _final_dosed)
    expected_budget = (
        _expected_ladder_budget(n_points, cmc_steps, int(k_final))
        if order is not None else float("nan")
    )
    ladder_budget_error = (
        abs(updates_e1 - expected_budget) if order is not None else 0.0
    )

    only_early = occupancy["early"] - occupancy["late"]
    only_late = occupancy["late"] - occupancy["early"]
    union = occupancy["early"] | occupancy["late"]
    jaccard_distance = (len(only_early) + len(only_late)) / float(len(union)) if union else 0.0

    probe_digest = _tensor_digest(
        [ep["combined"].reshape(-1) for ep in early_probes + late_probes]
    )
    writepath_mode = str(getattr(agent.e1.config, "sd016_writepath_mode", "off"))
    finite = all(
        torch.isfinite(p).all().item()
        for m in (agent.e1, agent.e2) for p in m.parameters()
    )

    print(
        f"  {arm} seed={seed} e1_early={e1_early:.6g} e1_late={e1_late:.6g} "
        f"e1_skill_early={skill_early:.4f} forget_ratio={forgetting_ratio:.3f} "
        f"probe_var_early={var_early:.4g} "
        f"e2_early={e2_early:.6g} e2_late={e2_late:.6g} "
        f"extra_steps={total_extra_steps:.0f} final_dose={realized_final_dose:.0f} "
        f"updates_e1={updates_e1:.0f} updates_e2={updates_e2:.0f} "
        f"window={window_label} n_points={n_points} "
        f"replay_early_share={replay_early_share:.3f} "
        f"replay_early_share_whole_buffer_pass={replay_early_share_whole_buffer:.3f} "
        f"jaccard_dist={jaccard_distance:.3f} finite={finite}",
        flush=True,
    )
    cell_ok = bool(finite and e1_early == e1_early and e1_late == e1_late)
    print(f"verdict: {'PASS' if cell_ok else 'FAIL'}", flush=True)

    return {
        "arm": arm,
        "seed": seed,
        "ladder_order": order if order is not None else "",
        "ladder_k": float(k_final) if order is not None else float("nan"),
        "is_ladder_arm": bool(order is not None),
        "e1_holdout_mse_early": e1_early,
        "e1_holdout_mse_late": e1_late,
        "e2_selfforward_holdout_mse_early": e2_early,
        "e2_selfforward_holdout_mse_late": e2_late,
        "probe_target_variance_early": var_early,
        "probe_target_variance_late": var_late,
        "e1_persistence_baseline_mse_early": pers_early,
        "e1_persistence_baseline_mse_late": pers_late,
        "e1_skill_over_persistence_early": skill_early,
        "e1_skill_over_persistence_late": skill_late,
        "e1_holdout_mse_early_at_capture": e1_early_at_capture,
        "within_stratum_forgetting_ratio": forgetting_ratio,
        "recent_window_steps": recent_window,
        "total_extra_gradient_steps": total_extra_steps,
        "realized_final_dosed_steps": realized_final_dose,
        "expected_ladder_budget": expected_budget,
        "ladder_budget_error": ladder_budget_error,
        "n_consolidation_points": float(n_points),
        "updates_e1": updates_e1,
        "updates_e2": updates_e2,
        "replay_early_regime_share": replay_early_share,
        "replay_early_regime_share_whole_buffer_pass": replay_early_share_whole_buffer,
        "n_consolidation_passes": float(len([r for r in cmc_records if not r["skipped"]])),
        "consolidation_points": cmc_records,
        "early_late_jaccard_distance": jaccard_distance,
        "n_cells_early_only": len(only_early),
        "n_cells_late_only": len(only_late),
        "n_cells_union": len(union),
        "probe_digest": probe_digest,
        "sd016_writepath_mode": writepath_mode,
        "params_finite": finite,
        "cell_ok": cell_ok,
        "agent": agent,
    }


# ---------------------------------------------------------------------------
def _flat_scalar(d: Dict[str, object]) -> Dict[str, float]:
    """Flat numeric readout: bools as 0/1 ints, non-finite DROPPED."""
    out: Dict[str, float] = {}
    for k, v in d.items():
        if isinstance(v, bool):
            out[k] = int(v)
        elif isinstance(v, (int, float)):
            f = float(v)
            if f == f and abs(f) != float("inf"):
                out[k] = f
    return out


def _spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Spearman rho over a short, tie-free-by-construction abscissa.

    The abscissa is K_LADDER (strictly increasing, no ties). Ties on the ordinate
    are handled by average ranks. Returns nan for a degenerate ordinate (all
    equal), which _flat_scalar then DROPS rather than recording as a number --
    absent correctly reads as unmeasured, whereas 0.0 would read as "measured, no
    monotonicity".
    """
    n = len(xs)
    if n < 3 or len(ys) != n:
        return float("nan")

    def _ranks(vs: Sequence[float]) -> List[float]:
        order = sorted(range(len(vs)), key=lambda i: vs[i])
        rk = [0.0] * len(vs)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and vs[order[j + 1]] == vs[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for t in range(i, j + 1):
                rk[order[t]] = avg
            i = j + 1
        return rk

    rx, ry = _ranks(xs), _ranks(ys)
    mx, my = _mean(rx), _mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    if dx <= 0.0 or dy <= 0.0:
        return float("nan")
    return num / (dx * dy)


def main(dry_run: bool = False):
    """Returns (outcome, manifest_path). manifest_path is None on dry-run."""
    seeds = (SEEDS[0],) if dry_run else SEEDS
    warmup_eps = 1 if dry_run else WARMUP_EPISODES
    # A --dry-run life keeps BOTH blocks at 2 x CONSOLIDATE_EVERY so it still has
    # a NON-final and a FINAL late consolidation point. A one-point late block
    # would make every ladder arm's only late point the final one, and the smoke
    # would not exercise the is_final branch against its non-final sibling -- i.e.
    # the exact code the ladder adds would go untested by the smoke.
    early_eps = 2 * CONSOLIDATE_EVERY if dry_run else EARLY_EPISODES
    late_eps = 2 * CONSOLIDATE_EVERY if dry_run else LATE_EPISODES
    probe_eps = 1 if dry_run else PROBE_EPISODES
    episode_steps = 12 if dry_run else EPISODE_STEPS
    cmc_steps = 4 if dry_run else CMC_STEPS
    # The ladder is scaled with the dry-run budget so the endpoints stay at the
    # no-dose / full-dose extremes of the smoke's own CMC_STEPS. The FULL-SCALE
    # ladder is K_LADDER verbatim from the ratified sketch and is never rescaled.
    k_ladder = (0, 1, 2, cmc_steps) if dry_run else K_LADDER
    orders_seen = ORDERS
    ladder_arms = tuple(_ladder_arm(o, k) for o in orders_seen for k in k_ladder)
    arms = REFERENCE_ARMS + ladder_arms

    print(
        f"[{EXPERIMENT_TYPE}] starting (dry_run={dry_run}) n_arms={len(arms)} "
        f"k_ladder={k_ladder} seeds={seeds} "
        f"train_eps={warmup_eps + early_eps + late_eps} steps={episode_steps} "
        f"cmc_steps={cmc_steps} recent_window={_recent_window(episode_steps)}",
        flush=True,
    )
    t0 = time.time()

    rows: Dict[Tuple[str, int], Dict] = {}
    arm_results: List[Dict] = []
    agents_seen: List[REEAgent] = []
    for arm in arms:
        for seed in seeds:
            _lad = _parse_ladder_arm(arm)
            cfg_slice = {
                "experiment": EXPERIMENT_TYPE,
                "arm": arm,
                "ladder_order": _lad[0] if _lad else "",
                "ladder_k": _lad[1] if _lad else -1,
                "grid_size": GRID_SIZE,
                "num_hazards": N_HAZARDS,
                "num_resources": N_RESOURCES,
                "self_dim": SELF_DIM,
                "world_dim": WORLD_DIM,
                "alpha": ALPHA,
                "episode_steps": episode_steps,
                "warmup_episodes": warmup_eps,
                "early_episodes": early_eps,
                "late_episodes": late_eps,
                "probe_episodes": probe_eps,
                "consolidate_every": CONSOLIDATE_EVERY,
                "recent_window": _recent_window(episode_steps),
                "cmc_steps": cmc_steps,
                "wake_steps": WAKE_STEPS,
                "cmc_lr": CMC_LR,
                "cmc_batch": CMC_BATCH,
            }
            with arm_cell(seed, config_slice=cfg_slice, script_path=Path(__file__)) as cell:
                row = run_cell(
                    arm, seed, warmup_eps, early_eps, late_eps, probe_eps,
                    episode_steps, cmc_steps,
                )
                agents_seen.append(row.pop("agent"))
                cell.stamp(row)
            rows[(arm, seed)] = row
            arm_results.append(row)

    elapsed = time.time() - t0

    a_rows = [rows[(ARM_A, s)] for s in seeds]
    b_rows = [rows[(ARM_B, s)] for s in seeds]
    c_rows = [rows[(ARM_C, s)] for s in seeds]
    all_rows = [rows[(a, s)] for a in arms for s in seeds]
    ladder_rows = [rows[(a, s)] for a in ladder_arms for s in seeds]

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{ts}_v3"

    def _cells(order: str, k: int, key: str) -> List[float]:
        """Per-seed values of `key` for one ladder point, seed order preserved."""
        return [rows[(_ladder_arm(order, k), s)][key] for s in seeds]

    k_lo, k_hi = k_ladder[0], k_ladder[-1]

    # ---------------- the ladder itself -------------------------------------
    # Four legs: 2 orders x {early, late} probe stratum. The autopsy's null is
    # "both curves flat in k", asserted for both orders -- so four curves.
    STRATA = (("early", "e1_holdout_mse_early"), ("late", "e1_holdout_mse_late"))

    legs: Dict[str, Dict] = {}
    for order in orders_seen:
        for stratum, key in STRATA:
            leg = f"{ORDER_TAG[order]}_{stratum}"
            per_k_mean = {k: _mean(_cells(order, k, key)) for k in k_ladder}
            lo_vals, hi_vals = _cells(order, k_lo, key), _cells(order, k_hi, key)
            # Per-seed relative endpoint contrast. Seed-paired: the same seed's
            # k_lo and k_hi cells share env, action stream and probe set exactly,
            # so this differences out everything except the dose.
            rel_delta = [(hi - lo) / max(abs(lo), EPS)
                         for lo, hi in zip(lo_vals, hi_vals)]
            # SIGN-CONSISTENT counts, not a count of |rel_delta| > band: symmetric
            # seed noise must not read as a dose response.
            n_up = sum(1 for d in rel_delta if d > FLAT_BAND)
            n_down = sum(1 for d in rel_delta if d < -FLAT_BAND)
            n_consistent = max(n_up, n_down)
            nonflat = bool(n_consistent >= SIGN_CONSISTENCY_REQUIRED)
            # SHAPE -- reported, never gated (no prior distribution exists to set
            # a monotonicity floor from; see the docstring).
            rhos = [_spearman(list(k_ladder), [rows[(_ladder_arm(order, k), s)][key]
                                               for k in k_ladder])
                    for s in seeds]
            rhos_finite = [r for r in rhos if r == r]
            total_change = per_k_mean[k_hi] - per_k_mean[k_lo]
            k_half = float("nan")
            if abs(total_change) > 0.0:
                for k in k_ladder:
                    if abs(per_k_mean[k] - per_k_mean[k_lo]) >= 0.5 * abs(total_change):
                        k_half = float(k)
                        break
            legs[leg] = {
                "order": order, "stratum": stratum, "dv_key": key,
                "per_k_mean": per_k_mean,
                "per_k_values": {k: _cells(order, k, key) for k in k_ladder},
                "rel_delta_per_seed": rel_delta,
                "mean_rel_delta": _mean(rel_delta),
                "n_seeds_up": float(n_up), "n_seeds_down": float(n_down),
                "n_seeds_sign_consistent": float(n_consistent),
                "nonflat": nonflat,
                "spearman_rho_per_seed": rhos,
                "mean_spearman_rho": _mean(rhos_finite) if rhos_finite else float("nan"),
                "n_seeds_rho_ge_0p9": float(sum(1 for r in rhos_finite if abs(r) >= 0.9)),
                "k_half": k_half,
                "total_change_k_lo_to_k_hi": total_change,
            }

    nonflat_legs = [name for name, leg in legs.items() if leg["nonflat"]]
    n_nonflat_legs = len(nonflat_legs)

    # ORDER ASYMMETRY -- the H-schedule observable. Reported whether or not the
    # load-bearing criterion fires: a leg non-flat in ONE order and flat in the
    # MIRROR on the same stratum is the last-writer signature.
    order_asymmetry = {
        stratum: bool(legs[f"D1_{stratum}"]["nonflat"] != legs[f"D2_{stratum}"]["nonflat"])
        for stratum, _key in STRATA
    }

    # ---------------- retention gates: COMPUTED, REPORTED, NEVER GATING ------
    # See WHY THE RETENTION GATES ARE REPORTED HERE AND NEVER GATE. Both are kept
    # in the SAME form V3-EXQ-1048/1057/1057a used -- same DV, same comparator
    # arm, same seed-majority constant -- and evaluated at EVERY ladder point, so
    # the result is a retention DOSE-RESPONSE rather than a single pass/fail.
    def _retains_replay_gain(arm_name: str) -> int:
        """Seeds where this arm still carries replay's early-probe benefit over B."""
        return sum(1 for s in seeds
                   if rows[(arm_name, s)]["e1_holdout_mse_early"]
                   < rows[(ARM_B, s)]["e1_holdout_mse_early"])

    def _retains_recency_benefit(arm_name: str) -> int:
        """Seeds where this arm still beats A on the LATE probes."""
        return sum(1 for s in seeds
                   if rows[(arm_name, s)]["e1_holdout_mse_late"]
                   < rows[(ARM_A, s)]["e1_holdout_mse_late"])

    retention_curves = {
        ORDER_TAG[order]: {
            "replay_gain_by_k": {k: float(_retains_replay_gain(_ladder_arm(order, k)))
                                 for k in k_ladder},
            "recency_benefit_by_k": {k: float(_retains_recency_benefit(_ladder_arm(order, k)))
                                     for k in k_ladder},
        }
        for order in orders_seen
    }

    # ---------------- preconditions (the claim's own non-degeneracy list) ----
    # Unchanged in form from V3-EXQ-1057a except where marked NEW. None of these
    # is a retention gate.
    forgetting_ratio = _mean([r["within_stratum_forgetting_ratio"] for r in c_rows])
    min_replay_share_a = min(r["replay_early_regime_share"] for r in a_rows)
    replay_share_b = _mean([r["replay_early_regime_share"] for r in b_rows])
    min_updates_e1_a = min(r["updates_e1"] for r in a_rows)
    min_updates_e2_a = min(r["updates_e2"] for r in a_rows)
    min_jaccard = min(r["early_late_jaccard_distance"] for r in all_rows)
    min_probe_var = min(
        min(r["probe_target_variance_early"], r["probe_target_variance_late"])
        for r in all_rows
    )
    n_seeds_probe_match = sum(
        1 for s in seeds if len({rows[(a, s)]["probe_digest"] for a in arms}) == 1
    )
    n_writepath_off = sum(1 for r in all_rows if r["sd016_writepath_mode"] == "off")
    n_cells = len(arms) * len(seeds)
    budget_a = _mean([r["total_extra_gradient_steps"] for r in a_rows])
    budget_b = _mean([r["total_extra_gradient_steps"] for r in b_rows])

    # NEW (v): the ladder's abscissa must be the pre-registered dose. Worst
    # absolute error over EVERY ladder cell of |realized total extra steps -
    # ((n_points-1)*2*CMC + CMC + k)|. If the dose is not what was pre-registered
    # the x-axis of the whole ladder is wrong, which is an INSTRUMENT failure and
    # not evidence about MECH-017. Counterpart of 1057a's
    # additive_budget_is_a_plus_b, generalized across the ladder.
    worst_ladder_budget_error = max(
        (r["ladder_budget_error"] for r in ladder_rows), default=0.0
    )
    worst_budget_cell = max(
        ladder_rows, key=lambda r: r["ladder_budget_error"], default=None
    )

    min_window_separation = min(
        rows[(ARM_A, s)]["replay_early_regime_share"]
        - rows[(ARM_B, s)]["replay_early_regime_share"]
        for s in seeds
    )
    n_seeds_dv_moved = sum(
        1 for s in seeds
        if rows[(ARM_A, s)]["e1_holdout_mse_early"]
        != rows[(ARM_B, s)]["e1_holdout_mse_early"]
    )

    # NEW (vi): the DOSE must reach the DV. In EACH order, the number of seeds on
    # which the k_lo and k_hi cells differ AT ALL on the early-probe DV; the gate
    # takes the WORSE of the two orders. Without this, a bit-identical endpoint
    # pair -- a dead manipulation -- would be scored as a clean "flat in k" null.
    # This is what separates "flat because the dose does nothing" from "flat
    # because the dose was never applied", the one verdict aliasing this design
    # is exposed to.
    def _dose_moved(order: str) -> int:
        return sum(
            1 for s in seeds
            if rows[(_ladder_arm(order, k_lo), s)]["e1_holdout_mse_early"]
            != rows[(_ladder_arm(order, k_hi), s)]["e1_holdout_mse_early"]
        )

    n_seeds_dose_moved = min(_dose_moved(o) for o in orders_seen)

    # Persistence-skill gate, carried verbatim from 1057a (its red-team F2 fix):
    # per-seed MAX over the two COMPARED reference arms, never a min over every
    # arm -- ARM_C below persistence is the phenomenon, not a defect.
    per_seed_best_skill = {
        s: max(rows[(ARM_A, s)]["e1_skill_over_persistence_early"],
               rows[(ARM_B, s)]["e1_skill_over_persistence_early"])
        for s in seeds
    }
    min_best_skill_ab = min(per_seed_best_skill.values())
    worst_skill_seed = min(per_seed_best_skill, key=per_seed_best_skill.get)
    min_skill_ab = min(r["e1_skill_over_persistence_early"] for r in a_rows + b_rows)
    min_skill_c = min(r["e1_skill_over_persistence_early"] for r in c_rows)
    worst_forget_seed = min(
        seeds, key=lambda s: rows[(ARM_C, s)]["within_stratum_forgetting_ratio"]
    )

    try:
        preconditions = p0_readiness_gate([
            {"name": "forgetting_present_in_control", "measured": forgetting_ratio,
             "threshold": FORGETTING_FLOOR, "direction": "lower",
             "control": "MEAN over seeds, ARM_C only (no extra training): the WITHIN-STRATUM "
                        "ratio (EARLY-probe E1 MSE at end of life) / (same probe set, same arm, "
                        "measured immediately after capture, before the late block). Same probe "
                        "tensors on both sides, so intrinsic regime difficulty cancels. `met` is "
                        "the same central-tendency comparison the measured value reports. Below "
                        "the floor the model has not forgotten the early states, so there is "
                        "nothing for replay to counter and every ladder point ties. Read off an "
                        "arm no criterion of this run uses.",
             "worst_seed": int(worst_forget_seed)},
            {"name": "ladder_budget_is_pre_registered_dose",
             "measured": worst_ladder_budget_error, "threshold": 0.5, "direction": "upper",
             "control": "NEW for the ladder. WORST CELL (not the mean) over every ladder cell of "
                        "|realized consolidation E1 gradient steps - ((n_points-1)*2*CMC_STEPS + "
                        "CMC_STEPS + k)|, read off CrossModuleConsolidator's own updates_e1 "
                        "counter rather than from the config. updates_e1 and NOT n_updates: "
                        "n_updates counts one update PER MODULE per step (e1 AND e2), so it is "
                        "2x the pre-registered STEP count -- the first smoke of this script "
                        "reported 56 against an expected 28 on exactly that confusion. "
                        "updates_e1 equals the step count exactly and is the module the "
                        "load-bearing DV reads. The dose IS the abscissa of every "
                        "curve in this run, so a mis-dosed cell makes the x-axis wrong and is an "
                        "INSTRUMENT failure, not evidence about MECH-017. Counterpart of "
                        "V3-EXQ-1057a's additive_budget_is_a_plus_b, generalized across the "
                        "ladder. ARM_A/ARM_B/ARM_C are excluded by construction: they are not "
                        "ladder arms and carry no dose.",
             "offending_cell": (f"{worst_budget_cell['arm']}/seed{worst_budget_cell['seed']}"
                                if worst_budget_cell is not None else "none"),
             "reference_budget_a": budget_a, "reference_budget_b": budget_b},
            {"name": "dose_reaches_dv", "measured": float(n_seeds_dose_moved),
             "threshold": float(len(seeds)), "direction": "lower",
             "control": "NEW for the ladder. WORSE of the two orders of the number of seeds on "
                        f"which the k={k_lo} and k={k_hi} cells produced DIFFERENT early-probe "
                        "E1 MSE. A bit-identical endpoint pair is a DEAD MANIPULATION and would "
                        "otherwise be scored as a clean 'flat in k' null -- the one verdict "
                        "aliasing this design is exposed to. Same form and same threshold as "
                        "manipulation_reaches_dv, re-pointed at the ladder endpoints.",
             "per_order": {ORDER_TAG[o]: float(_dose_moved(o)) for o in orders_seen}},
            {"name": "replay_window_separation", "measured": min_window_separation,
             "threshold": MIN_WINDOW_SEPARATION, "direction": "lower",
             "control": "min over seeds of (ARM_A early-regime replay share - ARM_B early-regime "
                        "replay share). This is the WINDOW manipulation itself: if the two "
                        "windows do not separate, the whole-buffer and recent-window passes are "
                        "the same pass and the ladder doses nothing. Caught exactly this state "
                        "in the first smoke of V3-EXQ-1057."},
            {"name": "manipulation_reaches_dv", "measured": float(n_seeds_dv_moved),
             "threshold": float(len(seeds)), "direction": "lower",
             "control": "seeds where ARM_A and ARM_B produced DIFFERENT early-probe E1 MSE. "
                        "Unchanged from 1057a; retained because the ladder's two passes ARE "
                        "A's pass and B's pass, so a dead A-vs-B contrast is a dead ladder."},
            {"name": "e1_has_skill_over_persistence_compared_arms",
             "measured": min_best_skill_ab,
             "threshold": PERSISTENCE_SKILL_FLOOR, "direction": "lower", "comparator": ">",
             "control": "WORST SEED of max(ARM_A skill, ARM_B skill), where skill = 1 - (E1 "
                        "early-probe MSE / do-nothing persistence-baseline MSE on the same "
                        "windows). Ranges over the two reference arms the ladder is read "
                        "against -- NOT the ARM_C reference, whose sub-persistence skill is the "
                        "phenomenon under study. A seed where NEITHER reference arm beats a "
                        "do-nothing predictor makes that seed's whole column noise about noise.",
             "offending_seed": int(worst_skill_seed),
             "min_skill_over_compared_arms": min_skill_ab,
             "min_skill_arm_c_reference_only": min_skill_c},
            {"name": "gradient_budget_matched", "measured": abs(budget_a - budget_b),
             "threshold": 0.5, "direction": "upper",
             "control": "|extra gradient steps ARM_A - ARM_B|. The DESIGN INVARIANT that makes "
                        "the A-vs-B reference contrast a replay result rather than a compute "
                        "result. Deliberately still compares A against B ONLY: the ladder arms "
                        "are budget-UNMATCHED on purpose -- that IS the manipulation -- and are "
                        "covered by ladder_budget_is_pre_registered_dose instead."},
            {"name": "replay_covers_early_regime", "measured": min_replay_share_a,
             "threshold": REPLAY_EARLY_SHARE_FLOOR, "direction": "lower",
             "control": "ARM_A, min over seeds of the mean early-regime share of the buffer "
                        "slice visible at the LATE-block consolidation points. "
                        "compute_prediction_loss draws start_idx ~ Uniform[0, buf_len-1), so "
                        f"this share IS the per-draw probability of an early-regime trace. "
                        f"ARM_B comparator = {replay_share_b:.4f}.",
             "arm_b_share": replay_share_b},
            {"name": "consolidation_updated_e1", "measured": min_updates_e1_a,
             "threshold": UPDATES_FLOOR, "direction": "lower",
             "control": "ARM_A, min over seeds of CrossModuleConsolidator's own updates_e1 "
                        "counter -- the pass must actually move the module the DV reads."},
            {"name": "consolidation_updated_e2", "measured": min_updates_e2_a,
             "threshold": UPDATES_FLOOR, "direction": "lower",
             "control": "ARM_A, min over seeds of updates_e2 (E2 SELF-forward; secondary)."},
            {"name": "early_late_state_divergence", "measured": min_jaccard,
             "threshold": STATE_DIVERGENCE_FLOOR, "direction": "lower",
             "control": "min over ALL cells of the Jaccard DISTANCE between early-regime and "
                        "late-regime occupied grid cells. If old and recent states coincide "
                        "there is nothing for replay to add (sleep_substrate:GAP-2)."},
            {"name": "probe_set_identical_across_arms", "measured": float(n_seeds_probe_match),
             "threshold": float(len(seeds)), "direction": "lower",
             "control": "sha256 over each cell's concatenated probe [z_self, z_world] tensors; "
                        "ALL arms of a seed must agree EXACTLY, which is what makes the probe "
                        "set FIXED rather than arm-dependent. With 13 arms this is a strictly "
                        "stronger assertion than 1057a's 5-arm form."},
            {"name": "probe_target_variance", "measured": min_probe_var,
             "threshold": PROBE_TARGET_VAR_FLOOR, "direction": "lower",
             "control": "min over all cells/strata of the variance of the held-out E1 target "
                        "tensor -- a frozen target would make the MSE trivially satisfiable, "
                        "and a pinned DV is the other way a 'flat in k' null could be vacuous."},
            {"name": "sd016_writepath_mode_off", "measured": float(n_writepath_off),
             "threshold": float(n_cells), "direction": "lower",
             "control": "runtime read of E1Config.sd016_writepath_mode in every cell. 'off' "
                        "means the open CORRUPTING defect "
                        "contextmemory-write-path-addressing-degeneracy "
                        "(e1_deep.py::ContextMemory.write) is never called on this run's path."},
        ])
        gate_ok = True
    except P0NotReady as e:
        preconditions = e.preconditions
        gate_ok = False

    # Ranges over EVERY cell including all ten ladder cells -- V3-EXQ-1057's
    # red-team F1: a non-finite cell silently counts as an ordinary comparison
    # (nan <= y is False) and routes to a CLAIM VERDICT rather than to
    # instrument-not-ready. The highest-k ladder arms take the most gradient of
    # any arm, so they are exactly the cells this must cover.
    cells_ok = all(r["cell_ok"] for r in all_rows)

    base_manifest = {
        "schema_version": "v1",
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "supersedes": SUPERSEDES,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "sleep_driver_pattern": (
            "N/A -- no SleepLoopManager built; the MECH-423 R3 CrossModuleConsolidator "
            "is called directly (V3-EXQ-680e call shape). Sleep-cycle wiring of this "
            "consolidator is separately validated by V3-EXQ-1026."
        ),
        "timestamp_utc": ts,
        "started_utc": datetime.utcnow().isoformat() + "Z",
        "arm_results": arm_results,
        "substrate_limits_declared": {
            "e2_world_forward_readout": (
                "NOT ATTEMPTED. substrate_queue 'e2-world-forward-sleep-trainer' "
                "(pending_implementation, degrading): CrossModuleConsolidator updates only E2's "
                "SELF-forward head and E2.world_forward has no trainer in ree_core, so a "
                "world_forward DV is identically 0.0 in every arm. The E2 numbers reported here "
                "are SELF-forward and are secondary / non-load-bearing."
            ),
            "contextmemory_write_defect": (
                "OFF-PATH. sd016_writepath_mode='off' in every cell (asserted as a "
                "precondition), so e1_deep.py::ContextMemory.write is never called."
            ),
            "probe_time_lstm_dropout": (
                "LIVE, declared, NOT fixed. e1_deep.py:648 builds the transition LSTM with "
                "dropout=0.1 at the default num_layers=3, and torch.no_grad() does not disable "
                "it, so the held-out probe samples a dropout mask that differs per arm with RNG "
                "consumption. V3-EXQ-1057a measured the resulting spread at 0.81% on a "
                "toy-scale trained agent. NOT fixed here because calling .eval() would change "
                "the DV distribution relative to V3-EXQ-1048/1057/1057a, whose A/B/C cells this "
                "run reproduces bit-identically and reads as its prior. This 0.8% IS the noise "
                "band FLAT_BAND (0.10) is set 12.5x above."
            ),
            "d3_interleaved_arm_owed": (
                "OWED, NOT BUILT. The autopsy's second fan-out probe (interleave whole-buffer "
                "and recent-window steps inside ONE optimisation) has no producer: "
                "consolidate() takes one schedule for the whole call and its 'interleaved' mode "
                "interleaves MODULES, not DATA WINDOWS (cross_module_consolidation.py:185-201); "
                "there is no per-step window argument and no mixture-buffer utility exists. A "
                "driver-side n_steps=1 alternation is NOT equivalent -- each call builds a fresh "
                "per-module Adam (lines 155-162), so it would cold-start momentum every step. "
                "D3 needs /implement-substrate first."
            ),
        },
        # ANCHOR REACHABILITY -- recorded, not asserted by synthetic replay.
        # `validate_experiments --checks anchor_reachability` WARNs on this script
        # (advisory, WARN-only in both modes; the script exits OK). The warning is
        # LEFT STANDING, deliberately un-silenced, and disposed of here in writing.
        #
        # WHY IT FIRES AT ALL: the lint is scoped to `diagnostic`/`baseline`
        # scripts. V3-EXQ-1057a carries THE SAME precondition set with THE SAME
        # self-route and does NOT trip it, purely because it is `evidence`. So this
        # warning is a consequence of this run's deliberate purpose upgrade, not of
        # a new predicate.
        #
        # WHY IT IS NOT SILENCED WITH ANCHOR_REACHABILITY_EXEMPT: that marker is
        # for the case where the predicate IS the degeneracy definition, so a
        # replay would be tautological. That is true of this set's STRUCTURAL
        # anchors (exact equality / arithmetic identity) but NOT of its numeric
        # FLOORS, so a blanket exemption would over-claim. The lint's own docstring
        # names reaching for EXEMPT in the wrong circumstance as the documented
        # error.
        #
        # WHAT IS OFFERED INSTEAD, and why it is stronger than a synthetic replay:
        # every inherited anchor has already been MEASURED GREEN AT FULL SCALE,
        # TWICE, on this exact harness and machine_class (1057: 14/15 gates green;
        # 1057a: 17/18) -- the recorded values below. The single gate that failed on
        # both runs was the RETENTION gate, which this run deliberately does not
        # carry as a precondition at all (see the retention discussion above). The
        # two NEW anchors are reachable by construction and were confirmed in the
        # smoke: ladder_budget error was exactly 0 at every dosed cell (updates_e1
        # 28/29/30/32 against pre-registered 28/29/30/32), and dose_reaches_dv is a
        # float-inequality that the smoke showed separating on every rung.
        "anchor_reachability_evidence": {
            "lint_status": "WARN left standing, un-silenced; disposed of in writing",
            "why_it_fires": ("anchor_reachability_lint is scoped to diagnostic/baseline "
                             "purpose; V3-EXQ-1057a carries the same precondition set "
                             "under `evidence` and does not trip it"),
            "recorded_full_scale_clearances": {
                "forgetting_present_in_control": {
                    "threshold": FORGETTING_FLOOR,
                    "v3_exq_1057": 2.4853917328308244, "v3_exq_1057a": 2.4750167490267976},
                "replay_covers_early_regime": {
                    "threshold": REPLAY_EARLY_SHARE_FLOOR,
                    "v3_exq_1057": 0.7333333333333334, "v3_exq_1057a": 0.7333333333333334},
                "replay_window_separation": {
                    "threshold": MIN_WINDOW_SEPARATION,
                    "v3_exq_1057": 0.7333333333333334, "v3_exq_1057a": 0.7333333333333334},
                "consolidation_updated_e1": {
                    "threshold": UPDATES_FLOOR, "v3_exq_1057": 96.0, "v3_exq_1057a": 96.0},
                "consolidation_updated_e2": {
                    "threshold": UPDATES_FLOOR, "v3_exq_1057": 96.0, "v3_exq_1057a": 96.0},
                "early_late_state_divergence": {
                    "threshold": STATE_DIVERGENCE_FLOOR,
                    "v3_exq_1057": 0.8780487804878049, "v3_exq_1057a": 0.8780487804878049},
                "probe_target_variance": {
                    "threshold": PROBE_TARGET_VAR_FLOOR,
                    "v3_exq_1057": 0.006881691515445709,
                    "v3_exq_1057a": 0.006881691515445709},
                "manipulation_reaches_dv": {
                    "threshold": "all seeds", "v3_exq_1057": 5.0, "v3_exq_1057a": 5.0},
                "probe_set_identical_across_arms": {
                    "threshold": "all seeds", "v3_exq_1057": 5.0, "v3_exq_1057a": 5.0},
                "e1_has_skill_over_persistence_compared_arms": {
                    "threshold": PERSISTENCE_SKILL_FLOOR,
                    "v3_exq_1057": 0.1054424323446228,
                    "v3_exq_1057a": 0.10544178957951644,
                    "note": "clears a 0.0 floor but by the narrowest margin of the set"},
            },
            "new_anchors_reachable_by_construction": {
                "ladder_budget_is_pre_registered_dose": (
                    "arithmetic identity on the consolidator's own updates_e1 counter; "
                    "smoke measured error exactly 0 at every rung (28/29/30/32 against "
                    "pre-registered 28/29/30/32)"),
                "dose_reaches_dv": (
                    "float inequality between two cells; smoke showed the endpoints "
                    "separating in both orders on every rung"),
            },
        },
        "config": {
            "arms": list(arms),
            "reference_arms": list(REFERENCE_ARMS),
            "ladder_arms": list(ladder_arms),
            "k_ladder": list(k_ladder),
            "k_ladder_full_scale": list(K_LADDER),
            "orders": list(orders_seen),
            "dosed_pass_by_order": {ORDER_WR: "recent_window", ORDER_RW: "whole_buffer"},
            "dose_scope": "final consolidation point only; the first (n_points - 1) "
                          "points run both passes at cmc_steps",
            "seeds": list(seeds),
            "grid_size": GRID_SIZE,
            "num_hazards": N_HAZARDS,
            "num_resources": N_RESOURCES,
            "self_dim": SELF_DIM,
            "world_dim": WORLD_DIM,
            "alpha_world": ALPHA,
            "alpha_self": ALPHA,
            "episode_steps": episode_steps,
            "warmup_episodes": warmup_eps,
            "early_episodes": early_eps,
            "late_episodes": late_eps,
            "probe_episodes": probe_eps,
            "train_episodes": warmup_eps + early_eps + late_eps,
            "consolidate_every": CONSOLIDATE_EVERY,
            "recent_window": _recent_window(episode_steps),
            "cmc_steps": cmc_steps,
            "wake_steps": WAKE_STEPS,
            "cmc_lr": CMC_LR,
            "cmc_batch": CMC_BATCH,
            "early_start": list(EARLY_START),
            "late_start": list(LATE_START),
            "hazards": [list(h) for h in HAZARDS],
            "resources": [list(r) for r in RESOURCES],
            "registered_thresholds": {
                "flat_band": FLAT_BAND,
                "sign_consistency_required": SIGN_CONSISTENCY_REQUIRED,
                "late_tol": LATE_TOL,
                "min_window_separation": MIN_WINDOW_SEPARATION,
                "persistence_skill_floor": PERSISTENCE_SKILL_FLOOR,
                "wake_lr": WAKE_LR,
                "cmc_lr": CMC_LR,
                "min_rel_gain_early": MIN_REL_GAIN_EARLY,
                "forgetting_floor": FORGETTING_FLOOR,
                "replay_early_share_floor": REPLAY_EARLY_SHARE_FLOOR,
                "updates_floor": UPDATES_FLOOR,
                "state_divergence_floor": STATE_DIVERGENCE_FLOOR,
                "probe_target_var_floor": PROBE_TARGET_VAR_FLOOR,
            },
        },
        "ladder_legs": legs,
        "retention_curves_reported_never_gating": retention_curves,
        "elapsed_seconds": elapsed,
    }

    def _write(manifest: Dict) -> Optional[str]:
        if dry_run:
            print(f"[{EXPERIMENT_TYPE}] dry-run complete; not writing manifest.")
            return None
        out_path = write_flat_manifest(
            manifest,
            dry_run=False,
            config=manifest.get("config"),
            seeds=list(seeds),
            script_path=Path(__file__),
            agent=agents_seen,
        )
        print(f"Result written to: {out_path}")
        return str(out_path)

    # ---------------- FORK: any precondition unmet -> not a MECH-017 verdict --
    if not gate_ok or not cells_ok:
        unmet = [p["name"] for p in preconditions if not p.get("met", True)]
        if not cells_ok:
            unmet = unmet + ["cell_parameters_finite"]
        reason = "unmet preconditions: " + ", ".join(unmet)
        print(f"\n[{EXPERIMENT_TYPE}] -> substrate_not_ready_requeue: {reason}")
        manifest = dict(base_manifest)
        manifest.update({
            "outcome": "FAIL",
            "result": "FAIL",
            "evidence_direction": "inconclusive",
            "evidence_direction_note": (
                "Non-degeneracy precondition unmet; this run does NOT bear on MECH-017 in "
                "either direction. " + reason
            ),
            "non_degenerate": False,
            "degeneracy_reason": "substrate_not_ready: " + reason,
            "interpretation": {
                "label": "substrate_not_ready_requeue",
                "preconditions": preconditions,
                "criteria_non_degenerate": {},
            },
            "readout": _flat_scalar({
                "gate_ok": False,
                "forgetting_ratio_control": forgetting_ratio,
                "min_replay_early_share_a": min_replay_share_a,
                "replay_early_share_b": replay_share_b,
                "min_updates_e1_a": min_updates_e1_a,
                "min_updates_e2_a": min_updates_e2_a,
                "min_early_late_jaccard_distance": min_jaccard,
                "n_seeds_probe_set_identical": float(n_seeds_probe_match),
                "min_probe_target_variance": min_probe_var,
                "min_replay_window_separation": min_window_separation,
                "n_seeds_dv_moved": float(n_seeds_dv_moved),
                "n_seeds_dose_moved": float(n_seeds_dose_moved),
                "worst_ladder_budget_error": worst_ladder_budget_error,
                "min_per_seed_best_skill_ab": min_best_skill_ab,
                "min_skill_ab": min_skill_ab,
                "min_skill_c": min_skill_c,
            }),
        })
        return "FAIL", _write(manifest)

    # ---------------- criteria ----------------------------------------------
    early_a = [r["e1_holdout_mse_early"] for r in a_rows]
    early_b = [r["e1_holdout_mse_early"] for r in b_rows]
    early_c = [r["e1_holdout_mse_early"] for r in c_rows]
    late_a = [r["e1_holdout_mse_late"] for r in a_rows]
    late_b = [r["e1_holdout_mse_late"] for r in b_rows]
    late_c = [r["e1_holdout_mse_late"] for r in c_rows]

    # Reference legs, carried from V3-EXQ-1048/1057/1057a UNCHANGED and
    # load_bearing:false. They are a DETERMINISTIC RE-DERIVATION on this
    # machine_class, not independent replication (all 15 A/B/C cells of 1057 came
    # back bit-identical to 1048's), so they pin the harness against drift and
    # supply the comparator the retention curves are defined against -- and must
    # not be cited as if they added confirmation.
    n_a_better_early = sum(1 for x, y in zip(early_a, early_b) if x < y)
    n_a_not_worse_late = sum(
        1 for x, y in zip(late_a, late_b) if x <= y * (1.0 + LATE_TOL)
    )
    rel_gain_early = _mean([(y - x) / max(y, EPS) for x, y in zip(early_a, early_b)])
    c_ref1 = bool(n_a_better_early >= SIGN_CONSISTENCY_REQUIRED)
    c_ref2 = bool(n_a_not_worse_late >= SIGN_CONSISTENCY_REQUIRED)
    c_ref3 = bool(rel_gain_early > MIN_REL_GAIN_EARLY)

    # THE load-bearing criterion. The autopsy's null is "both curves flat in k",
    # asserted for both orders -- a CONJUNCTION over four legs -- so rejecting it
    # is a DISJUNCTION. That shape is forced by the null's own wording, not chosen
    # for sensitivity; every leg is reported individually below so no reader has
    # to trust the OR.
    c1 = bool(n_nonflat_legs >= 1)

    criteria = [
        {"name": "C1_dose_response_detected", "load_bearing": True,
         "passed": c1, "measured": float(n_nonflat_legs), "threshold": 1.0,
         "comparator": ">=",
         "detail": (
             f"Number of the FOUR ladder legs (2 orders x {{early, late}} probe stratum) that "
             f"are NON-FLAT in k. A leg is non-flat iff the per-seed relative endpoint contrast "
             f"(mse[k={k_hi}] - mse[k={k_lo}]) / |mse[k={k_lo}]| exceeds FLAT_BAND={FLAT_BAND} "
             f"in magnitude WITH A CONSISTENT SIGN on at least {SIGN_CONSISTENCY_REQUIRED} of "
             f"{len(seeds)} seeds (the sign-consistent count is max(n_up, n_down), NOT the "
             f"count of |rel_delta| > band -- symmetric seed noise must not read as a dose "
             f"response). PASS = the autopsy's 'both curves flat in k' null is REJECTED: the "
             f"step count of the FINAL consolidation pass moves held-out E1 fidelity. "
             f"FAIL = the null HOLDS, which is informative and not a shrug -- it falsifies the "
             f"last-writer reading that the V3-EXQ-1057/1057a pass-order swing suggests, and "
             f"says the displacement is a property of the whole four-point schedule rather "
             f"than of its last writer. CONSTANT PROVENANCE: FLAT_BAND is LATE_TOL, reused "
             f"unchanged from V3-EXQ-1048 C2 -> 1057 C4 -> 1057a C4; "
             f"SIGN_CONSISTENCY_REQUIRED is unchanged from all three. Nothing was invented. "
             f"Legs non-flat: {nonflat_legs if nonflat_legs else 'none'}."),
         "nonflat_legs": nonflat_legs,
         "per_leg": {name: {"nonflat": leg["nonflat"],
                            "n_seeds_sign_consistent": leg["n_seeds_sign_consistent"],
                            "mean_rel_delta": leg["mean_rel_delta"]}
                     for name, leg in legs.items()}},
        {"name": "C2_shape_monotonicity_REPORTED_ONLY", "load_bearing": False,
         "passed": True,
         "measured": _mean([leg["mean_spearman_rho"] for leg in legs.values()
                            if leg["mean_spearman_rho"] == leg["mean_spearman_rho"]] or [0.0]),
         "threshold_not_applicable": (
             "SHAPE is the scientific payload of this run and is deliberately NOT gated. "
             "Mean per-seed Spearman rho of the DV against k, and k_half (the smallest k at "
             "which the mean curve has covered >= 50% of its total k_lo -> k_hi change), "
             "distinguish 'gradual trade-off' from 'a step early' -- the question the autopsy "
             "says the intermediate points exist to answer. No floor is set because no prior "
             "distribution exists to set one from: only k=24 has ever been measured, and only "
             "as a whole-life dose. Inventing a monotonicity threshold here would be an "
             "unpre-registered constant. Same disposition V3-EXQ-1057a took for its "
             "late-probe skill measurement (its red-team F4): MEASURE now, let a successor "
             "set a floor once there is a distribution. `passed` is True by construction and "
             "carries no information -- read the per-leg rho and k_half values."),
         "detail": "per-leg mean Spearman rho and k_half are in readout and ladder_legs."},
        {"name": "C3_retention_dose_response_REPORTED_ONLY", "load_bearing": False,
         "passed": True,
         "measured": float(retention_curves[ORDER_TAG[ORDER_WR]]["replay_gain_by_k"][k_hi]),
         "threshold_not_applicable": (
             "The two retention gates of V3-EXQ-1048/1057/1057a are KEPT -- computed in their "
             "original form (replay gain vs ARM_B on early probes; recency benefit vs ARM_A on "
             "late probes; same seed-majority constant) and evaluated at EVERY ladder point, so "
             "the record is a retention DOSE-RESPONSE rather than a single pass/fail. They do "
             "NOT gate, and must not: (a) V3-EXQ-1057 measured the whole-then-recent order at "
             "0/5 on replay-gain retention, and ARM_D1_K24 reproduces that order exactly, so an "
             "AND'd whole-run gate would be a pre-registered guaranteed failure that self-routes "
             "before a single ladder point is reported; (b) at k=0 the gates are structurally "
             "not meaningful -- there is no full additive dose at the final point to retain "
             "anything with; (c) retention LOSS at high k is the PHENOMENON this ladder "
             "measures, so gating on it would gate away the dependent variable. This is the "
             "disposition CLAUDE.md's multi-arm precondition rule mandates (scope the gate, "
             "never lower the threshold) and the EXACT precedent V3-EXQ-1057a set in this same "
             "lineage for ARM_D1. `measured` is reported for shape only."),
         "detail": "retention_curves_reported_never_gating carries both curves, per order, per k."},
        {"name": "C_REF1_replay_beats_budget_matched_on_early_probes", "load_bearing": False,
         "passed": c_ref1, "measured": float(n_a_better_early),
         "threshold": float(SIGN_CONSISTENCY_REQUIRED), "comparator": ">=",
         "detail": "DETERMINISTIC RE-DERIVATION of V3-EXQ-1048 C1 (A vs B at matched budget), "
                   "NOT load-bearing and NOT independent replication: all 15 A/B/C cells of "
                   "V3-EXQ-1057 came back BIT-IDENTICAL to V3-EXQ-1048's on this "
                   "machine_class. Carried because it is free, it pins the harness against "
                   "drift, and ARM_B is the comparator the replay-gain retention curve is "
                   "defined against."},
        {"name": "C_REF2_no_cost_on_late_probes", "load_bearing": False,
         "passed": c_ref2, "measured": float(n_a_not_worse_late),
         "threshold": float(SIGN_CONSISTENCY_REQUIRED), "comparator": ">=",
         "detail": f"RE-DERIVATION of V3-EXQ-1048 C2, NOT load-bearing: seeds with "
                   f"e1_holdout_mse_late(A) <= (1+{LATE_TOL})*e1_holdout_mse_late(B). It "
                   "FAILED on 1048 and 1057; it is the matched-budget deficit the whole "
                   "additive-budget line exists to explain, and ARM_A is the comparator the "
                   "recency-benefit retention curve is defined against."},
        {"name": "C_REF3_early_effect_size_clears_floor", "load_bearing": False,
         "passed": c_ref3, "measured": rel_gain_early, "threshold": MIN_REL_GAIN_EARLY,
         "comparator": ">",
         "detail": "RE-DERIVATION of V3-EXQ-1048 C3, NOT load-bearing: mean over seeds of "
                   "(early_B - early_A)/early_B."},
    ]

    criteria_non_degenerate = {
        # The ladder criterion is degenerate iff the DV does not vary across the
        # ladder at all. Ranges over EVERY ladder cell in every leg, not just the
        # endpoints, so a ladder that moved only between two interior points still
        # reads non-degenerate.
        "C1_dose_response_detected": bool(
            len({round(v, 12) for leg in legs.values()
                 for vals in leg["per_k_values"].values() for v in vals}) > 1
        ),
        "C2_shape_monotonicity_REPORTED_ONLY": bool(len(k_ladder) >= 3),
        "C3_retention_dose_response_REPORTED_ONLY": bool(
            len({round(x, 12) for x in early_a + early_b}) > 1
        ),
        "C_REF1_replay_beats_budget_matched_on_early_probes": bool(
            len({round(x, 12) for x in early_a + early_b}) > 1
        ),
        "C_REF2_no_cost_on_late_probes": bool(
            len({round(x, 12) for x in late_a + late_b}) > 1
        ),
        "C_REF3_early_effect_size_clears_floor": bool(
            len({round(x, 12) for x in early_a + early_b}) > 1
        ),
    }

    degeneracy = check_degeneracy({
        "e1_holdout_mse_early_ladder_d1": {
            "groups": [[legs["D1_early"]["per_k_values"][k][i] for k in k_ladder]
                       for i in range(len(seeds))]},
        "e1_holdout_mse_early_ladder_d2": {
            "groups": [[legs["D2_early"]["per_k_values"][k][i] for k in k_ladder]
                       for i in range(len(seeds))]},
        "e1_holdout_mse_late_ladder_d1": {
            "groups": [[legs["D1_late"]["per_k_values"][k][i] for k in k_ladder]
                       for i in range(len(seeds))]},
        "e1_holdout_mse_late_ladder_d2": {
            "groups": [[legs["D2_late"]["per_k_values"][k][i] for k in k_ladder]
                       for i in range(len(seeds))]},
        "e1_holdout_mse_early_reference": {
            "groups": [[a, b] for a, b in zip(early_a, early_b)]},
    })

    all_pass = all(c["passed"] for c in criteria if c["load_bearing"])

    # ---------------- verdict grid ------------------------------------------
    # Every branch records evidence_direction `non_contributory` for MECH-017.
    # That is a DESIGN PROPERTY, stated so it cannot be read as an oversight:
    # this run's load-bearing criterion is about the SHAPE of a dose curve on a
    # consolidation SCHEDULE, and neither its PASS nor its FAIL makes MECH-017's
    # "replay counters forgetting at no cost to recency" more or less likely. What
    # the grid DOES discriminate -- which of the autopsy's three live hypotheses
    # the data favours -- is carried in `label` and in the machine-readable
    # `hypothesis_verdict` block, which is what routes the next decision (build D3
    # via /implement-substrate, or accept V3-EXQ-1048's mixed reading as the
    # record, per the autopsy's own stop rule).
    d1_early_nf = legs["D1_early"]["nonflat"]
    d2_early_nf = legs["D2_early"]["nonflat"]
    d1_late_nf = legs["D1_late"]["nonflat"]
    d2_late_nf = legs["D2_late"]["nonflat"]
    # Same-sign test over the two orders, per stratum: a shared direction is the
    # budget/intrinsic signature; opposite or single-order movement is the
    # schedule signature.
    def _same_sign(leg_a: str, leg_b: str) -> bool:
        a, b = legs[leg_a]["mean_rel_delta"], legs[leg_b]["mean_rel_delta"]
        return bool(a * b > 0.0)

    early_symmetric = bool(d1_early_nf and d2_early_nf and _same_sign("D1_early", "D2_early"))
    late_symmetric = bool(d1_late_nf and d2_late_nf and _same_sign("D1_late", "D2_late"))
    early_asymmetric = bool(d1_early_nf != d2_early_nf)
    late_asymmetric = bool(d1_late_nf != d2_late_nf)

    if not c1:
        label = "final_pass_dose_flat_displacement_not_attributable_to_last_writer"
        hypothesis_verdict = {
            "H-schedule": "disfavoured_as_last_writer_effect",
            "H-reallocation": "undetermined",
            "H-intrinsic": "undetermined",
            "reading": (
                "All four legs flat. Titrating the FINAL consolidation pass does not move "
                "held-out E1 fidelity, so the V3-EXQ-1057-vs-1057a pass-order swing is NOT "
                "produced by the last writer alone; it is a property of the whole four-point "
                "blocked schedule. This is an informative null: it removes the cheapest "
                "mechanism for the cluster's structural property and routes the next question "
                "to the schedule as a whole."),
        }
    elif early_asymmetric or late_asymmetric:
        label = "final_pass_dose_response_order_asymmetric_schedule_signature"
        hypothesis_verdict = {
            "H-schedule": "favoured",
            "H-reallocation": "disfavoured",
            "H-intrinsic": "disfavoured",
            "reading": (
                "The dose moves the DV in ONE order and not in its MIRROR, at identical total "
                "budget and identical windows. A cost that depends on WHICH window is written "
                "last, and not on how much budget went where, is a schedule artefact. This "
                "UNDERMINES the prior reading that the recency cost is a fact about replay, "
                "without SUPPORTING MECH-017's recency conjunct -- which is why the direction "
                "below is non_contributory and not `supports`."),
        }
    elif early_symmetric or late_symmetric:
        label = "final_pass_dose_response_order_symmetric_budget_signature"
        hypothesis_verdict = {
            "H-schedule": "disfavoured",
            "H-reallocation": "favoured" if late_symmetric else "undetermined",
            "H-intrinsic": "favoured" if late_symmetric else "undetermined",
            "reading": (
                "Both orders move with k in the SAME direction. k ADDS steps rather than "
                "moving them, so a shared dose response points at the amount of non-recent "
                "training rather than at its position in the schedule. H-reallocation and "
                "H-intrinsic are not separated by this run -- separating them is exactly what "
                "the OWED interleaved arm D3 is for."),
        }
    else:
        label = "final_pass_dose_response_detected_pattern_unclassified"
        hypothesis_verdict = {
            "H-schedule": "undetermined",
            "H-reallocation": "undetermined",
            "H-intrinsic": "undetermined",
            "reading": (
                "At least one leg is non-flat but the pattern matches neither the "
                "order-asymmetric (schedule) nor the order-symmetric (budget) signature -- "
                "e.g. both orders move in OPPOSITE directions on the same stratum, or only a "
                "cross-stratum subset fires. Recorded as unclassified rather than forced into "
                "a hypothesis; read the per-leg curves."),
        }
    direction = "non_contributory"

    note = (
        f"DOSE LADDER on the FINAL consolidation pass. {len(arms)} arms x {len(seeds)} seeds. "
        f"After a FIXED pass at cmc_steps={cmc_steps}, the LAST pass at the FINAL of "
        f"{int(all_rows[0]['n_consolidation_points'])} consolidation points runs k in "
        f"{list(k_ladder)} steps; the first points run the full unchanged additive "
        f"construction for that order. ORDER_WR doses the recent-window pass (V3-EXQ-1057's "
        f"D1 lineage), ORDER_RW doses the whole-buffer pass (V3-EXQ-1057a's D2 lineage, the "
        f"MIRROR). "
        f"VERDICT: {label}; {n_nonflat_legs}/4 legs non-flat"
        f"{' (' + ', '.join(nonflat_legs) + ')' if nonflat_legs else ''} against "
        f"FLAT_BAND={FLAT_BAND} on >={SIGN_CONSISTENCY_REQUIRED}/{len(seeds)} sign-consistent "
        f"seeds. Mean rel_delta k={k_lo}->k={k_hi}: "
        + ", ".join(f"{n} {legs[n]['mean_rel_delta']:+.4f}" for n in sorted(legs)) + ". "
        f"Mean Spearman rho (SHAPE, reported not gated): "
        + ", ".join(f"{n} {legs[n]['mean_spearman_rho']:+.3f}" for n in sorted(legs)) + ". "
        f"k_half: " + ", ".join(f"{n} {legs[n]['k_half']}" for n in sorted(legs)) + ". "
        f"PURPOSE IS `diagnostic`, DELIBERATELY AND UNLIKE ITS PREDECESSORS "
        f"(V3-EXQ-1057/1057a were `evidence`): the load-bearing criterion here is the SHAPE "
        f"of a dose curve on a consolidation SCHEDULE, and neither its PASS nor its FAIL makes "
        f"MECH-017's 'replay counters forgetting at no cost to recency' more or less likely. "
        f"The ratified autopsy's own debt_class for this node is 'complex (probe-gated)', and "
        f"a probe on such a node is a diagnostic whose job is to convert it to "
        f"'complicated (buildable)' backlog. Scoring a curve-shape criterion into MECH-017's "
        f"confidence would corrupt it; every branch of the grid therefore records "
        f"non_contributory, by design and not by oversight. IF A REVIEWER DISAGREES this is a "
        f"one-line change to EXPERIMENT_PURPOSE. "
        f"RETENTION GATES KEPT BUT NEVER GATING (routing_detail 'Both retention gates kept'): "
        f"both are computed in their V3-EXQ-1048/1057/1057a form at EVERY ladder point, giving "
        f"a retention DOSE-RESPONSE. Gating on them would be a pre-registered guaranteed "
        f"failure (ARM_D1_K{k_hi} reproduces the order V3-EXQ-1057 measured at 0/5), would be "
        f"structurally meaningless at k=0, and would gate away the dependent variable -- the "
        f"exact precedent V3-EXQ-1057a set for ARM_D1. replay-gain retention by k: "
        + "; ".join(f"{tag} " + ",".join(f"k{k}={int(v)}" for k, v in cur['replay_gain_by_k'].items())
                    for tag, cur in retention_curves.items()) + ". "
        f"recency-benefit retention by k: "
        + "; ".join(f"{tag} " + ",".join(f"k{k}={int(v)}" for k, v in cur['recency_benefit_by_k'].items())
                    for tag, cur in retention_curves.items()) + ". "
        f"ANCHORING: k={k_hi} reproduces V3-EXQ-1057a's ARM_D1/ARM_D2 construction EXACTLY "
        f"(all points at both passes x cmc_steps), so a departure from the recorded 1.74e-04 / "
        f"7.48e-05 early means is harness drift. k={k_lo} does NOT reproduce ARM_A -- it still "
        f"runs the dosed pass at the non-final points -- so the pre-flight's claim that both "
        f"endpoints were already measured is HALF WRONG and is recorded as such in the "
        f"docstring; ARM_A/ARM_B/ARM_C are carried in-run precisely so the ladder is read "
        f"against in-run anchors. Reference legs (RE-DERIVATION, not replication, and "
        f"non-load-bearing): C_REF1 {n_a_better_early}/{len(seeds)}, C_REF2 "
        f"{n_a_not_worse_late}/{len(seeds)}, C_REF3 rel gain {rel_gain_early:.4f}; control "
        f"(ARM_C) forgetting ratio {forgetting_ratio:.3f}. "
        f"D3 (the interleaved arm) IS OWED, NOT DONE: no producer exists -- consolidate() "
        f"takes one schedule per call and interleaves MODULES not DATA WINDOWS, there is no "
        f"per-step window argument, and a driver-side n_steps=1 alternation is NOT equivalent "
        f"because each call builds a fresh per-module Adam. Route to /implement-substrate. "
        f"SCOPE: MECH-017 'is two claims wearing one id' (V3-EXQ-1048's confirmed autopsy); "
        f"this run adjudicates neither. E2 numbers are SELF-forward, NOT the world_forward "
        f"readout MECH-017 names -- that head has no trainer in ree_core."
    )

    print(f"\n[{EXPERIMENT_TYPE}] verdict:")
    for c in criteria:
        _m = c.get("measured")
        _t = c.get("threshold", "n/a")
        print(f"  {c['name']}: passed={c['passed']} measured="
              f"{_m:.6g} threshold={_t}" if isinstance(_m, (int, float))
              else f"  {c['name']}: passed={c['passed']}")
    for name in sorted(legs):
        leg = legs[name]
        print(f"  leg {name}: nonflat={leg['nonflat']} "
              f"mean_rel_delta={leg['mean_rel_delta']:+.4f} "
              f"sign_consistent={leg['n_seeds_sign_consistent']:.0f}/{len(seeds)} "
              f"rho={leg['mean_spearman_rho']:+.3f} k_half={leg['k_half']} "
              f"curve=" + ",".join(f"k{k}:{leg['per_k_mean'][k]:.4g}" for k in k_ladder))
    print(f"  -> {label} ({'PASS' if all_pass else 'FAIL'}); elapsed={elapsed:.1f}s")

    manifest = dict(base_manifest)
    manifest.update({
        "outcome": "PASS" if all_pass else "FAIL",
        "result": "PASS" if all_pass else "FAIL",
        "evidence_direction": direction,
        "evidence_direction_note": note,
        "hypothesis_verdict": hypothesis_verdict,
        "interpretation": {
            "label": label,
            "criteria": criteria,
            "combination_rule": (
                "PASS iff the single LOAD-BEARING criterion C1_dose_response_detected passes "
                "(plain all() over the load_bearing subset). C1 is itself a DISJUNCTION over "
                "the four ladder legs -- n_nonflat_legs >= 1 -- and that shape is FORCED by "
                "the autopsy's own null, which is the conjunction 'both curves flat in k' "
                "asserted for both orders; the negation of a conjunction is a disjunction. It "
                "was not chosen for sensitivity, and every leg's count, sign and effect size "
                "is reported individually in `interpretation.criteria[0].per_leg` and in "
                "`ladder_legs`, so no reader has to trust the OR. Each leg's own test is a "
                "SIGN-CONSISTENT seed majority (max(n_up, n_down) >= "
                "SIGN_CONSISTENCY_REQUIRED), not a count of |rel_delta| > band, so symmetric "
                "seed noise cannot make a leg fire. "
                "C2 (shape/monotonicity) and C3 (retention dose-response) are "
                "load_bearing:false and carry threshold_not_applicable with their reasons: C2 "
                "because no prior distribution exists from which to set a monotonicity floor, "
                "C3 because the two retention gates are kept as REPORTED curves and must "
                "never gate (gating them would be a pre-registered guaranteed failure at "
                "k=k_hi in the D1 order, would be structurally meaningless at k=0, and would "
                "gate away the dependent variable -- the precedent V3-EXQ-1057a set for "
                "ARM_D1). C_REF1/2/3 are non-load-bearing re-derivations of V3-EXQ-1048's "
                "matched-budget legs, carried to pin the harness and to supply the comparator "
                "arms the retention curves are defined against. "
                "Every non-degeneracy assertion -- the pre-registered ladder dose, the dose "
                "reaching the DV, window separation, probe-set identity, probe-target "
                "variance, forgetting in the control, and the rest -- is a PRECONDITION, so "
                "any of them failing routes to substrate_not_ready_requeue instead of to a "
                "MECH-017 verdict."),
            "criteria_non_degenerate": criteria_non_degenerate,
            "preconditions": preconditions,
        },
        "readout": _flat_scalar({
            "gate_ok": True,
            "overall_pass": all_pass,
            "c1_dose_response_detected": c1,
            "n_nonflat_legs": float(n_nonflat_legs),
            # --- the ladder, flat and machine-readable -----------------------
            **{f"leg_{name}_nonflat": leg["nonflat"] for name, leg in legs.items()},
            **{f"leg_{name}_mean_rel_delta": leg["mean_rel_delta"]
               for name, leg in legs.items()},
            **{f"leg_{name}_n_seeds_sign_consistent": leg["n_seeds_sign_consistent"]
               for name, leg in legs.items()},
            **{f"leg_{name}_mean_spearman_rho": leg["mean_spearman_rho"]
               for name, leg in legs.items()},
            **{f"leg_{name}_n_seeds_rho_ge_0p9": leg["n_seeds_rho_ge_0p9"]
               for name, leg in legs.items()},
            **{f"leg_{name}_k_half": leg["k_half"] for name, leg in legs.items()},
            **{f"mean_{leg['dv_key']}_{ORDER_TAG[leg['order']]}_k{k:02d}": v
               for leg in legs.values() for k, v in leg["per_k_mean"].items()},
            # --- retention dose-response (reported, never gating) ------------
            **{f"n_seeds_{tag}_retains_replay_gain_k{k:02d}": v
               for tag, cur in retention_curves.items()
               for k, v in cur["replay_gain_by_k"].items()},
            **{f"n_seeds_{tag}_retains_recency_benefit_k{k:02d}": v
               for tag, cur in retention_curves.items()
               for k, v in cur["recency_benefit_by_k"].items()},
            # --- order asymmetry (the H-schedule observable) -----------------
            "order_asymmetry_early": order_asymmetry["early"],
            "order_asymmetry_late": order_asymmetry["late"],
            "early_symmetric_same_sign": early_symmetric,
            "late_symmetric_same_sign": late_symmetric,
            # --- reference arms ----------------------------------------------
            "mean_e1_holdout_mse_early_a": _mean(early_a),
            "mean_e1_holdout_mse_early_b": _mean(early_b),
            "mean_e1_holdout_mse_early_c": _mean(early_c),
            "mean_e1_holdout_mse_late_a": _mean(late_a),
            "mean_e1_holdout_mse_late_b": _mean(late_b),
            "mean_e1_holdout_mse_late_c": _mean(late_c),
            "n_seeds_a_better_early": float(n_a_better_early),
            "n_seeds_a_not_worse_late": float(n_a_not_worse_late),
            "rel_gain_early_a_over_b": rel_gain_early,
            "c_ref1_pass": c_ref1, "c_ref2_pass": c_ref2, "c_ref3_pass": c_ref3,
            # --- preconditions, as numbers ------------------------------------
            "forgetting_ratio_control": forgetting_ratio,
            "worst_ladder_budget_error": worst_ladder_budget_error,
            "n_seeds_dose_moved": float(n_seeds_dose_moved),
            "n_seeds_dv_moved": float(n_seeds_dv_moved),
            "min_replay_window_separation": min_window_separation,
            "min_replay_early_share_a": min_replay_share_a,
            "replay_early_share_b": replay_share_b,
            "min_updates_e1_a": min_updates_e1_a,
            "min_updates_e2_a": min_updates_e2_a,
            "min_early_late_jaccard_distance": min_jaccard,
            "n_seeds_probe_set_identical": float(n_seeds_probe_match),
            "min_probe_target_variance": min_probe_var,
            "min_per_seed_best_skill_ab": min_best_skill_ab,
            "min_skill_ab": min_skill_ab,
            "min_skill_c": min_skill_c,
            "gradient_budget_delta_a_minus_b": abs(budget_a - budget_b),
            "extra_gradient_steps_a": budget_a,
            "extra_gradient_steps_b": budget_b,
            # --- late-probe skill: measured, NOT gated (1057a red-team F4) ----
            "min_e1_skill_over_persistence_late_a": min(
                r["e1_skill_over_persistence_late"] for r in a_rows),
            "min_e1_skill_over_persistence_late_b": min(
                r["e1_skill_over_persistence_late"] for r in b_rows),
            "min_e1_skill_over_persistence_late_ladder": min(
                r["e1_skill_over_persistence_late"] for r in ladder_rows),
        }),
        # Per-seed values retained in full for every cell -- never collapsed to
        # mean+-sd (Experimental Recording Standard sec 3c; the 732a/738 cost).
        "per_seed_results": [
            {
                "seed": s,
                "e1_early_a": rows[(ARM_A, s)]["e1_holdout_mse_early"],
                "e1_early_b": rows[(ARM_B, s)]["e1_holdout_mse_early"],
                "e1_early_c": rows[(ARM_C, s)]["e1_holdout_mse_early"],
                "e1_late_a": rows[(ARM_A, s)]["e1_holdout_mse_late"],
                "e1_late_b": rows[(ARM_B, s)]["e1_holdout_mse_late"],
                "e1_late_c": rows[(ARM_C, s)]["e1_holdout_mse_late"],
                "e2_early_a": rows[(ARM_A, s)]["e2_selfforward_holdout_mse_early"],
                "e2_early_b": rows[(ARM_B, s)]["e2_selfforward_holdout_mse_early"],
                "e2_early_c": rows[(ARM_C, s)]["e2_selfforward_holdout_mse_early"],
                "ladder": {
                    a: {
                        "k": rows[(a, s)]["ladder_k"],
                        "order": rows[(a, s)]["ladder_order"],
                        "e1_early": rows[(a, s)]["e1_holdout_mse_early"],
                        "e1_late": rows[(a, s)]["e1_holdout_mse_late"],
                        "e2_early": rows[(a, s)]["e2_selfforward_holdout_mse_early"],
                        "e2_late": rows[(a, s)]["e2_selfforward_holdout_mse_late"],
                        "extra_steps": rows[(a, s)]["total_extra_gradient_steps"],
                        "realized_final_dose": rows[(a, s)]["realized_final_dosed_steps"],
                        "skill_late": rows[(a, s)]["e1_skill_over_persistence_late"],
                    }
                    for a in ladder_arms
                },
                "probe_digests": {a: rows[(a, s)]["probe_digest"] for a in arms},
            }
            for s in seeds
        ],
    })
    manifest.update(degeneracy)
    return manifest["outcome"], _write(manifest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Smoke run, no manifest.")
    args = parser.parse_args()
    _outcome, _manifest_path = main(dry_run=args.dry_run)
    _clean = str(_outcome).upper() if str(_outcome).upper() in ("PASS", "FAIL") else "FAIL"
    emit_outcome(outcome=_clean, manifest_path=_manifest_path, dry_run=args.dry_run)
    sys.exit(0)
