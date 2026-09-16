"""V3-EXQ-1044 -- Hippocampal campaign ASSAY A: access mechanism at a frozen interface.

Implements Gates A0 (instrument), A1 (the decisive access-mechanism contrast) and A2
(specificity) of
`REE_assembly/evidence/planning/hippocampal_campaign_assay_specifications_20260910.md`
sections 1, 2 and 4. Read-only over FROZEN endpoints: no `ree_core` change, no agent
warm-up, no new substrate, no behavioural training of any endpoint.

EXPERIMENT_PURPOSE = "diagnostic"

SLEEP DRIVER: not applicable -- no sleep flag is set anywhere in this driver (no REEAgent
is constructed at all). Recorded as sleep_driver_pattern="none".

red-team (Step 4.5): see the RED-TEAM RECORD section at the end of this docstring and the
V3-EXQ-1044 queue entry note.

=== THE QUESTION (spec section 2.1) ===

At a frozen interface with an adequate source, does a CONDITIONAL transformation add
held-out consumer use beyond good retrieval, fixed partner-specific routing, and frame
conditioning -- and if it does, is the gain attributable to the sender's content or to
information the RECEIVER supplied?

=== WHAT THE FIRST DRAFT GOT WRONG, AND WHAT CHANGED (red-team BLOCKING, 2026-09-16) ===

The first draft of this driver fitted every bridge by MSE to the NATIVE field and presented
both frames as a row stratum. The red-team reviewer (opus) showed from the source that under
that objective the receiver-state block `A4_receiver_state_cond` is manipulated on is
EXACTLY INDEPENDENT of the regression target (the native field does not depend on which
query the receiver is running), so weight decay drives A4's state weights to zero and A4
collapses onto `A1_source_only` -- while the frame bit, which every bridge NEEDS to invert
a two-frame stratum, was handed to `A3_frame_cond` alone. C1 could therefore only ever
record `A3 > A4`, and the run would have reported "conditional access earns nothing" from an
arm that was never given an objective in which receiver state could matter. Verified against
the source (old lines 1418 / 1480 / 953) and against a design probe.

Two structural changes close that, and a design probe (seed 42, 30 episodes, 120 head
passes, 60 bridge passes -- recorded below) confirms the redesign discriminates:

  1. THE FIT OBJECTIVE IS CONSUMER USE. Every fitted arm is optimised END-TO-END through the
     FROZEN heads (head weights fixed, gradient flows to the message): the bridge's loss is
     the frozen consumer's cross-entropy on the query-specific oracle action. That is the
     section 1.6 primary metric itself, so the map the bridge must learn now depends on
     what the receiver is doing -- and the DV is still the frozen head's own committed
     action on held-out rows (rung 6). The bridge only ever sees the source field plus its
     permitted conditioning bits, so nothing it produces can carry information that is not
     in the source; what conditioning can change is WHICH source content reaches the head.

  2. BANDWIDTH IS THE BUDGET AXIS. With a full-width channel the native field is the one
     optimal message for every query, so receiver state can play no role by arithmetic (the
     reviewer's point survives the objective change at full width; the probe shows A1 == A4
     at M >= 4). The spec's L4 rung is "constrained nonlinear (NARROW BOTTLENECK, weight
     decay, no recurrence)", and the bottleneck width M IS the capacity budget of section
     1.3. So every fitted arm is a `[message_in ; conditioning] -> tanh(M) -> 25` bridge and
     M is SWEPT over M_GRID; the two evaluation widths are SELECTED IN-RUN on the
     model-selection block (block 2, otherwise unused) by pre-registered rules, and scored
     ONCE on block 3.

Design probe (seed 42, single rot90 frame, end-to-end fit; block-3 consumer-use agreement):

    M  | A1_source_only | A4_receiver_state_cond | A4 state-permuted | A5_ident_cond
     2 |     0.359      |        0.574           |       0.526       |     0.544
     3 |     0.400      |        0.726           |       0.589       |     0.530
     4 |     0.741      |        0.641           |       0.533       |     0.696
     6 |     0.719      |        0.733           |       0.678       |     0.719
    25 |     0.748      |        0.756           |       0.704       |     0.715
    (native canonical 0.748; A0 under rot90 0.244; strongest trivial 0.419)

The conditional advantage is LARGE below the unconditional bridge's sufficiency width and
ZERO above it, and identity conditioning shows a real partner-routing effect at M = 2. A
single pre-registered M would therefore make C1's verdict an artefact of that integer. The
estimand is split into the two regimes the probe exposes, both pre-registered here.

=== WHY GATE A1 PRESENTS ONE FRAME (the rotation), NOT TWO ===

Spec section 2.6 stages Gate A1 on ONE frame. Under the canonical frame there is nothing to
translate: the frozen consumer reads the source natively at its ceiling, so every bridge is
bounded above by ~0 gain before the run starts (section 2.8 "all arms ~= A0_native").
Gate A1 therefore presents the known invertible 90-degree rotation of section 2.2 (the same
transform G4 uses as its positive control) on EVERY row: the frozen consumer was trained on
the canonical frame, so under rot90 its native access is BROKEN and `A0_native` is a genuine
floor. G9 re-measures the break IN-RUN and refuses the run if it did not happen.

Consequence, declared rather than hidden: at a single frame `A3_frame_cond`'s conditioning
block is a CONSTANT one-hot, so A3 is informationally identical to `A1_source_only` and acts
as a second, independently initialised draw of the unconditional bridge inside the
best-rival set (which can only make C1 HARDER to pass). Its `cond_message_spread`
precondition is scoped out (disposition (a): not meaningful at one frame). Frame mediation
(R2) is the subject of Gate A3's frame x query factorial, deferred to a successor.

=== WHY THE QUERY SET IS NOT THE SPEC'S LITERAL THREE ===

Spec section 2.2 names `approach_nearest_resource`, `avoid_nearest_hazard`, `hold_position`.
`avoid_nearest_hazard` is undefined on this rung (`D3_hazard_free`: `num_hazards: 0`);
`hold_position` is a CONSTANT label (always action 4), on which every arm including the
zero-message control scores 1.0. Both are replaced with queries that are computable from env
ground truth on a hazard-free rung, non-constant, and genuinely different action mappings
over the SAME state:

  q0 approach    move along the dominant axis TOWARD the highest-valued local cell
  q1 retreat     the sign-flipped q0
  q2 orthogonal  move along the axis ORTHOGONAL to the nearest-resource direction

All three are functions of the same nearest-resource offset. That is deliberate and it is
what makes the bandwidth question sharp: an UNCONDITIONAL bridge must transmit content the
head can read under all three queries at once, a CONDITIONED one only under the query in
force. Non-vacuity is asserted in-run (`query_not_state_blind`: no query's majority-class
share may reach STATE_BLIND_CEILING) and the per-query label balance is recorded.

=== WHY THREE HEADS AND WHY EACH SEES EVERY QUERY ===

Spec section 2.2: receiver STATE is which query the head is executing; receiver IDENTITY is
which of the k frozen heads the message is aimed at. If head h only ever executed query h
the two would be the same variable and A4 / A5 would be aliased. So each of K_HEADS = 3
frozen heads is behaviour-cloned on ALL K_QUERIES = 3 queries from `(canonical field, query
one-hot)`, differing only by initialisation seed. Identity is then a real axis orthogonal
to state. Query and identity are assigned to rows deterministically by row index (every
(query, identity) cell equally populated; independent of content and of each other).

=== THE ARMS (one architecture, one budget, information differs) ===

Every fitted arm is `[message_in (25) ; conditioning (COND_DIM = 8)] -> tanh(M) -> 25`,
identical width, identical parameter count at a given M, identical rows, passes,
optimiser, learning rate and grad-clip (section 1.3). Arms differ ONLY in which sub-blocks
of the conditioning vector carry information; the others are constant zeros of the same
width. All bridges are fitted end-to-end through the frozen heads (above).

  A0_native              identity map, no bridge -- the FLOOR reference under the broken
                         frame; not a budget-matched competitor, excluded from rivals
  A1_source_only         all conditioning blocks zero (the unconditional constrained bridge)
  A2_index_route         CONTENT-ADDRESSED reinstatement: the stored NATIVE associate of the
                         nearest stored rotated source in the fitting block (L2 nearest
                         neighbour, leave-one-EPISODE-out at fit time so fitting mirrors the
                         held-out situation), then a fixed per-identity LINEAR route of rank
                         M, fitted end-to-end like every other arm. An EXACT (episode, step)
                         lookup would hand the arm the answer; content-addressed recall is
                         imperfect on novel input and is the honest operationalisation.
  A3_frame_cond          frame block live (a CONSTANT at Gate A1 -- see above)
  A4_receiver_state_cond state (query) block live -- the VERDICT arm, MECH-547 strong reading
  A5_receiver_ident_cond identity (head) block live -- tranche 3 A1, "who is listening"
  A6_selector_k          message squeezed to a K_SELECTOR-way HARD index (nearest of
                         K_SELECTOR k-means centroids fitted on the fitting block -- the most
                         informative k-way quantisation the sender can choose), then a fixed
                         random code; state + identity blocks live (receiver + query supply
                         the mapping -- tranche 3 A3, T3-10)
  A7_receiver_only       message channel ZEROED; state + identity blocks live (the
                         receiver-only information control, section 1.4)

Best-rival set of the primary estimand: {A1, A2, A3, A5, A6}. A0 fits nothing; A7 is the
zero-message control.

=== THE BANDWIDTH LADDER AND THE TWO PRE-REGISTERED EVALUATION WIDTHS ===

Every sweep arm is fitted on block 1 at every M in M_GRID and scored on block 2 (selection)
and block 3 (test). Per arm, `m_suff(arm)` = the NARROWEST M at which the arm's block-2
consumer-use agreement is ELEVATED over the strongest trivial predictor (measured on block
2) by at least AGREEMENT_ELEVATION_MIN -- i.e. the narrowest channel at which that access
mechanism is functional. Then:

  M*     = min over the rival set of m_suff(rival): the tightest budget at which ANY
           unconditional / routing mechanism works. This is the spec's "matched budget"
           evaluation point: comparing A4 to a rival at a budget where the rival is starved
           would measure starvation, not conditioning.
  M_A4   = m_suff(A4): the narrowest channel at which the conditioned arm is functional.

The four episode blocks are STRATIFIED BY DRIVER (oracle-driven and random-driven
episodes are split by BLOCK_FRACS separately and merged per block, asserted in-run): a
contiguous cut of the collection order would have put all-oracle episodes in the fitting
block and all-random ones in the test block, so every criterion would have measured transfer
across a visitation shift rather than bandwidth (red-team pass 2, F1).

Both are chosen on block 2 only; block 3 is read once, after every arm is frozen. Every
(arm, M) cell is fitted N_RESTARTS = 3 times from different initialisations and the restart
with the best block-2 agreement is the cell (model selection on the selection block, applied
identically to every arm; the single-seed confirmer showed a one-shot unconditional fit is
non-monotone in M at 60 passes). All restarts' values are recorded per cell.

=== PRE-REGISTERED CRITERIA ===

C1b (LOAD-BEARING; the spec's primary estimand, section 2.6) -- at M*, the paired per-seed
block-3 consumer-use gain of A4 over the BEST-performing rival at M* satisfies BOTH
`mean(delta) >= DELTA_FLOOR` (0.05) AND `mean(delta) >= 2 * SD(delta)` across seeds, on at
least SEED_MAJORITY (2) of 3 seeds individually. Positive = receiver-state-conditioned
access adds consumer use even where an unconditional mechanism already suffices.

C1a (LOAD-BEARING; the bandwidth regime) -- at M_A4 (the narrowest width at which the
conditioned arm is functional on block 2), the paired per-seed block-3 gain of A4 over the
best rival at M_A4 satisfies the SAME section 1.6 rule as C1b. Positive = conditional access
adds use where the channel is narrower than the unconditional map's sufficiency. The ordinal
saving M* - M_A4 is recorded per seed as a DIAGNOSTIC, not a criterion: a bare integer
crossing has no variability discipline and pits one A4 draw against a minimum over five
rivals (red-team pass 2, F2). At every width the best rival's IDENTITY is chosen on block 2
and its block-3 value is what A4 is compared to (F3: never model-select on the test block).

C2 -- `A7_receiver_only` sits BELOW AGREEMENT_BAR at every M and seed. If the receiver alone
answers the query, no interface claim is licensed. A7's information is exactly the query
and head one-hots, which can at most reproduce the per-query majority class, so this
control is EXPECTED to pass; it is recorded as `structurally_expected_pass` with that
reason rather than presented as a discriminating test.

Gate A2 refinements, evaluated at M_eval = M* if C1b passed else M_A4:

C3 -- A4's gain is STATE-SPECIFIC: A4 minus A4-under-receiver-state-permutation (the bridge
is fed a query one-hot shuffled WITHIN the identity stratum; the head still receives the true
query) is positive by the section 1.6 rule. Failing = the advantage survives state
permutation, so it was never conditioned on state. NOTE the design probe: a wrong-query
message still carries the shared nearest-resource content, so the permuted arm sits well
above A1 -- which is why "dies" is the paired CONTRAST against intact A4, not an absolute
floor.

C4 -- IDENTITY conditioning by itself adds nothing: `A5_receiver_ident_cond` minus the best
unconditional arm ({A1, A3}) at M_eval stays below DELTA_FLOOR (mean over seeds). This is
the fallible form of the section 2.8 routing test: A4 does not see the identity block and
the three heads are interchangeable by construction (same rows, same labels, different
initialisation), so delivering A4's message to a shuffled head cannot fail (red-team pass 2,
F5) -- that permutation is recorded as a diagnostic (`A4_head_permuted`), not scored. C3
failing together with C4 failing (state does not matter, partner does) is the section 2.8
"fixed partner-specific routing" row.

C5 -- SENDER-STATE NULL, as a pipeline-integrity check: `A1_source_only`'s produced message
is INVARIANT under the receiver state (cross-state range of the message == 0, exactly),
i.e. the state block did not leak into the unconditional arm. The spec's null (tranche 3
A2) targets a sender whose CODE differs across receiver states; in this frozen-source
design the source is recorded before any query is assigned and the assignment is by row
index, so the null holds by construction and the check can only fail through a defect.
A1's per-state CONSUMER-USE agreement is recorded as a diagnostic (it differs across
queries because the queries differ in difficulty, which is a property of the head, not of
the sender's code) and is NOT the criterion.

C6 -- RECEIVER-CONTRIBUTION separation, the spec's conjunction (section 2.8 row 6): FAILS
only if `A7_receiver_only` is HIGH (at or above the per-query-constant predictor's level
plus DELTA_FLOOR) AND `D_joint` exceeds `D_sender` by more than DELTA_FLOOR. The first
draft applied the second clause alone; on a query-conditioned task `D_joint - D_sender` is
large BY CONSTRUCTION (a query-blind decoder of a query-specific label is capped near the
majority share), so the clause alone relabelled every positive run. The three decodings are
still reported separately and never summed (section 1.4), decoders fitted on block 1 and
scored on block 3; message-only decodings of A4's and A1's produced messages are recorded
as diagnostics of how query-shaped the conditioned message is.

C7 (MECH-548, spec section 1.8) -- REPEATED-USE STABILITY, genuinely closed-loop IN THE
ENVIRONMENT: the frozen head (head 0, query q0) is driven for ROLLOUT_EPISODES = 30 whole
episodes with EVERY observation passing through the arm's fitted interface at M_eval, so the
consumer's own resulting state (the next observation) is what it feeds forward. Per-step
agreement with the q0 oracle along the self-generated trajectory is denominated on EVERY
started episode -- an episode the policy let die counts as non-agreement afterwards, with the
alive fraction recorded alongside so death and disagreement stay separable (red-team pass 2,
F4: survivor-denominated shares over 3-4 episodes were being read against a bar from another
distribution). C7 passes if agreement at step N_REPEATED_USE (20) is not below the closed-loop
trivial bar measured on THAT policy's own trajectory (max of the q0 majority share and the
previous-executed-action agreement along it). Foraging competence (rung 7) is recorded
alongside, with A0-native and a random policy as references. C7 is read at M_eval, i.e. at
the width the positive C1 finding is about; when that is M_A4 it is the conditioned arm's
narrowest functional width, and a one-step-only reading there says exactly that.

combination_rule: Gate A1 is adjudicated on ((C1b OR C1a) AND C2); C1b and C1a name two
regimes and are reported separately. Gate A2's C3..C7 REFINE the label at M_eval; they do
not retro-fit C1. C2 and C5 are declared structurally-expected guards and are serialised
`load_bearing: False` (red-team pass 2, F7); C5 red is an instrument refusal, not a science
verdict.

=== NULL TABLE -- every branch is informative (spec section 2.8) ===

  C1b pass, C2, C3, C4, C6, C7 pass -> receiver-state-conditioned access adds held-out
       consumer use AT THIS INTERFACE even at a budget where an unconditional mechanism
       suffices, attributable to sender content. FORBIDDEN: any biological statement.
  C1b FAIL, C1a pass (+ C2..C7)      -> conditional access buys BANDWIDTH ONLY: below M*
       the receiver's state is necessary for a functional interface; at M* the named best
       rival matches A4, so the strong R3/MECH-547 reading is unnecessary at sufficient
       bandwidth (the T3-10 direction), and necessary below it.
  C1a and C1b FAIL                   -> conditional access earns nothing at any width in
       the grid; the best rival is named per seed (A2: indexed reinstatement suffices; A6:
       a low-cardinality selector + the receiver's query suffices; A1/A3: the unconditional
       bridge suffices; A5: partner routing suffices).
  C3 FAIL and C4 FAIL                -> fixed partner-specific routing, not conditioning
       (state does not matter, partner does; section 2.7's sharper falsifier).
  C3 FAIL, C4 pass                   -> the advantage survives state permutation and identity
       conditioning adds nothing either: attributable to neither; recorded as unattributed
       (a conditioned bridge with ANY varying auxiliary input fitting better at that width).
  C2 FAIL                            -> the receiver supplied the information; no claim.
  C6 FAIL                            -> receiver supplied the information; bridge introduced
       rather than translated it.
  C7 FAIL                            -> one-step-only; MECH-548 instability blocks adoption.
  C5 red                             -> pipeline defect (state leaked into A1): refusal.
  G9 red                             -> the rotation did not break access:
       `substrate_not_ready_requeue`.
  G10 red / NEITHER A4 nor any rival functional at any M -> nothing bridgeable at
       admissible complexity: instrument refusal. (Rivals non-functional everywhere while A4
       is functional is a RESULT: M* falls back to M_A4 and C1b is scored there.)

A C1b failure is the OUTCOME THE PRIOR FAVOURS (tranche 3 section 4.2(2), T3-10); C1a is
where the probe says the effect lives. Both directions are informative.

=== DV-SYMMETRY INVARIANCE (mandatory per-arm declaration) ===

DV = mean over held-out block-3 rows of 1[argmax(frozen_head(message, query_onehot)) ==
query-specific oracle action]. Symmetry group: any transformation of the message that leaves
the frozen head's ARGMAX unchanged -- a uniform additive constant on the head's 5 output
logits, any monotone rescaling of those logits. Per arm:

  A0_native      manipulation = the presentation frame (a coordinate PERMUTATION of the
                 25-dim field). A permutation of the head's INPUT is not a broadcast on its
                 OUTPUT logits; the head is a nonlinear MLP. Measured by G9, not argued.
  A1_source_only manipulation = a fitted rank-M message map. Changes the head's input
                 vector, not its output by a constant. Not invariant.
  A2_index_route manipulation = WHICH stored native associate is retrieved, through a
                 rank-M route. Different retrievals are different inputs, not an offset.
  A3/A4/A5/A6    manipulation = the INFORMATION on a conditioning sub-block at FIXED width.
                 It enters BEFORE the tanh bottleneck, so its effect on the message is
                 state-dependent, not a broadcast; distinct conditions produce distinct
                 messages, asserted in-run by `cond_message_spread` (a cross-condition
                 RANGE, the same statistic family the criterion routes on). A3's block is
                 constant at one frame and its spread is scoped out, declared above.
  A7_receiver_only manipulation = ZEROING the message. A degenerate limit ON PURPOSE: A7 is
                 the control whose job is to show the DV does NOT move when the sender is
                 removed. Excluded from the best-rival set.
  bandwidth M    manipulation = the bottleneck width. A rank change is not a logit
                 broadcast; the probe shows the DV moving by > 0.3 across the grid.

=== INSTRUMENT GATES (spec section 1.7, plus G9 and G10) ===

  G1  source adequacy        rawfield25 @ CONSUMER_RUNG (external decoder, q0) clears
                             AGREEMENT_BAR
  G2  consumer range         oracle foraging competence - random competence >= 5.0
  G3  consumer floor         frozen heads on the CANONICAL field (no bridge), ELEVATED over
                             the strongest trivial predictor by >= AGREEMENT_ELEVATION_MIN.
                             The elevation form: AGREEMENT_BAR = 0.80 was calibrated on a
                             SINGLE-task decoder (V3-EXQ-1002); this consumer is a THREE-task
                             query-conditioned head on the same 25 inputs.
  G4  known-inverse control  the exact analytic inverse of the presentation, realised as a
                             LINEAR ROUTE MODULE and pushed through the same
                             _FittedArm.message -> _heads_logits -> _score_arm path every
                             fitted arm takes, returns the head to within the equivalence
                             band of G3 (a PIPELINE-INTEGRITY control, not an information
                             control; an index round trip scored directly is bit-identical
                             to G3 -- red-team pass 2, F8)
  G5  negative-control floor zero message and moment-matched random message at or below the
                             STRONGEST TRIVIAL PREDICTOR + band. The strongest trivial
                             predictor on this three-query task is re-measured in-run as the
                             max of (previous executed action) and (the per-query constant
                             predictor, majority class chosen on the fitting block); on this
                             task the second dominates (0.42 vs 0.17), and comparing a
                             zero-message head to the wrong one is what turned G5 red in the
                             first draft's smoke.
  G6  capacity witness       the top ladder rung memorises the fitting block
  G7  no divergence          no fit DIVERGED: final consumer CE finite and not above both the
                             uniform-logit value and its own first-pass CE (a control capped
                             near uniform by design -- A7, A6 -- is not a divergence)
  G8  degeneracy             the C1b deltas and the block-3 agreements at M* pass
                             `check_degeneracy`
  G9  broken-access headroom A0 under rot90 sits >= HEADROOM_MIN below G3's canonical level;
                             red self-routes `substrate_not_ready_requeue`
  G10 bridgeability          at least one CONSTRAINED rung (L1..L4) of the UNCONDITIONAL
                             `interface_probe.bridge_ladder` (rotated -> native) drives the
                             frozen head to within the band of G3

Per-arm readiness (`precondition_gate`, regime-conditioned, never AND'd whole-run):
`query_not_state_blind`, `head_native_elevation`, `fit_final_ce_below_uniform` (fitted
arms that carry a message), `cond_message_spread` and `cond_reaches_committed_action` (arms
whose conditioning VARIES and that carry a message -- the second is the DV-denominated
form: at least one block-3 row's committed action must change across conditioning levels).
The VERDICT arm's gate at each evaluation width is read by the adjudication: a red A4 gate
(an inert conditioning block, or a non-fitting bridge) is an instrument refusal, never a
science verdict. Other arms' red gates are carried under `per_arm_gate` and mark their
criteria non-degenerate:false without vacating A4.

=== WHAT THIS RUN CANNOT SAY, STATED BEFORE IT RUNS ===

Nothing here licenses a statement about biology. `claim_ids` is DELIBERATELY EMPTY,
following this lineage's own convention (V3-EXQ-1002, V3-EXQ-1010 `BEARS_ON`): the claims
this bears on -- ARC-139, MECH-537/538/540/547/548, INV-105 -- are mostly
`implementation_phase: v4` and are about BIOLOGICAL or ARCHITECTURAL mechanisms that this
synthetic frozen-interface probe does not test with its actual implementation. They are
recorded under `bears_on`, which governance reads as context and not as evidence.
`evidence_direction` is set per branch for a human reader; with no claim tagged it reaches
no confidence score.

Gate A3 (frame x query factorial, compositional holdout) and the `ws250_pca32` replication
are DEFERRED to a successor conditional on this run, which is the spec's own staging.

=== KNOWN OPEN SUBSTRATE DEFECTS OVERLAPPING THIS DRIVER (skill Step 2.5c) ===

No OPEN `corrupting` substrate_queue entry overlaps this driver's module footprint (no
REEAgent is constructed; the e3_selector / tonic_vigor / blocked_agency / entropy-floor paths
are never reached). Three OPEN `degrading` entries overlap and are recorded, not blocking:

  mech357-freeze-incompatible-pressure-mechanism  ree_core/environment/causal_grid_world.py
  SD-MECH303-THRESHOLD-SOURCING                   ree_core/environment/causal_grid_world.py
  dv-dynamic-range-precondition-class             experiments/_metrics.py::p0_readiness_gate

=== GOV-REUSE-1 (Step 2.4) ===

Decisive readout: held-out consumer-use agreement of a FROZEN consumer driven through a
CONDITIONAL bandwidth-limited bridge under a broken presentation frame, as a function of
bottleneck width. Checked V3-EXQ-1002 / 1008 / 1010 manifests (the only runs on this
substrate_hash family carrying agreement readouts): all three score an externally-fitted
DECODER's own argmax (rung 2), none drives a frozen consumer through any bridge, none has a
conditional arm or a bandwidth axis. Not recorded, not derivable -> run.

=== RED-TEAM RECORD (Step 4.5) ===

Pass 1 (opus, on the first draft): BLOCKING -- C1 unattributable (fit objective independent
of receiver state; frame bit handed to A3 only), plus C6 structurally failing, C5 measuring
head difficulty, C4 built on A5, G5 compared to the wrong trivial predictor, per-arm gates
never reaching the adjudication. Every finding is dispositioned above; the causal chain
(objective, budget axis, criteria) changed, so a second pass was spawned per the skill's
one-re-spawn rule.

Pass 2 (opus, on the redesign): CONTESTED, eight findings, each verified against the source
and dispositioned as follows (all FIXED unless stated):
  F1 blocks confounded with the episode driver (fit all-oracle, test all-random) -> blocks
     stratified by driver, asserted in-run.
  F2 C1a an ordinal crossing without variability discipline, one A4 draw vs a minimum over
     five rivals; init seeds correlated across widths -> C1a is the paired section 1.6
     contrast at M_A4; ordinal saving recorded as a diagnostic; init seed includes width.
  F3 best-rival identity chosen on block 3 -> chosen on block 2, scored on block 3.
  F4 C7 survivor-denominated on 3-10 episodes against a mismatched bar -> denominated on
     every started episode, alive fraction recorded, closed-loop q0 bar on the policy's own
     trajectory, 30 rollout episodes. Evaluation at M_eval kept, stated.
  F5 C4 (head-delivery permutation) cannot fail -> C4 is now A5 minus the best unconditional
     arm; the permutation kept as a diagnostic.
  F6 rivals non-functional everywhere refused as an instrument failure -> refuse only when
     neither A4 nor any rival is functional; otherwise M* falls back to M_A4.
  F7 C2 / C5 structurally-expected passes serialised load-bearing -> load_bearing False.
  F8 G4 bit-identical to G3 -> inverse realised as a route module through the fitted-arm path.
  Note (COND_SPREAD_FLOOR is message-space) -> added `cond_reaches_committed_action`.
  Note (C1b's 2SD rule near-unreachable at this fit noise) -> the spec's own rule (1.6);
     kept; the C1b FAIL branch is reported with its measured SD and the restart values.
  Note (A6 non-functional on seed 42) -> a measured property of one seed; recorded.
The second-pass model and verdict are also in the queue entry note. Per the skill, the pass
is not iterated to CLEAR.

Post-fix full-scale confirmers (--scratch-out, ~675 s/seed on DLAPTOP, every gate G1-G10
green, no arm gate red, no fit diverged): seed 42 M*=3 (A5 best, 0.690) M_A4=4, C1b delta
-0.165, C1a delta +0.014, state permutation costs 0.069, A5 minus unconditional -0.161;
seed 43 M*=M_A4=3 (A5 best), C1b = C1a delta +0.020, state permutation costs 0.015. Both
seeds self-route `conditional_access_earns_nothing_at_any_bandwidth`; A6 non-functional at
every width on both; A7 0.29-0.38 throughout; C7 fails on both (A4 closed-loop step-20
agreement 0.0-0.47, foraging 1.5-4.8 vs native 12.6). Recorded here as authoring evidence,
not as the run.
"""

from __future__ import annotations

import argparse
import math as _math
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from experiment_protocol import emit_outcome  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402
from experiments._metrics import check_degeneracy  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell, reset_all_rng  # noqa: E402
from experiments._lib.capability_eval import (  # noqa: E402
    LocalViewGreedyPolicy,
    RandomPolicy,
    evaluate_seed,
)
from experiments._lib.precondition_gate import (  # noqa: E402
    PreconditionSpec,
    aggregate_arm_gates,
    assert_no_structurally_unsatisfiable_gate,
    evaluate_arm_gate,
)
from experiments._lib import interface_probe as iprobe  # noqa: E402
from experiments._lib.interface_probe import BridgeLevel  # noqa: E402

# IMPORTED, NEVER REDEFINED -- the dataset recipe, the standardiser, the capacity ladder and
# the env rung are 1002's / 1010's / 734's own, so this run's anchors are reproductions of
# theirs rather than transcriptions.
import experiments.v3_exq_1002_zworld_actor_adequacy_oracle_adapter as x1002  # noqa: E402
import experiments.v3_exq_1010_zworld_overcapacity_decoder_sweep as x1010  # noqa: E402
import experiments.v3_exq_734_env_difficulty_competence_recovery_sweep as x734  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_1044_hippocampal_assay_a_access_mechanism"
QUEUE_ID = "V3-EXQ-1044"
EXPERIMENT_PURPOSE = "diagnostic"
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"
SLEEP_DRIVER_PATTERN = "none"

# claim_ids DELIBERATELY EMPTY -- see WHAT THIS RUN CANNOT SAY.
CLAIM_IDS: List[str] = []
BEARS_ON = ["ARC-139", "MECH-537", "MECH-538", "MECH-540", "MECH-547", "MECH-548", "INV-105"]

DEVICE = x1002.DEVICE

# ---------------------------------------------------------------------------------------
# ENV / DATA -- inherited from 1002, unchanged
# ---------------------------------------------------------------------------------------
RUNG = x1002.RUNG                                   # D3_hazard_free
RUNG_ID = x1002.RUNG_ID
STEPS_PER_EPISODE = x1002.STEPS_PER_EPISODE         # 200
RESOURCE_FIELD_DIM = x1002.RESOURCE_FIELD_DIM       # 25
BC_EPISODES = x1002.BC_EPISODES                     # 40 oracle-driven
BC_RANDOM_EPISODES = x1002.BC_RANDOM_EPISODES       # 20 random-driven
TOTAL_EPISODES = BC_EPISODES + BC_RANDOM_EPISODES   # 60 -- the episodes_per_run denominator
CONSUMER_RUNG = x1010.CONSUMER_RUNG                 # "mlp128" == x734.PPO_TRUNK_HIDDEN
MAX_CAPACITY_RUNG = x1010.MAX_CAPACITY_RUNG         # "deep2048x4" -- G6's witness rung
GRAD_CLIP_NORM = x1010.GRAD_CLIP_NORM               # 1.0
ADAPTER_LR = x1002.ADAPTER_LR                       # 1e-3
ADAPTER_BATCH = x1002.ADAPTER_BATCH                 # 256

SEEDS = [42, 43, 45]
# Seed 44 is DELIBERATELY ABSENT and replaced by 45: recurring per-seed instability on a
# reef-config env (early death ~step 40; EXQ-539-540, V3-EXQ-538a) makes it a known bad
# seed for this family.
SEED_MAJORITY = x1002.SEED_MAJORITY                 # 2 of 3

# ---------------------------------------------------------------------------------------
# PRE-REGISTERED BARS -- inherited (section 1.6), never re-derived here
# ---------------------------------------------------------------------------------------
AGREEMENT_BAR = x1002.AGREEMENT_BAR                       # 0.80
AGREEMENT_ELEVATION_MIN = x1002.AGREEMENT_ELEVATION_MIN   # 0.20
MEMORISE_FLOOR = x1010.MEMORISE_FLOOR                     # 0.95
# section 1.6 difference bar: absolute floor AND 2x SD of the paired delta.
DELTA_FLOOR = 0.05
DELTA_SD_MULTIPLE = 2.0
# section 1.6 equivalence band -- same magnitude as the positivity floor.
EQUIVALENCE_BAND = 0.05
# G9: the rotation must cost at least this much consumer-use agreement.
HEADROOM_MIN = 0.15
# G2: the oracle must out-forage a random policy by at least this much on this rung.
COMPETENCE_RANGE_MIN = 5.0
# A query answerable to within the elevation margin of the bar by a CONSTANT is vacuous.
# Derived from the two pre-registered constants; no new number.
STATE_BLIND_CEILING = AGREEMENT_BAR - AGREEMENT_ELEVATION_MIN   # 0.60
# Cross-condition spread floor for the conditioned arms' produced message (a RANGE).
COND_SPREAD_FLOOR = 1.0e-3
# C5 message-invariance tolerance (float noise only; the range is exactly 0 by construction).
INVARIANCE_TOL = 1.0e-6
# MECH-548 repeated-use horizon (section 1.8): the step index read by C7.
N_REPEATED_USE = 20

# ---------------------------------------------------------------------------------------
# HEADS / QUERIES / FRAMES
# ---------------------------------------------------------------------------------------
K_HEADS = 3
K_QUERIES = 3
QUERY_IDS = ["q0_approach", "q1_retreat", "q2_orthogonal"]
FRAME_CANONICAL = "f0_canonical"
FRAME_ROT90 = "f1_rot90"
FRAME_IDS = [FRAME_CANONICAL, FRAME_ROT90]
N_FRAME_LEVELS = len(FRAME_IDS)
# Gate A1 presents ONE frame on every row: the rotation that breaks native access.
GATE_A1_FRAME = FRAME_ROT90
BROKEN_FRAME = FRAME_ROT90

COND_DIM = K_QUERIES + K_HEADS + N_FRAME_LEVELS      # 3 + 3 + 2 = 8
COND_SLICE_STATE = (0, K_QUERIES)
COND_SLICE_IDENT = (K_QUERIES, K_QUERIES + K_HEADS)
COND_SLICE_FRAME = (K_QUERIES + K_HEADS, COND_DIM)

HEAD_IN_DIM = RESOURCE_FIELD_DIM + K_QUERIES
# 200 head passes: measured at authoring time (seed 42) the query-conditioned head reaches
# canonical agreement 0.677 at 60 passes and 0.765 at 200, CE still falling.
HEAD_PASSES = 200
BRIDGE_PASSES = 60
BRIDGE_WEIGHT_DECAY = 1.0e-2
K_SELECTOR = 8                         # A6's discrete message cardinality
KMEANS_ITERS = 25
# G10's L4 rung width (the UNCONDITIONAL bridgeability witness; not the swept budget).
LADDER_L4_BOTTLENECK = 64
# THE BUDGET AXIS. Fine at the low end because C1a is ordinal on this grid.
M_GRID = [1, 2, 3, 4, 5, 6, 8, 12, 16, 25]
# Every (arm, M) cell is fitted N_RESTARTS times from different initialisations and the
# restart with the best BLOCK-2 (selection) agreement is the cell; block 3 is read once,
# for that restart only. Applied identically to every fitted arm (section 1.3). Measured
# need: at 60 passes a single unconditional fit is non-monotone in M (seed 42 confirmer:
# A1 0.675 at M=4, 0.395 at M=5, 0.377 at M=6, 0.713 at M=8), so a one-shot m_suff would
# be an optimisation artefact. All restarts' block-2/3 values are recorded per cell.
N_RESTARTS = 3
ROLLOUT_EPISODES = 30

BLOCK_FRACS = (0.35, 0.30, 0.15, 0.20)  # block0 consumer / block1 fit / block2 select / block3 test

# ---------------------------------------------------------------------------------------
# ARMS
# ---------------------------------------------------------------------------------------
ARM_A0 = "A0_native"
ARM_A1 = "A1_source_only"
ARM_A2 = "A2_index_route"
ARM_A3 = "A3_frame_cond"
ARM_A4 = "A4_receiver_state_cond"
ARM_A5 = "A5_receiver_ident_cond"
ARM_A6 = "A6_selector_k"
ARM_A7 = "A7_receiver_only"
GATE_A1_ARMS = [ARM_A0, ARM_A1, ARM_A2, ARM_A3, ARM_A4, ARM_A5, ARM_A6, ARM_A7]
SWEEP_ARMS = [ARM_A1, ARM_A2, ARM_A3, ARM_A4, ARM_A5, ARM_A6, ARM_A7]   # A0 has no width
RIVAL_ARMS = [ARM_A1, ARM_A2, ARM_A3, ARM_A5, ARM_A6]
VERDICT_ARM = ARM_A4
BRIDGE_ARMS = [ARM_A1, ARM_A3, ARM_A4, ARM_A5, ARM_A6, ARM_A7]           # tanh bridges
FITTED_ARMS = BRIDGE_ARMS + [ARM_A2]                                      # A2 fits routes
# Arms whose conditioning input actually VARIES at Gate A1 (A3's is a constant at one frame).
VARYING_COND_ARMS = [ARM_A4, ARM_A5, ARM_A6, ARM_A7]

# Gate A2 controls (scored separately, never in the best-rival set); all built on A4.
CTRL_STATE_PERM = "A2c_state_permuted"
CTRL_HEAD_PERM = "A2c_ident_permuted"
CTRL_WRONG_EP = "A2c_wrong_episode"
CTRL_ZERO_MSG = "A2c_zero_message"
CTRL_RANDOM_MSG = "A2c_moment_matched_random"
GATE_A2_CONTROLS = [CTRL_STATE_PERM, CTRL_HEAD_PERM, CTRL_WRONG_EP, CTRL_ZERO_MSG,
                    CTRL_RANDOM_MSG]

ARM_IDS = GATE_A1_ARMS + GATE_A2_CONTROLS

DRY_RUN_SEEDS = [42]
DRY_RUN_BC_EPISODES = 6
DRY_RUN_BC_RANDOM_EPISODES = 4
DRY_RUN_STEPS = 20
DRY_RUN_HEAD_PASSES = 3
DRY_RUN_BRIDGE_PASSES = 3
DRY_RUN_REPEATED_USE = 3
DRY_RUN_COMPETENCE_EPISODES = 2
DRY_RUN_ROLLOUT_EPISODES = 2
DRY_RUN_M_GRID = [2, 4, 25]
DRY_RUN_RESTARTS = 1

UNIFORM_LOGIT_CE = float(_math.log(5.0))

_EXTRA_SUBSTRATE_PATHS = [Path(m.__file__) for m in (x1002, x1010, x734, iprobe)]


# ---------------------------------------------------------------------------------------
# PRECONDITIONS (regime-conditioned -- never AND'd whole-run)
# ---------------------------------------------------------------------------------------
def _arm_ctx(arm_id: str) -> Dict[str, Any]:
    base = arm_id
    if arm_id in (CTRL_STATE_PERM, CTRL_HEAD_PERM, CTRL_WRONG_EP):
        base = ARM_A4
    return {
        "arm_id": arm_id,
        "is_fitted": base in FITTED_ARMS,
        "conditioning_varies": base in VARYING_COND_ARMS,
        "is_floor_reference": arm_id == ARM_A0,
        "carries_message": arm_id not in (ARM_A7, CTRL_ZERO_MSG),
    }


def _arm_contexts() -> List[Dict[str, Any]]:
    return [_arm_ctx(a) for a in ARM_IDS]


PRECONDITION_SPECS = [
    PreconditionSpec(
        name="query_not_state_blind",
        description=("worst query's MAJORITY-CLASS share of its own oracle labels on the "
                     "block-3 rows -- what a state-blind constant predictor scores on it"),
        control=("STATE_BLIND_CEILING = AGREEMENT_BAR - AGREEMENT_ELEVATION_MIN, derived "
                 "from the pre-registered constants: a query a constant answers to within "
                 "the elevation margin of the bar is vacuous for every arm alike"),
        threshold=STATE_BLIND_CEILING,
        direction="upper",
        kind="readiness",
    ),
    PreconditionSpec(
        name="head_native_elevation",
        description=("the frozen heads' block-3 consumer-use agreement on the CANONICAL "
                     "field MINUS the strongest trivial predictor -- the G3 consumer floor"),
        control=("frozen heads + canonical field (a known-good positive control), elevated "
                 "over the strongest of {previous executed action, per-query constant "
                 "predictor}. The ELEVATION form: AGREEMENT_BAR was calibrated on a "
                 "single-task decoder; this is a three-task query-conditioned head."),
        threshold=AGREEMENT_ELEVATION_MIN,
        direction="lower",
        kind="readiness",
    ),
    PreconditionSpec(
        name="fit_final_ce_below_uniform",
        description="the arm's end-to-end fit reached a consumer CE below the uniform value",
        control="log(action_dim) -- a fit at or above it has not fitted at all",
        threshold=UNIFORM_LOGIT_CE,
        direction="upper",
        kind="readiness",
        applies_to=lambda ctx: bool(ctx["is_fitted"] and ctx["carries_message"]),
        applies_note=("only a FITTED arm that CARRIES a message can be expected to fit below "
                      "uniform; the zero-message control A7 is capped at the per-query label "
                      "entropy BY DESIGN and is covered by G7's divergence test instead"),
    ),
    PreconditionSpec(
        name="cond_message_spread",
        description=("CROSS-CONDITION RANGE (max-min over the conditioning levels this arm "
                     "may see, worst row) of the message this arm produces -- the "
                     "V3-EXQ-643 same-statistic guard: a RANGE, never a magnitude"),
        control="the same block-3 rows re-presented under every conditioning level",
        threshold=COND_SPREAD_FLOOR,
        direction="lower",
        kind="readiness",
        applies_to=lambda ctx: bool(ctx["conditioning_varies"] and ctx["carries_message"]),
        applies_note=("not meaningful for an arm whose conditioning is constant at Gate A1 "
                      "(A1; A3 at a single frame) nor for a zeroed-message arm"),
    ),
    PreconditionSpec(
        name="cond_reaches_committed_action",
        description=("fraction of block-3 rows whose FROZEN-HEAD COMMITTED ACTION changes when "
                     "the arm's conditioning level is overridden -- the DV-denominated form of "
                     "cond_message_spread (red-team pass 2): a message-space range says "
                     "nothing about whether the conditioning can move the argmax"),
        control=("threshold = 1 / n_rows: at least ONE row's committed action must change "
                 "across the conditioning levels, i.e. the conditioning reaches the DV at all"),
        threshold=1.0e-9,
        direction="lower",
        kind="readiness",
        applies_to=lambda ctx: bool(ctx["conditioning_varies"] and ctx["carries_message"]),
        applies_note="same scope as cond_message_spread",
    ),
]


# ---------------------------------------------------------------------------------------
# FRAME TRANSFORMS -- an exact, invertible coordinate permutation + its action relabelling
# ---------------------------------------------------------------------------------------
_DELTAS = dict(LocalViewGreedyPolicy._DELTAS)   # {0:(-1,0),1:(1,0),2:(0,-1),3:(0,1),4:(0,0)}
_DELTA_TO_ACTION = {v: k for k, v in _DELTAS.items()}


def _rot90_field_permutation() -> List[int]:
    """Index permutation of the flattened 5x5 local field under (dx, dy) -> (-dy, dx).

    x1008 fixes the layout: view index for offset (dx, dy) is (2 + dx) * 5 + (2 + dy).
    Under the rotation, row r = 2 + dx and column c = 2 + dy map to r' = 4 - c, c' = r.
    """
    perm = [0] * (RESOURCE_FIELD_DIM)
    for r in range(5):
        for c in range(5):
            src = r * 5 + c
            dst = (4 - c) * 5 + r
            perm[dst] = src
    return perm


ROT90_PERM = _rot90_field_permutation()
ROT90_INV_PERM = [0] * RESOURCE_FIELD_DIM
for _i, _s in enumerate(ROT90_PERM):
    ROT90_INV_PERM[_s] = _i


def _rot90_action_map() -> Dict[int, int]:
    out = {}
    for a, (dx, dy) in _DELTAS.items():
        out[a] = _DELTA_TO_ACTION[(-dy, dx)]
    return out


ROT90_ACTION = _rot90_action_map()


def _apply_frame(x: torch.Tensor, frame: str) -> torch.Tensor:
    if frame == FRAME_CANONICAL:
        return x
    if frame == FRAME_ROT90:
        return x[:, ROT90_PERM]
    raise KeyError("unknown frame: %s" % frame)


def _invert_frame(x: torch.Tensor, frame: str) -> torch.Tensor:
    if frame == FRAME_CANONICAL:
        return x
    if frame == FRAME_ROT90:
        return x[:, ROT90_INV_PERM]
    raise KeyError("unknown frame: %s" % frame)


def _present(rows: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Apply each row's OWN frame to that row's field -- the Gate A1 presentation."""
    x = rows["field"]
    out = x.clone()
    for fi, fname in enumerate(FRAME_IDS):
        sel = (rows["frame"] == fi)
        if bool(sel.any()):
            out[sel] = _apply_frame(x[sel], fname)
    return out


def _unpresent(msg: torch.Tensor, rows: Dict[str, torch.Tensor]) -> torch.Tensor:
    """The exact analytic inverse of `_present`, row by row (G4's control)."""
    out = msg.clone()
    for fi, fname in enumerate(FRAME_IDS):
        sel = (rows["frame"] == fi)
        if bool(sel.any()):
            out[sel] = _invert_frame(msg[sel], fname)
    return out


# ---------------------------------------------------------------------------------------
# QUERIES -- three non-degenerate oracle label functions over the SAME local field
# ---------------------------------------------------------------------------------------
def _nearest_resource_offset(field: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per row, the (dx, dy) offset of the highest-valued cell in the 5x5 local field."""
    idx = torch.argmax(field, dim=-1)
    r = torch.div(idx, 5, rounding_mode="floor")
    c = idx % 5
    return (r - 2), (c - 2)


def _sign(v: int) -> int:
    return (v > 0) - (v < 0)


def _query_labels(field: torch.Tensor, query: int) -> torch.Tensor:
    """Oracle action for `query` at every row of `field` (the CANONICAL-frame field).

    q0 approach   -- move along the dominant axis TOWARD the nearest resource.
    q1 retreat    -- the sign-flipped q0.
    q2 orthogonal -- move along the axis ORTHOGONAL to the nearest-resource direction.
    """
    dx, dy = _nearest_resource_offset(field)
    n = field.shape[0]
    out = torch.full((n,), 4, dtype=torch.long)
    adx = dx.abs()
    ady = dy.abs()
    for i in range(n):
        ix, iy = int(dx[i]), int(dy[i])
        if ix == 0 and iy == 0:
            out[i] = 4
            continue
        if query == 0:
            step = (int(_sign(ix)), 0) if int(adx[i]) >= int(ady[i]) else (0, int(_sign(iy)))
        elif query == 1:
            step = (-int(_sign(ix)), 0) if int(adx[i]) >= int(ady[i]) else (0, -int(_sign(iy)))
        elif query == 2:
            if int(adx[i]) >= int(ady[i]):
                s = int(_sign(iy)) or 1
                step = (0, s)
            else:
                s = int(_sign(ix)) or 1
                step = (s, 0)
        else:
            raise KeyError("unknown query index: %d" % query)
        out[i] = _DELTA_TO_ACTION[step]
    return out


def _class_shares(y: torch.Tensor, n_classes: int = 5) -> List[float]:
    if int(y.shape[0]) == 0:
        return [0.0] * n_classes
    counts = torch.bincount(y, minlength=n_classes).to(torch.float64)
    return [float(v) for v in (counts / counts.sum()).tolist()]


def _majority_class_frac(y: torch.Tensor, n_classes: int = 5) -> float:
    """Share of the MOST-represented class -- what a state-blind constant predictor scores."""
    if int(y.shape[0]) == 0:
        return 1.0
    return max(_class_shares(y, n_classes))


# ---------------------------------------------------------------------------------------
# FROZEN CONSUMER HEADS
# ---------------------------------------------------------------------------------------
def _make_head(seed_offset: int):
    """One frozen consumer head: x734.PPOPolicyNet ITSELF at the consumer rung, built
    through x1010._make_decoder so the capacity is the lineage's own construction site."""
    torch.manual_seed(1_000_003 + int(seed_offset))
    return x1010._make_decoder(CONSUMER_RUNG, HEAD_IN_DIM, 5)


def _heads_logits(heads: List[Any], message: torch.Tensor, q_oh: torch.Tensor,
                  hidx: torch.Tensor) -> torch.Tensor:
    """Each row through ITS OWN head. Differentiable w.r.t. `message` (heads are frozen)."""
    n = int(message.shape[0])
    out = torch.zeros(n, 5)
    for h in range(K_HEADS):
        sel = (hidx == h)
        if bool(sel.any()):
            logits, _v = heads[h](torch.cat([message[sel], q_oh[sel]], dim=-1))
            out = out.index_put((torch.nonzero(sel, as_tuple=False).reshape(-1),), logits)
    return out


def _train_head(head, x: torch.Tensor, q: torch.Tensor, y: torch.Tensor, passes: int,
                seed: int, label: str) -> Dict[str, Any]:
    """Behaviour-clone the query-conditioned oracle onto one head, then FREEZE it."""
    opt = torch.optim.Adam(head.parameters(), lr=ADAPTER_LR)
    lossfn = nn.CrossEntropyLoss()
    n = int(x.shape[0])
    g = torch.Generator().manual_seed(int(seed) + 7717)
    inp = torch.cat([x, q], dim=-1)
    losses: List[float] = []
    for p in range(int(passes)):
        order = torch.randperm(n, generator=g)
        tot, nb = 0.0, 0
        for s in range(0, n, ADAPTER_BATCH):
            idx = order[s:s + ADAPTER_BATCH]
            logits, _v = head(inp[idx])
            loss = lossfn(logits, y[idx])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(), GRAD_CLIP_NORM)
            opt.step()
            tot += float(loss.item())
            nb += 1
        losses.append(tot / max(1, nb))
        if (p + 1) % max(1, passes // 3) == 0:
            print("  [head] %s seed=%d pass %d of %d ce=%.4f"
                  % (label, seed, p + 1, int(passes), losses[-1]), flush=True)
    for prm in head.parameters():
        prm.requires_grad_(False)
    head.eval()
    return {"first_ce": (losses[0] if losses else None),
            "final_ce": (losses[-1] if losses else None),
            "n_passes": int(passes)}


# ---------------------------------------------------------------------------------------
# THE BRIDGE -- one architecture for every fitted arm (section 1.3 budget match)
# ---------------------------------------------------------------------------------------
class _ConditionalBridge(nn.Module):
    """[message_in (25) ; conditioning (COND_DIM)] -> tanh(M) -> message_out (25).

    The spec's L4 rung. The hidden width M IS the bandwidth budget. Identical for every
    bridge arm at a given M: identical input width, identical parameter count.
    """

    def __init__(self, hidden: int):
        super().__init__()
        self.hidden = int(hidden)
        self.net = nn.Sequential(
            nn.Linear(RESOURCE_FIELD_DIM + COND_DIM, int(hidden)),
            nn.Tanh(),
            nn.Linear(int(hidden), RESOURCE_FIELD_DIM),
        )

    def forward(self, msg: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([msg, cond], dim=-1))


class _LinearRoute(nn.Module):
    """A2's fixed per-identity LINEAR route of rank M: 25 -> M -> 25, no nonlinearity."""

    def __init__(self, hidden: int):
        super().__init__()
        self.hidden = int(hidden)
        self.down = nn.Linear(RESOURCE_FIELD_DIM, int(hidden), bias=False)
        self.up = nn.Linear(int(hidden), RESOURCE_FIELD_DIM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(x))


def _make_bridge(seed: int, hidden: int) -> _ConditionalBridge:
    torch.manual_seed(2_000_003 + int(seed))
    return _ConditionalBridge(hidden)


def _make_routes(seed: int, hidden: int) -> List[_LinearRoute]:
    torch.manual_seed(3_000_003 + int(seed))
    return [_LinearRoute(hidden) for _ in range(K_HEADS)]


def _n_params(modules: List[nn.Module]) -> int:
    return int(sum(p.numel() for m in modules for p in m.parameters()))


def _cond_mask(arm_id: str) -> torch.Tensor:
    """Which conditioning sub-blocks this arm may see. Width is ALWAYS COND_DIM."""
    m = torch.zeros(COND_DIM)
    if arm_id in (ARM_A4, ARM_A6, ARM_A7):
        m[COND_SLICE_STATE[0]:COND_SLICE_STATE[1]] = 1.0
    if arm_id in (ARM_A5, ARM_A6, ARM_A7):
        m[COND_SLICE_IDENT[0]:COND_SLICE_IDENT[1]] = 1.0
    if arm_id == ARM_A3:
        m[COND_SLICE_FRAME[0]:COND_SLICE_FRAME[1]] = 1.0
    return m


def _fit_end_to_end(params: List[torch.Tensor], forward: Callable[[torch.Tensor], torch.Tensor],
                    heads: List[Any], rows: Dict[str, torch.Tensor], passes: int,
                    seed: int, label: str) -> Dict[str, Any]:
    """Fit an arm's trainable object END-TO-END through the FROZEN heads.

    `forward(idx)` returns the arm's 25-dim message for the batch rows `idx`; the loss is
    the frozen consumer's cross-entropy on the query-specific oracle action, each row
    through its own head with its own true query. The heads' parameters are frozen
    (requires_grad False); gradient flows to the message and into `params`.

    One optimiser, learning rate, batch, pass count, grad-clip and weight decay for EVERY
    fitted arm -- the section 1.3 optimisation match.
    """
    opt = torch.optim.Adam(params, lr=ADAPTER_LR, weight_decay=BRIDGE_WEIGHT_DECAY)
    lossfn = nn.CrossEntropyLoss()
    n = int(rows["label"].shape[0])
    q_oh = _onehot(rows["query"], K_QUERIES)
    hidx = rows["head"]
    y = rows["label"]
    g = torch.Generator().manual_seed(int(seed) + 4241)
    losses: List[float] = []
    for p in range(int(passes)):
        order = torch.randperm(n, generator=g)
        tot, nb = 0.0, 0
        for s in range(0, n, ADAPTER_BATCH):
            idx = order[s:s + ADAPTER_BATCH]
            out = forward(idx)
            logits = _heads_logits(heads, out, q_oh[idx], hidx[idx])
            loss = lossfn(logits, y[idx])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, GRAD_CLIP_NORM)
            opt.step()
            tot += float(loss.item())
            nb += 1
        losses.append(tot / max(1, nb))
        if (p + 1) % max(1, passes // 3) == 0:
            print("  [fit] %s seed=%d pass %d of %d ce=%.4f"
                  % (label, seed, p + 1, int(passes), losses[-1]), flush=True)
    return {"first_ce": (losses[0] if losses else None),
            "final_ce": (losses[-1] if losses else None),
            "n_passes": int(passes)}


# ---------------------------------------------------------------------------------------
# DATA
# ---------------------------------------------------------------------------------------
def _collect(seed: int, env_kwargs: Dict[str, Any], n_oracle: int, n_random: int,
             steps: int) -> List[Dict[str, Any]]:
    """Episode collection, with the runner's progress denominator on the EPISODE axis."""
    total = int(n_oracle) + int(n_random)
    eps: List[Dict[str, Any]] = []
    done_ct = 0
    for driver, count in (("oracle", int(n_oracle)), ("random", int(n_random))):
        for k in range(count):
            batch = x1002._collect_episodes(seed * 100 + done_ct, env_kwargs, driver, 1, steps)
            for ep in batch:
                ep["driver"] = driver
            eps.extend(batch)
            done_ct += 1
            if done_ct % max(1, total // 6) == 0 or done_ct == total:
                print("  [train] collect seed=%d ep %d/%d" % (seed, done_ct, total), flush=True)
    return eps


def _split_list(items: List[Any]) -> List[List[Any]]:
    n = len(items)
    cuts = []
    acc = 0
    for f in BLOCK_FRACS[:-1]:
        acc += max(1, int(round(n * f)))
        cuts.append(min(acc, n))
    return [items[:cuts[0]], items[cuts[0]:cuts[1]], items[cuts[1]:cuts[2]], items[cuts[2]:]]


def _blocks(episodes: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Four-way split BY EPISODE (section 1.5 level 1), STRATIFIED BY DRIVER.

    `_collect` stores every oracle-driven episode before every random-driven one, so a
    contiguous cut would put all-oracle episodes in the fitting block and all-random ones in
    the test block (red-team pass 2, F1): every criterion would then measure transfer across a
    visitation shift, not bandwidth. Each driver's episodes are split by BLOCK_FRACS
    separately and the per-block lists are merged, so every block carries both visitation
    distributions in the same proportion. Asserted, not assumed.
    """
    names = ["block0_consumer", "block1_fit", "block2_select", "block3_test"]
    out: Dict[str, List[Dict[str, Any]]] = {k: [] for k in names}
    drivers = sorted(set(str(ep.get("driver", "unknown")) for ep in episodes))
    for drv in drivers:
        parts = _split_list([ep for ep in episodes if str(ep.get("driver", "unknown")) == drv])
        for k, part in zip(names, parts):
            out[k].extend(part)
    total = sum(len(v) for v in out.values())
    assert total == len(episodes), "block split lost episodes: %d != %d" % (total, len(episodes))
    if len(drivers) > 1:
        for k in names:
            present = set(str(ep.get("driver")) for ep in out[k])
            assert present == set(drivers), (
                "block %s carries drivers %r, not every driver %r -- the split is confounded "
                "with the visitation distribution" % (k, sorted(present), drivers))
    return out


def _rows(episodes: List[Dict[str, Any]], seed: int, query_cycle: bool
          ) -> Dict[str, torch.Tensor]:
    """Feature rows for a block, with query and head assignment.

    `query_cycle` True assigns queries and head identities deterministically by ROW INDEX
    (query cycles fastest, identity next), so every arm sees the identical assignment on the
    identical rows (section 1.3 data match), the assignment is independent of content, and
    every (query, identity) cell is equally populated. The frame is the Gate A1 frame on
    every row.
    """
    x, _y_oracle = x1002._rawfield_features(episodes)
    n = int(x.shape[0])
    fr_idx = FRAME_IDS.index(GATE_A1_FRAME)
    if n == 0:
        z = torch.zeros(0, RESOURCE_FIELD_DIM)
        return {"field": z, "query": torch.zeros(0, dtype=torch.long),
                "head": torch.zeros(0, dtype=torch.long),
                "frame": torch.zeros(0, dtype=torch.long),
                "label": torch.zeros(0, dtype=torch.long),
                "prev_action": torch.zeros(0, dtype=torch.long),
                "episode": torch.zeros(0, dtype=torch.long)}
    if query_cycle:
        q = torch.arange(n) % K_QUERIES
        h = (torch.arange(n) // K_QUERIES) % K_HEADS
    else:
        q = torch.zeros(n, dtype=torch.long)
        h = torch.zeros(n, dtype=torch.long)
    fr = torch.full((n,), fr_idx, dtype=torch.long)
    labels = torch.zeros(n, dtype=torch.long)
    for qi in range(K_QUERIES):
        sel = (q == qi)
        if bool(sel.any()):
            labels[sel] = _query_labels(x[sel], qi)
    ep_idx: List[int] = []
    for j, ep in enumerate(episodes):
        ep_idx.extend([j] * len(ep["labels"]))
    return {
        "field": x,
        "query": q.to(torch.long),
        "head": h.to(torch.long),
        "frame": fr,
        "label": labels,
        "prev_action": x1002._prev_action_vector(episodes),
        "episode": torch.tensor(ep_idx[:n], dtype=torch.long),
    }


def _onehot(idx: torch.Tensor, k: int) -> torch.Tensor:
    out = torch.zeros(int(idx.shape[0]), int(k))
    if int(idx.shape[0]):
        out[torch.arange(int(idx.shape[0])), idx.to(torch.long)] = 1.0
    return out


def _cond_vector(rows: Dict[str, torch.Tensor], arm_id: str,
                 state_override: Optional[torch.Tensor] = None,
                 ident_override: Optional[torch.Tensor] = None) -> torch.Tensor:
    """The FULL-WIDTH conditioning vector, with this arm's forbidden blocks zeroed.

    Width is COND_DIM for every arm, always -- an unconditional arm receives a constant
    zero vector of the same width, so arms differ in INFORMATION and not dimensionality.
    """
    state = rows["query"] if state_override is None else state_override
    ident = rows["head"] if ident_override is None else ident_override
    full = torch.cat([_onehot(state, K_QUERIES), _onehot(ident, K_HEADS),
                      _onehot(rows["frame"], N_FRAME_LEVELS)], dim=-1)
    return full * _cond_mask(arm_id).unsqueeze(0)


# ---------------------------------------------------------------------------------------
# MESSAGE CONSTRUCTION HELPERS
# ---------------------------------------------------------------------------------------
def _moment_matched_random(x: torch.Tensor, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(int(seed) + 9091)
    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, keepdim=True).clamp_min(1e-8)
    return mean + std * torch.randn(x.shape, generator=g)


def _fit_kmeans(x: torch.Tensor, k: int, seed: int, iters: int = KMEANS_ITERS) -> torch.Tensor:
    """Lloyd's k-means on the fitting block's presented sources: A6's quantiser."""
    g = torch.Generator().manual_seed(int(seed) + 3313)
    n = int(x.shape[0])
    if n == 0:
        return torch.zeros(k, x.shape[1])
    cent = x[torch.randperm(n, generator=g)[:k]].clone()
    if int(cent.shape[0]) < k:
        cent = torch.cat([cent, cent[:1].repeat(k - int(cent.shape[0]), 1)], dim=0)
    for _ in range(int(iters)):
        idx = torch.argmin(torch.cdist(x, cent), dim=-1)
        for j in range(k):
            sel = (idx == j)
            if bool(sel.any()):
                cent[j] = x[sel].mean(dim=0)
    return cent


def _selector_codes(seed: int) -> torch.Tensor:
    """A6's fixed random 25-dim code per index (distinct codes: pairwise distance > 0)."""
    g = torch.Generator().manual_seed(int(seed) + 3314)
    return torch.randn(K_SELECTOR, RESOURCE_FIELD_DIM, generator=g)


def _selector_message(msg: torch.Tensor, centroids: torch.Tensor,
                      codes: torch.Tensor) -> torch.Tensor:
    idx = torch.argmin(torch.cdist(msg, centroids), dim=-1)
    return codes[idx]


def _nn_retrieve(query_msg: torch.Tensor, store_key: torch.Tensor, store_val: torch.Tensor,
                 query_ep: Optional[torch.Tensor] = None,
                 store_ep: Optional[torch.Tensor] = None) -> torch.Tensor:
    """A2: CONTENT-ADDRESSED nearest-neighbour reinstatement over the fitting block.

    With `query_ep`/`store_ep` given, a query row may not retrieve from its OWN episode
    (leave-one-episode-out), so the fitting-time retrieval mirrors the held-out situation
    where the nearest neighbour always comes from a different episode.
    """
    if int(query_msg.shape[0]) == 0:
        return torch.zeros(0, RESOURCE_FIELD_DIM)
    d = torch.cdist(query_msg, store_key)
    if query_ep is not None and store_ep is not None:
        same = (query_ep.reshape(-1, 1) == store_ep.reshape(1, -1))
        d = d.masked_fill(same, float("inf"))
    nn_idx = torch.argmin(d, dim=-1)
    return store_val[nn_idx]


# ---------------------------------------------------------------------------------------
# SCORING
# ---------------------------------------------------------------------------------------
def _trivial_family(fit_rows: Dict[str, torch.Tensor], rows: Dict[str, torch.Tensor]
                    ) -> Dict[str, Any]:
    """The STRONGEST trivial predictor on this three-query task, re-measured per block.

    Two content-free predictors: (1) the previous executed action (1002's strongest, at
    0.57-0.58 on the single approach task; ~0.2 here) and (2) the per-query CONSTANT
    predictor -- the majority class of each query chosen on the FITTING block and predicted
    on every row of that query. The strongest is what every elevation is measured against.
    """
    n = int(rows["label"].shape[0])
    if n == 0:
        return {"prev_action": 0.0, "per_query_constant": 0.0, "strongest": 0.0,
                "strongest_name": "none", "per_query_majority_action": {}}
    prev = float((rows["prev_action"] == rows["label"]).float().mean().item())
    maj: Dict[int, int] = {}
    correct = 0
    for qi in range(K_QUERIES):
        yf = fit_rows["label"][fit_rows["query"] == qi]
        a = int(torch.argmax(torch.bincount(yf, minlength=5)).item()) if int(yf.shape[0]) else 4
        maj[qi] = a
        sel = (rows["query"] == qi)
        correct += int((rows["label"][sel] == a).sum().item())
    sb = float(correct) / float(n)
    strongest = max(prev, sb)
    return {"prev_action": prev, "per_query_constant": sb, "strongest": strongest,
            "strongest_name": ("per_query_constant" if sb >= prev else "prev_action"),
            "per_query_majority_action": {QUERY_IDS[k]: v for k, v in maj.items()}}


def _score_arm(arm_id: str, heads: List[Any], message: torch.Tensor,
               rows: Dict[str, torch.Tensor],
               head_override: Optional[torch.Tensor] = None) -> Dict[str, Any]:
    """Consumer-use agreement: the FROZEN head's committed action against the oracle's.

    Each row is scored by ITS OWN head (or by `head_override` -- the identity-permutation
    control DELIVERS the message to a different head). The head always receives the row's
    TRUE query.
    """
    q_oh = _onehot(rows["query"], K_QUERIES)
    hidx = rows["head"] if head_override is None else head_override
    n = int(message.shape[0])
    if n == 0:
        return {"agreement": None, "per_head": {}, "per_query": {}, "n_rows": 0}
    with torch.no_grad():
        pred = torch.argmax(_heads_logits(heads, message, q_oh, hidx), dim=-1)
    ok = (pred == rows["label"])
    per_head: Dict[str, Optional[float]] = {}
    for h in range(K_HEADS):
        sel = (hidx == h)
        per_head["head_%d" % h] = (float(ok[sel].float().mean().item()) if bool(sel.any())
                                   else None)
    per_query: Dict[str, Optional[float]] = {}
    for qi in range(K_QUERIES):
        sel = (rows["query"] == qi)
        per_query[QUERY_IDS[qi]] = (float(ok[sel].float().mean().item()) if bool(sel.any())
                                    else None)
    return {"agreement": float(ok.float().mean().item()), "per_head": per_head,
            "per_query": per_query, "n_rows": int(n)}


def _paired_delta_verdict(deltas: List[Optional[float]]) -> Dict[str, Any]:
    """The section 1.6 difference rule, applied to the paired per-seed deltas."""
    arr = np.asarray([d for d in deltas if d is not None], dtype=np.float64)
    if arr.size == 0:
        return {"mean_delta": None, "sd_delta": None, "n_seeds": 0,
                "n_seeds_positive": 0, "seeds_required": int(SEED_MAJORITY),
                "passed": False, "measured": None, "threshold": DELTA_FLOOR,
                "sd_gate_measured": None, "sd_gate_threshold": None}
    mean_d = float(arr.mean())
    sd_d = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    n_pos = int((arr >= DELTA_FLOOR).sum())
    sd_bar = DELTA_SD_MULTIPLE * sd_d
    passed = bool(mean_d >= DELTA_FLOOR and mean_d >= sd_bar and n_pos >= SEED_MAJORITY)
    return {"mean_delta": mean_d, "sd_delta": sd_d, "n_seeds": int(arr.size),
            "n_seeds_positive": n_pos, "seeds_required": int(SEED_MAJORITY),
            "passed": passed, "measured": mean_d, "threshold": DELTA_FLOOR,
            "sd_gate_measured": mean_d, "sd_gate_threshold": sd_bar}


# ---------------------------------------------------------------------------------------
# ARM CONSTRUCTION -- one fitted object per (arm, M); message_fn re-presents any rows
# ---------------------------------------------------------------------------------------
class _FittedArm:
    """A fitted access mechanism at width M. `message(rows, presented, ...)` produces the
    25-dim message the frozen head will read, for ANY rows (block 2, block 3, a rollout)."""

    def __init__(self, arm_id: str, width: int, fit_info: Dict[str, Any],
                 capacity: Dict[str, Any], fn: Callable[..., torch.Tensor]):
        self.arm_id = arm_id
        self.width = int(width)
        self.fit_info = fit_info
        self.capacity = capacity
        self._fn = fn

    def message(self, rows: Dict[str, torch.Tensor], presented: torch.Tensor,
                state_override: Optional[torch.Tensor] = None,
                ident_override: Optional[torch.Tensor] = None) -> torch.Tensor:
        with torch.no_grad():
            return self._fn(rows, presented, state_override, ident_override)


def _build_arm(arm_id: str, width: int, seed: int, heads: List[Any],
               r1: Dict[str, torch.Tensor], pres1: torch.Tensor, store: Dict[str, Any],
               passes: int, restart: int = 0) -> _FittedArm:
    """Construct and fit one sweep arm at width M on block 1 (end-to-end, frozen heads).

    `restart` only changes the initialisation seed of the fitted object."""
    label = "%s@M%d#r%d" % (arm_id, width, restart)
    init_seed = seed * 100 + 10_000 * int(restart) + 1_000_000 * int(width)
    if arm_id == ARM_A2:
        routes = _make_routes(init_seed + 2, width)
        retr_fit = _nn_retrieve(pres1, store["key"], store["val"], r1["episode"],
                                store["episode"])
        params = [p for r in routes for p in r.parameters()]

        def _route_apply(retrieved: torch.Tensor, hidx: torch.Tensor) -> torch.Tensor:
            out = torch.zeros_like(retrieved)
            for h in range(K_HEADS):
                sel = (hidx == h)
                if bool(sel.any()):
                    out = out.index_put((torch.nonzero(sel, as_tuple=False).reshape(-1),),
                                        routes[h](retrieved[sel]))
            return out

        fit = _fit_end_to_end(params, lambda idx: _route_apply(retr_fit[idx], r1["head"][idx]),
                              heads, r1, passes, seed, label)
        for r in routes:
            for p in r.parameters():
                p.requires_grad_(False)
            r.eval()

        def _fn(rows, presented, so, io):
            retrieved = _nn_retrieve(presented, store["key"], store["val"])
            return _route_apply(retrieved, rows["head"])

        cap = {"module_class": "LinearRoute x K_HEADS", "rank": int(width),
               "total_params": _n_params(routes)}
        return _FittedArm(arm_id, width, fit, cap, _fn)

    bridge = _make_bridge(init_seed + ARM_IDS.index(arm_id), width)
    if arm_id == ARM_A6:
        fit_in = _selector_message(pres1, store["centroids"], store["codes"])
    elif arm_id == ARM_A7:
        fit_in = torch.zeros_like(pres1)
    else:
        fit_in = pres1
    cond_fit = _cond_vector(r1, arm_id)
    fit = _fit_end_to_end(list(bridge.parameters()),
                          lambda idx: bridge(fit_in[idx], cond_fit[idx]),
                          heads, r1, passes, seed, label)
    for p in bridge.parameters():
        p.requires_grad_(False)
    bridge.eval()

    def _fn(rows, presented, so, io):
        if arm_id == ARM_A6:
            x = _selector_message(presented, store["centroids"], store["codes"])
        elif arm_id == ARM_A7:
            x = torch.zeros_like(presented)
        else:
            x = presented
        return bridge(x, _cond_vector(rows, arm_id, state_override=so, ident_override=io))

    cap = {"module_class": type(bridge).__name__, "in_dim": int(RESOURCE_FIELD_DIM + COND_DIM),
           "cond_dim": int(COND_DIM), "hidden": int(width), "out_dim": int(RESOURCE_FIELD_DIM),
           "total_params": _n_params([bridge])}
    return _FittedArm(arm_id, width, fit, cap, _fn)


def _cross_condition_spread(arm: _FittedArm, rows: Dict[str, torch.Tensor],
                            presented: torch.Tensor, heads: Optional[List[Any]] = None
                            ) -> Tuple[float, float]:
    """(worst-row cross-condition RANGE of the produced message, fraction of rows whose
    committed action changes) over the conditioning levels this arm may see. The range
    catches a uniform per-row offset (the DV-symmetry failure mode); the action-change
    fraction is the DV-denominated reach check (red-team pass 2)."""
    mask = _cond_mask(arm.arm_id)
    n = int(presented.shape[0])
    if n == 0:
        return 0.0, 0.0
    outs: List[torch.Tensor] = []
    if float(mask[COND_SLICE_STATE[0]:COND_SLICE_STATE[1]].sum()) > 0:
        for qi in range(K_QUERIES):
            outs.append(arm.message(rows, presented,
                                    state_override=torch.full((n,), qi, dtype=torch.long)))
    if float(mask[COND_SLICE_IDENT[0]:COND_SLICE_IDENT[1]].sum()) > 0:
        for hi in range(K_HEADS):
            outs.append(arm.message(rows, presented,
                                    ident_override=torch.full((n,), hi, dtype=torch.long)))
    if not outs:
        return 0.0, 0.0
    stack = torch.stack(outs, dim=0)                            # [L, N, 25]
    rng = (stack.max(dim=0).values - stack.min(dim=0).values)   # [N, 25]
    spread = float(rng.max(dim=-1).values.min().item())
    changed = 0.0
    if heads is not None:
        q_oh = _onehot(rows["query"], K_QUERIES)
        with torch.no_grad():
            acts = torch.stack([torch.argmax(_heads_logits(heads, m, q_oh, rows["head"]), dim=-1)
                                for m in outs], dim=0)          # [L, N]
        changed = float(((acts != acts[0:1]).any(dim=0)).float().mean().item())
    return spread, changed


def _state_invariance_range(arm: _FittedArm, rows: Dict[str, torch.Tensor],
                            presented: torch.Tensor) -> float:
    """C5: the cross-STATE range of an arm's message (exactly 0 for an unconditional arm)."""
    n = int(presented.shape[0])
    if n == 0:
        return 0.0
    outs = [arm.message(rows, presented, state_override=torch.full((n,), qi, dtype=torch.long))
            for qi in range(K_QUERIES)]
    stack = torch.stack(outs, dim=0)
    return float((stack.max(dim=0).values - stack.min(dim=0).values).max().item())


def _permute_within(stratum: torch.Tensor, values: torch.Tensor,
                    g: torch.Generator) -> torch.Tensor:
    """Shuffle `values` WITHIN each level of `stratum` (marginals preserved; section 1.5
    guard 6 -- an off-distribution permutation measures distribution sensitivity)."""
    out = values.clone()
    for lv in torch.unique(stratum):
        sel = torch.nonzero(stratum == lv, as_tuple=False).reshape(-1)
        if int(sel.numel()) > 1:
            out[sel] = values[sel[torch.randperm(int(sel.numel()), generator=g)]]
    return out


def _within_cell_row_permutation(rows: Dict[str, torch.Tensor], g: torch.Generator
                                 ) -> torch.Tensor:
    """Row permutation WITHIN each (query, identity) cell (the wrong-episode control)."""
    n = int(rows["query"].shape[0])
    out = torch.arange(n)
    for qi in range(K_QUERIES):
        for hi in range(K_HEADS):
            sel = torch.nonzero((rows["query"] == qi) & (rows["head"] == hi),
                                as_tuple=False).reshape(-1)
            if int(sel.numel()) > 1:
                out[sel] = sel[torch.randperm(int(sel.numel()), generator=g)]
    return out


# ---------------------------------------------------------------------------------------
# CLOSED-LOOP ROLLOUTS (C7 / MECH-548, and rung 7)
# ---------------------------------------------------------------------------------------
def _closed_loop(policy_fn: Callable[[torch.Tensor], int], seed: int,
                 env_kwargs: Dict[str, Any], n_eps: int, steps: int, n_use: int,
                 trivial: float) -> Dict[str, Any]:
    """Drive the env with `policy_fn(canonical field [1,25]) -> action` for whole episodes.

    Per-step agreement with the q0 oracle along the SELF-GENERATED trajectory (the consumer's
    own resulting state is what it feeds forward -- section 1.8), plus foraging competence
    (resources per episode, rung 7).
    """
    # Denominated on EVERY started episode (red-team pass 2, F4): an episode the arm's own
    # policy let die contributes a non-agreement at every later step (the consumer is no
    # longer acting), and the alive fraction is recorded alongside so death and disagreement
    # stay separable. The trivial bar is measured on THIS policy's own trajectory
    # distribution and on q0 only: max(per-step majority class of the q0 oracle labels along
    # the trajectory, previous-executed-action agreement along the trajectory).
    per_t = np.zeros(int(steps), dtype=np.float64)
    alive = np.zeros(int(steps), dtype=np.float64)
    resources: List[int] = []
    labels_seen: List[int] = []
    prev_hits = 0
    prev_n = 0
    for ep in range(int(n_eps)):
        env = x734._make_env(seed * 1000 + 9000 + ep, env_kwargs)
        _flat, obs = env.reset()
        res = 0
        last_a = -1
        for t in range(int(steps)):
            field = x1002._localfield_vector(obs)
            label = int(_query_labels(field, 0)[0])
            a = int(policy_fn(field))
            per_t[t] += float(a == label)
            alive[t] += 1.0
            labels_seen.append(label)
            prev_hits += int(last_a == label)
            prev_n += 1
            last_a = a
            _f, _h, done, info, obs = env.step(a)
            if isinstance(info, dict) and str(info.get("transition_type", "")) == "resource":
                res += 1
            if done:
                break
        resources.append(res)
    n_eps_f = float(max(1, int(n_eps)))
    series = [float(per_t[i] / n_eps_f) for i in range(int(steps))]
    alive_frac = [float(alive[i] / n_eps_f) for i in range(int(steps))]
    maj = (max(_class_shares(torch.tensor(labels_seen, dtype=torch.long)))
           if labels_seen else 0.0)
    prev_agree = (float(prev_hits) / float(prev_n)) if prev_n else 0.0
    trivial_cl = max(maj, prev_agree)
    horizon = min(int(n_use), int(steps))
    step_n = series[horizon - 1] if horizon >= 1 else None
    first_below: Optional[int] = None
    for i in range(horizon):
        if series[i] < trivial_cl:
            first_below = i + 1
            break
    ys = series[:horizon]
    slope = None
    if len(ys) >= 2:
        arr = np.asarray(ys, dtype=np.float64)
        slope = float(np.polyfit(np.arange(arr.size, dtype=np.float64), arr, 1)[0])
    return {"step_1_agreement": series[0] if series else None,
            "step_N_agreement": step_n, "n_use": int(horizon),
            "alive_fraction_step_1": alive_frac[0] if alive_frac else None,
            "alive_fraction_step_N": alive_frac[horizon - 1] if horizon >= 1 else None,
            "decay_slope": slope, "first_step_below_trivial": first_below,
            "series": ys, "alive_fraction": alive_frac[:horizon],
            "trivial_closed_loop": trivial_cl,
            "trivial_closed_loop_parts": {"q0_majority_share_on_trajectory": maj,
                                          "prev_action_on_trajectory": prev_agree},
            "trivial_block3_reference": trivial,
            "foraging_competence": float(np.mean(resources)) if resources else None,
            "n_episodes": int(n_eps), "denominator": "every started episode",
            "one_step_only": bool(series and step_n is not None
                                  and series[0] >= trivial_cl and step_n < trivial_cl)}


def _rollout_policies(heads: List[Any], arms: Dict[Tuple[str, int], _FittedArm], width: int,
                      seed: int) -> Dict[str, Callable[[torch.Tensor], int]]:
    """Closed-loop policies: head 0, query q0; every observation through the interface."""
    q0 = _onehot(torch.zeros(1, dtype=torch.long), K_QUERIES)
    rows1 = {"query": torch.zeros(1, dtype=torch.long), "head": torch.zeros(1, dtype=torch.long),
             "frame": torch.full((1,), FRAME_IDS.index(GATE_A1_FRAME), dtype=torch.long)}
    rng = np.random.RandomState(int(seed) + 77)

    def _head_act(msg: torch.Tensor) -> int:
        with torch.no_grad():
            logits, _v = heads[0](torch.cat([msg, q0], dim=-1))
        return int(torch.argmax(logits.reshape(-1)).item())

    pol: Dict[str, Callable[[torch.Tensor], int]] = {
        "random": lambda field: int(rng.randint(0, 5)),
        "A0_native_canonical": lambda field: _head_act(field),
        "A0_native_under_broken_frame": lambda field: _head_act(_apply_frame(field, GATE_A1_FRAME)),
    }
    for arm_id in (ARM_A4, ARM_A1, ARM_A2):
        arm = arms.get((arm_id, width))
        if arm is None:
            continue
        pol[arm_id] = (lambda field, _a=arm: _head_act(
            _a.message(rows1, _apply_frame(field, GATE_A1_FRAME))))
    return pol


# ---------------------------------------------------------------------------------------
# THE PER-SEED PIPELINE
# ---------------------------------------------------------------------------------------
def _run_seed(seed: int, env_kwargs: Dict[str, Any], cfg: Dict[str, Any]
              ) -> Dict[str, Any]:
    print("Seed %d Condition assay_a_bandwidth_ladder" % seed, flush=True)
    reset_all_rng(seed)
    t_seed = time.perf_counter()

    episodes = _collect(seed, env_kwargs, cfg["bc_episodes"], cfg["bc_random_episodes"],
                        cfg["steps"])
    blocks = _blocks(episodes)
    r0 = _rows(blocks["block0_consumer"], seed, True)
    r1 = _rows(blocks["block1_fit"], seed, True)
    r2 = _rows(blocks["block2_select"], seed, True)
    r3 = _rows(blocks["block3_test"], seed, True)
    if not all(int(r["field"].shape[0]) > 0 for r in (r0, r1, r2, r3)):
        raise RuntimeError("empty block after split -- collected %d episodes" % len(episodes))

    # ---- frozen consumer heads: trained ONCE per seed on block 0, CANONICAL field --------
    heads = []
    head_fits = []
    for h in range(K_HEADS):
        head = _make_head(seed * 10 + h)
        fit = _train_head(head, r0["field"], _onehot(r0["query"], K_QUERIES), r0["label"],
                          cfg["head_passes"], seed, "h%d" % h)
        heads.append(head)
        head_fits.append(fit)
    head_hashes = [iprobe.hash_tensor_state(h) for h in heads]

    triv2 = _trivial_family(r1, r2)
    triv3 = _trivial_family(r1, r3)
    trivial = float(triv3["strongest"])
    state_blind_worst = max(_majority_class_frac(r3["label"][r3["query"] == qi])
                            for qi in range(K_QUERIES))
    label_balance = {
        "train_block0": {QUERY_IDS[qi]: _class_shares(r0["label"][r0["query"] == qi])
                         for qi in range(K_QUERIES)},
        "fit_block1": {QUERY_IDS[qi]: _class_shares(r1["label"][r1["query"] == qi])
                       for qi in range(K_QUERIES)},
        "test_block3": {QUERY_IDS[qi]: _class_shares(r3["label"][r3["query"] == qi])
                        for qi in range(K_QUERIES)},
    }

    pres1 = _present(r1)
    pres2 = _present(r2)
    pres3 = _present(r3)

    # ------------------------------ GATE A0: INSTRUMENT -----------------------------
    gates: Dict[str, Any] = {}

    st = x1002._fit_standardiser(r1["field"])
    dec = x1010._make_decoder(CONSUMER_RUNG, RESOURCE_FIELD_DIM, 5)
    x1010._train_decoder(dec, x1002._apply_standardiser(r1["field"], st),
                         _query_labels(r1["field"], 0), cfg["head_passes"], seed, "g1_source", 5)
    g1 = x1010._agreement(dec, x1002._apply_standardiser(r3["field"], st),
                          _query_labels(r3["field"], 0))
    gates["G1_source_adequacy"] = {"measured": g1, "threshold": AGREEMENT_BAR,
                                   "green": bool(g1 is not None and g1 >= AGREEMENT_BAR)}

    env_o = x734._make_env(seed * 1000 + 7001, env_kwargs)
    env_r = x734._make_env(seed * 1000 + 7001, env_kwargs)
    orc = evaluate_seed(LocalViewGreedyPolicy(seed), env_o, cfg["competence_episodes"],
                        cfg["steps"])
    rnd = evaluate_seed(RandomPolicy(seed + 991), env_r, cfg["competence_episodes"],
                        cfg["steps"])
    g2_range = float(orc["foraging_competence"] - rnd["foraging_competence"])
    gates["G2_consumer_range"] = {"measured": g2_range, "threshold": COMPETENCE_RANGE_MIN,
                                  "green": bool(g2_range >= COMPETENCE_RANGE_MIN),
                                  "oracle_foraging": orc["foraging_competence"],
                                  "random_foraging": rnd["foraging_competence"]}

    g3_score = _score_arm("G3_native_canonical", heads, r3["field"], r3)
    g3 = g3_score["agreement"]
    g3_elev = (None if g3 is None else float(g3 - trivial))
    gates["G3_consumer_floor"] = {"measured": g3_elev, "threshold": AGREEMENT_ELEVATION_MIN,
                                  "direction": "lower", "agreement": g3,
                                  "trivial": trivial, "trivial_family": triv3,
                                  "green": bool(g3_elev is not None
                                                and g3_elev >= AGREEMENT_ELEVATION_MIN),
                                  "per_head": g3_score["per_head"],
                                  "per_query": g3_score["per_query"]}

    # G4: the exact analytic inverse realised as a LINEAR ROUTE MODULE and pushed through
    # the SAME _FittedArm.message -> _heads_logits -> _score_arm path every fitted arm takes
    # (red-team pass 2, F8: an index-permutation round trip scored directly is bit-identical
    # to G3 and certifies nothing about the pipeline).
    inv_route = _LinearRoute(RESOURCE_FIELD_DIM)
    with torch.no_grad():
        inv_route.down.weight.copy_(torch.eye(RESOURCE_FIELD_DIM)[ROT90_INV_PERM])
        inv_route.up.weight.copy_(torch.eye(RESOURCE_FIELD_DIM))
        inv_route.up.bias.zero_()
    inv_arm = _FittedArm("G4_inverse_route", RESOURCE_FIELD_DIM, {}, {},
                         lambda rows_, presented_, so, io: inv_route(presented_))
    g4 = _score_arm("G4_inverse", heads, inv_arm.message(r3, pres3), r3)["agreement"]
    gates["G4_known_inverse"] = {
        "measured": g4, "reference": g3, "band": EQUIVALENCE_BAND,
        "green": bool(g4 is not None and g3 is not None and abs(g4 - g3) <= EQUIVALENCE_BAND),
        "route": "exact inverse permutation as a _LinearRoute module through the fitted-arm path",
        "is_pipeline_integrity_control_not_information_control": True}

    g5_zero = _score_arm(CTRL_ZERO_MSG, heads, torch.zeros_like(pres3), r3)["agreement"]
    g5_rand = _score_arm(CTRL_RANDOM_MSG, heads, _moment_matched_random(pres3, seed), r3
                         )["agreement"]
    g5_worst = max(v for v in (g5_zero, g5_rand) if v is not None)
    gates["G5_negative_floor"] = {"zero_message": g5_zero, "random_message": g5_rand,
                                 "measured": g5_worst,
                                 "threshold": float(trivial + EQUIVALENCE_BAND),
                                 "direction": "upper", "strongest_trivial": trivial,
                                 "strongest_trivial_name": triv3["strongest_name"],
                                 "green": bool(g5_worst <= trivial + EQUIVALENCE_BAND)}

    wit = x1010._make_decoder(MAX_CAPACITY_RUNG, RESOURCE_FIELD_DIM, 5)
    x1010._train_decoder(wit, x1002._apply_standardiser(r1["field"], st),
                         _query_labels(r1["field"], 0), cfg["head_passes"], seed,
                         "g6_witness", 5)
    g6 = x1010._agreement(wit, x1002._apply_standardiser(r1["field"], st),
                          _query_labels(r1["field"], 0))
    gates["G6_capacity_witness"] = {"measured": g6, "threshold": MEMORISE_FLOOR,
                                    "green": bool(g6 is not None and g6 >= MEMORISE_FLOOR)}

    a0_brk = _score_arm(ARM_A0, heads, pres3, r3)
    headroom = float((g3 or 0.0) - (a0_brk["agreement"] or 0.0))
    gates["G9_broken_access_headroom"] = {
        "measured": headroom, "threshold": HEADROOM_MIN, "direction": "lower",
        "native_canonical": g3, "native_under_broken_frame": a0_brk["agreement"],
        "broken_frame": BROKEN_FRAME, "n_rows": a0_brk["n_rows"],
        "green": bool(headroom >= HEADROOM_MIN)}

    ladder = iprobe.bridge_ladder(
        pres1, r1["field"], pres3, r3["field"],
        low_rank_k=4, l4_bottleneck=LADDER_L4_BOTTLENECK, l4_epochs=cfg["ladder_epochs"],
        l5_epochs=cfg["ladder_epochs"],
        consumer_eval_fn=lambda mapped: _score_arm(
            "ladder", heads, mapped.to(torch.float32), r3)["agreement"],
        seed=seed,
    )
    constrained = [BridgeLevel.L1_PROCRUSTES.value, BridgeLevel.L2_AFFINE.value,
                   BridgeLevel.L3_LOW_RANK_AFFINE.value,
                   BridgeLevel.L4_CONSTRAINED_NONLINEAR.value]
    min_clearing = None
    for lvl in constrained:
        eff = ladder.get(lvl, {}).get("downstream_behavioural_effect")
        if eff is not None and g3 is not None and abs(eff - g3) <= EQUIVALENCE_BAND:
            min_clearing = lvl
            break
    gates["G10_bridgeability"] = {
        "min_clearing_rung": min_clearing, "constrained_rungs_checked": constrained,
        "band": EQUIVALENCE_BAND, "reference": g3,
        "per_rung_consumer_agreement": {
            k: v.get("downstream_behavioural_effect") for k, v in ladder.items()},
        "l5_is_information_in_principle_only_not_a_plausible_interface": True,
        "green": bool(min_clearing is not None)}

    # ------------------------------ GATE A1: THE BANDWIDTH LADDER -------------------
    store = {"key": pres1, "val": r1["field"], "episode": r1["episode"],
             "centroids": _fit_kmeans(pres1, K_SELECTOR, seed), "codes": _selector_codes(seed)}
    m_grid = list(cfg["m_grid"])
    arms: Dict[Tuple[str, int], _FittedArm] = {}
    sweep_rows: List[Dict[str, Any]] = []
    fit_ces: List[Tuple[str, int, int, float, float]] = []
    base_slice = {"rung_id": RUNG_ID, "gate_a1_frame": GATE_A1_FRAME, "cond_dim": COND_DIM,
                  "n_restarts": int(cfg["n_restarts"]),
                  "bridge_passes": cfg["bridge_passes"], "head_passes": cfg["head_passes"],
                  "bridge_weight_decay": BRIDGE_WEIGHT_DECAY, "k_selector": K_SELECTOR,
                  "k_heads": K_HEADS, "k_queries": K_QUERIES, "consumer_rung": CONSUMER_RUNG,
                  "objective": "end_to_end_consumer_ce_through_frozen_heads",
                  "grad_clip_norm": GRAD_CLIP_NORM, "adapter_lr": ADAPTER_LR,
                  "adapter_batch": ADAPTER_BATCH, "steps": cfg["steps"],
                  "bc_episodes": cfg["bc_episodes"],
                  "bc_random_episodes": cfg["bc_random_episodes"]}

    def _row_at(arm_id: str, width: int) -> Dict[str, Any]:
        return next(x for x in sweep_rows if x["arm_id"] == arm_id and x["width"] == width)

    for width in m_grid:
        for arm_id in SWEEP_ARMS:
            cfg_slice = dict(base_slice, arm_id=arm_id, width=int(width))
            with arm_cell(seed, config_slice=cfg_slice, script_path=Path(__file__),
                          config_slice_declared=True,
                          extra_substrate_paths=_EXTRA_SUBSTRATE_PATHS,
                          include_driver_script_in_hash=False) as cell:
                restarts: List[Dict[str, Any]] = []
                for rs in range(int(cfg["n_restarts"])):
                    cand = _build_arm(arm_id, width, seed, heads, r1, pres1, store,
                                      cfg["bridge_passes"], restart=rs)
                    c2 = _score_arm(arm_id, heads, cand.message(r2, pres2), r2)
                    c3 = _score_arm(arm_id, heads, cand.message(r3, pres3), r3)
                    if cand.fit_info.get("final_ce") is not None:
                        fit_ces.append((arm_id, int(width), rs,
                                        float(cand.fit_info.get("first_ce") or 0.0),
                                        float(cand.fit_info["final_ce"])))
                    restarts.append({"restart": rs, "arm": cand, "s2": c2, "s3": c3})
                # Model selection on BLOCK 2 only (section 1.5 guard 4).
                best = max(restarts, key=lambda t: (t["s2"]["agreement"]
                                                    if t["s2"]["agreement"] is not None
                                                    else -1.0))
                arm, s2, s3 = best["arm"], best["s2"], best["s3"]
                arms[(arm_id, width)] = arm
                row: Dict[str, Any] = {
                    "arm_id": arm_id, "width": int(width), "seed": int(seed),
                    "selected_restart": int(best["restart"]),
                    "n_restarts": int(cfg["n_restarts"]),
                    "restarts_block2_agreement": [t["s2"]["agreement"] for t in restarts],
                    "restarts_block3_agreement": [t["s3"]["agreement"] for t in restarts],
                    "restarts_final_ce": [t["arm"].fit_info.get("final_ce") for t in restarts],
                    "block2_agreement": s2["agreement"],
                    "block2_elevation": (None if s2["agreement"] is None
                                         else float(s2["agreement"] - triv2["strongest"])),
                    "consumer_use_agreement": s3["agreement"],
                    "agreement_elevation": (None if s3["agreement"] is None
                                            else float(s3["agreement"] - trivial)),
                    "per_head_agreement": s3["per_head"],
                    "per_query_agreement": s3["per_query"],
                    "n_rows": s3["n_rows"], "trivial_predictor_agreement": trivial,
                    "bridge_fit": arm.fit_info, "capacity": arm.capacity,
                    "cond_mask": _cond_mask(arm_id).tolist(),
                }
                cell.stamp(row)
            sweep_rows.append(row)
        print("  [sweep] seed=%d M=%d %s" % (seed, width, " ".join(
            "%s=%.3f" % (a.split("_")[0], _row_at(a, width)["consumer_use_agreement"] or 0.0)
            for a in SWEEP_ARMS)), flush=True)

    # A0 row (no width): the floor under the broken frame.
    a0_row = {"arm_id": ARM_A0, "width": None, "seed": int(seed),
              "consumer_use_agreement": a0_brk["agreement"],
              "agreement_elevation": float((a0_brk["agreement"] or 0.0) - trivial),
              "per_head_agreement": a0_brk["per_head"], "per_query_agreement": a0_brk["per_query"],
              "n_rows": a0_brk["n_rows"], "trivial_predictor_agreement": trivial,
              "native_canonical_agreement": g3,
              "capacity": {"module_class": "identity", "total_params": 0}}

    # ---- selection on block 2 (pre-registered rules) --------------------------------
    def _m_suff(arm_id: str) -> Optional[int]:
        for width in m_grid:
            r = _row_at(arm_id, width)
            if r["block2_elevation"] is not None and r["block2_elevation"] >= AGREEMENT_ELEVATION_MIN:
                return int(width)
        return None

    m_suff = {a: _m_suff(a) for a in SWEEP_ARMS}
    rival_suff = [(a, m_suff[a]) for a in RIVAL_ARMS if m_suff[a] is not None]
    m_a4 = m_suff[ARM_A4]
    rivals_nonfunctional = not rival_suff
    if rival_suff:
        m_star = min(v for _a, v in rival_suff)
    else:
        # No rival is functional at ANY width. That is not an instrument refusal when A4 IS
        # functional (red-team pass 2, F6): compare at A4's own width, where every rival is
        # below the floor by construction of the selection rule.
        m_star = m_a4
    m_star_holders = [a for a, v in rival_suff if v == m_star] if rival_suff else []

    def _best_rival_at(width: int) -> Tuple[Optional[str], Optional[float], Optional[float]]:
        """Best rival IDENTITY chosen on BLOCK 2 (never on the test block -- red-team pass 2,
        F3); returns (arm_id, its block-2 agreement, its BLOCK-3 agreement)."""
        cands = [(a, _row_at(a, width)["block2_agreement"], _row_at(a, width)["consumer_use_agreement"])
                 for a in RIVAL_ARMS]
        cands = [c for c in cands if c[1] is not None and c[2] is not None]
        if not cands:
            return None, None, None
        return max(cands, key=lambda t: t[1])

    selection: Dict[str, Any] = {
        "rule": ("m_suff(arm) = narrowest M with block-2 elevation over the strongest "
                 "trivial predictor >= AGREEMENT_ELEVATION_MIN; M* = min over rivals (A4's "
                 "own width if no rival is functional anywhere); M_A4 = m_suff(A4). The best "
                 "rival at a width is chosen by BLOCK-2 agreement and scored on block 3. "
                 "Both widths chosen on block 2 only; block 3 read once."),
        "m_suff": m_suff, "M_star": m_star, "M_star_holders": m_star_holders,
        "M_A4": m_a4, "rivals_nonfunctional_at_every_width": bool(rivals_nonfunctional),
        "bandwidth_saving": (None if (m_star is None or m_a4 is None) else int(m_star - m_a4)),
        "trivial_block2": triv2,
    }
    for tag, width in (("M_star", m_star), ("M_A4", m_a4)):
        if width is None:
            selection["best_rival_at_" + tag] = None
            selection["C1_delta_at_" + tag] = None
            continue
        rid, r2v, r3v = _best_rival_at(width)
        a4v = _row_at(ARM_A4, width)["consumer_use_agreement"]
        selection["best_rival_at_" + tag] = rid
        selection["best_rival_block2_agreement_at_" + tag] = r2v
        selection["best_rival_agreement_at_" + tag] = r3v
        selection["A4_agreement_at_" + tag] = a4v
        selection["C1_delta_at_" + tag] = (None if (a4v is None or r3v is None)
                                           else float(a4v - r3v))
    selection["C1b_delta"] = selection["C1_delta_at_M_star"]
    selection["C1a_delta"] = selection["C1_delta_at_M_A4"]
    selection["bandwidth_saving_positive"] = bool(m_star is not None and m_a4 is not None
                                                  and m_a4 < m_star)

    # ---- evaluation widths: per-arm readiness gates + Gate A2 controls -------------
    eval_widths = sorted(set(w for w in (m_star, m_a4) if w is not None))
    arm_gates: List[Dict[str, Any]] = []
    control_rows: List[Dict[str, Any]] = []
    per_width: Dict[str, Any] = {}
    base_measured = {"query_not_state_blind": float(state_blind_worst),
                     "head_native_elevation": float(g3_elev or 0.0)}

    def _gate_for(arm_id: str, width: int, arm: Optional[_FittedArm]) -> Dict[str, Any]:
        ctx = _arm_ctx(arm_id)
        measured = dict(base_measured)
        if ctx["is_fitted"] and arm is not None:
            fce = arm.fit_info.get("final_ce")
            measured["fit_final_ce_below_uniform"] = float(UNIFORM_LOGIT_CE if fce is None
                                                           else fce)
        if ctx["conditioning_varies"] and ctx["carries_message"] and arm is not None:
            spread, changed = _cross_condition_spread(arm, r3, pres3, heads)
            measured["cond_message_spread"] = spread
            measured["cond_reaches_committed_action"] = changed
        gate = evaluate_arm_gate("%s@M%d" % (arm_id, width), ctx, PRECONDITION_SPECS, measured)
        gate["arm_id"] = arm_id
        gate["width"] = int(width)
        return gate

    for width in eval_widths:
        a4 = arms[(ARM_A4, width)]
        a4_msg = a4.message(r3, pres3)
        a4_score = _score_arm(ARM_A4, heads, a4_msg, r3)["agreement"]
        for arm_id in SWEEP_ARMS:
            arm_gates.append(_gate_for(arm_id, width, arms[(arm_id, width)]))

        # C3: receiver-STATE permutation (bridge input; the head keeps the true query).
        g = torch.Generator().manual_seed(seed + 6161)
        so = _permute_within(r3["head"], r3["query"], g)
        m_sp = a4.message(r3, pres3, state_override=so)
        s_sp = _score_arm(CTRL_STATE_PERM, heads, m_sp, r3)
        # C4: receiver-IDENTITY permutation: the SAME message DELIVERED to a shuffled head.
        g = torch.Generator().manual_seed(seed + 6262)
        ho = _permute_within(r3["query"], r3["head"], g)
        s_hp = _score_arm(CTRL_HEAD_PERM, heads, a4_msg, r3, head_override=ho)
        # Pairing specificity (rung 5): A4's message from ANOTHER row of the same cell.
        g = torch.Generator().manual_seed(seed + 5150)
        perm = _within_cell_row_permutation(r3, g)
        s_we = _score_arm(CTRL_WRONG_EP, heads, a4_msg[perm], r3)
        n3 = int(pres3.shape[0])
        ctrl_specs = [
            (CTRL_STATE_PERM, s_sp,
             {"shuffle_moved_fraction": float((so != r3["query"]).float().mean())}),
            (CTRL_HEAD_PERM, s_hp,
             {"shuffle_moved_fraction": float((ho != r3["head"]).float().mean())}),
            (CTRL_WRONG_EP, s_we,
             {"shuffle_moved_fraction": float((perm != torch.arange(n3)).float().mean())}),
            (CTRL_ZERO_MSG, {"agreement": g5_zero, "per_head": {}, "per_query": {},
                             "n_rows": n3}, {}),
            (CTRL_RANDOM_MSG, {"agreement": g5_rand, "per_head": {}, "per_query": {},
                               "n_rows": n3}, {}),
        ]
        for ctrl_id, sc, extra in ctrl_specs:
            cfg_slice = dict(base_slice, arm_id=ctrl_id, width=int(width))
            with arm_cell(seed, config_slice=cfg_slice, script_path=Path(__file__),
                          config_slice_declared=True,
                          extra_substrate_paths=_EXTRA_SUBSTRATE_PATHS,
                          include_driver_script_in_hash=False) as cell:
                row = {"arm_id": ctrl_id, "width": int(width), "seed": int(seed),
                       "built_on": ARM_A4, "consumer_use_agreement": sc["agreement"],
                       "agreement_elevation": (None if sc["agreement"] is None
                                               else float(sc["agreement"] - trivial)),
                       "per_head_agreement": sc.get("per_head", {}),
                       "per_query_agreement": sc.get("per_query", {}),
                       "n_rows": sc["n_rows"], "trivial_predictor_agreement": trivial,
                       "A4_intact_agreement": a4_score,
                       "capacity": a4.capacity}
                row.update(extra)
                cell.stamp(row)
            control_rows.append(row)

        # C7 / rung 7: closed-loop rollouts through the interface at this width.
        pol = _rollout_policies(heads, arms, width, seed)
        rollouts = {name: _closed_loop(fn, seed, env_kwargs, cfg["rollout_episodes"],
                                       cfg["steps"], cfg["repeated_use_steps"], trivial)
                    for name, fn in pol.items()}

        # Three decodings (section 1.4): fitted on block 1, scored on block 3.
        def _decode(x_fit: torch.Tensor, x_test: torch.Tensor, label: str) -> Optional[float]:
            net = x1010._make_decoder(CONSUMER_RUNG, int(x_fit.shape[1]), 5)
            x1010._train_decoder(net, x_fit, r1["label"], cfg["head_passes"], seed, label, 5)
            return x1010._agreement(net, x_test, r3["label"])

        def _receiver_side(rows: Dict[str, torch.Tensor]) -> torch.Tensor:
            return torch.cat([_onehot(rows["query"], K_QUERIES), _onehot(rows["head"], K_HEADS),
                              _onehot(rows["frame"], N_FRAME_LEVELS),
                              _onehot(rows["prev_action"].clamp_min(0), 5)], dim=-1)

        a4_msg1 = a4.message(r1, pres1)
        a1 = arms[(ARM_A1, width)]
        a1_msg1 = a1.message(r1, pres1)
        a1_msg3 = a1.message(r3, pres3)
        d_sender = _decode(pres1, pres3, "D_sender")
        d_joint = _decode(torch.cat([pres1, _receiver_side(r1)], dim=-1),
                          torch.cat([pres3, _receiver_side(r3)], dim=-1), "D_joint")
        d_recv = _decode(_receiver_side(r1), _receiver_side(r3), "D_receiver_only")
        decodings = {
            "D_sender": d_sender, "D_surface": d_sender, "D_joint": d_joint,
            "D_receiver_only": d_recv,
            "joint_minus_sender": (None if (d_joint is None or d_sender is None)
                                   else float(d_joint - d_sender)),
            "query_blind_cap_note": ("D_sender decodes a QUERY-SPECIFIC label from the source "
                                     "alone, so it is capped near the majority share BY "
                                     "CONSTRUCTION; joint_minus_sender is large on any "
                                     "query-conditioned task and C6 reads it only in "
                                     "conjunction with A7 being high (spec 2.8 row 6)."),
            "message_only_decoding_A4": _decode(a4_msg1, a4_msg, "D_msg_A4"),
            "message_only_decoding_A1": _decode(a1_msg1, a1_msg3, "D_msg_A1"),
            "message_only_note": ("label decoded from the PRODUCED message with no query: a "
                                  "conditioned message that carries the query-specific "
                                  "answer decodes far above the query-blind cap"),
        }

        # C5: sender-state null as message invariance of A1 (+ diagnostics).
        inv_range = _state_invariance_range(a1, r3, pres3)
        a1_per_state = _score_arm(ARM_A1, heads, a1_msg3, r3)["per_query"]
        sender_null = {"A1_message_cross_state_range": inv_range,
                       "measured": inv_range, "threshold": INVARIANCE_TOL, "direction": "upper",
                       "invariant": bool(inv_range <= INVARIANCE_TOL),
                       "A1_per_receiver_state_agreement": a1_per_state,
                       "per_state_note": ("consumer-use differs across queries because the "
                                          "queries differ in difficulty (a property of the "
                                          "head); the criterion is message invariance")}

        a5v = _row_at(ARM_A5, width)["consumer_use_agreement"]
        uncond = [v for v in (_row_at(ARM_A1, width)["consumer_use_agreement"],
                              _row_at(ARM_A3, width)["consumer_use_agreement"]) if v is not None]
        per_width[str(width)] = {
            "A4_agreement": a4_score,
            "A4_state_permuted": s_sp["agreement"],
            "A4_head_permuted": s_hp["agreement"],
            "A4_wrong_episode": s_we["agreement"],
            "A5_agreement": a5v,
            "best_unconditional_agreement": (max(uncond) if uncond else None),
            "A5_minus_unconditional": (None if (a5v is None or not uncond)
                                       else float(a5v - max(uncond))),
            "A4_gate_green": bool(next(gg for gg in arm_gates
                                       if gg["arm_id"] == ARM_A4 and gg["width"] == width)
                                  ["gate_green"]),
            "rollouts": rollouts, "decodings": decodings, "sender_state_null": sender_null,
        }

    # G7 tests DIVERGENCE, not non-learning: a fit diverged if its final consumer CE is
    # non-finite or ended ABOVE both the uniform-logit value and its own first-pass CE. A
    # control that is capped near uniform by design (A7: zero message; A6: an 8-way code)
    # legitimately ends near log(5) and must not refuse the run.
    diverged = [(a, w, rs, f0, f1) for a, w, rs, f0, f1 in fit_ces
                if not _math.isfinite(f1) or f1 > max(f0, UNIFORM_LOGIT_CE) + 1e-6]
    gates["G7_no_divergence"] = {
        "measured": (max(f1 for _a, _w, _r, _f0, f1 in fit_ces) if fit_ces else 0.0),
        "threshold": UNIFORM_LOGIT_CE, "direction": "upper", "n_fits": len(fit_ces),
        "rule": "diverged iff final CE non-finite or > max(first CE, log(5))",
        "diverged_cells": ["%s@M%d#r%d" % (a, w, rs) for a, w, rs, _f0, _f1 in diverged],
        "green": bool(not diverged)}

    seed_gate_green = all(bool(g.get("green", True)) for g in gates.values())
    print("  [seed] %d M*=%s M_A4=%s C1b_delta=%s C1a_delta=%s saving=%s gates_green=%s "
          "elapsed=%.0fs" % (seed, m_star, m_a4, selection.get("C1b_delta"),
                             selection.get("C1a_delta"), selection["bandwidth_saving"],
                             seed_gate_green, time.perf_counter() - t_seed), flush=True)
    print("verdict: %s" % ("PASS" if seed_gate_green else "FAIL"), flush=True)

    # Source exchangeability across query strata (diagnostic for the sender-state null).
    fmean = r3["field"].mean(dim=0)
    fsd = r3["field"].std(dim=0).clamp_min(1e-8)
    smd = max(float(((r3["field"][r3["query"] == qi].mean(dim=0) - fmean) / fsd).abs().max())
              for qi in range(K_QUERIES) if bool((r3["query"] == qi).any()))

    return {
        "seed": int(seed),
        "sweep_rows": sweep_rows, "a0_row": a0_row, "control_rows": control_rows,
        "arm_gates": arm_gates, "gates": gates, "selection": selection,
        "per_width": per_width, "eval_widths": eval_widths,
        "head_fits": head_fits, "head_hashes": head_hashes,
        "trivial_predictor_agreement": trivial, "trivial_family_block3": triv3,
        "query_state_blind_worst": state_blind_worst, "label_balance": label_balance,
        "source_stratum_max_smd": smd,
        "block_sizes": {k: len(v) for k, v in blocks.items()},
        "block_rows": {"block0": int(r0["field"].shape[0]), "block1": int(r1["field"].shape[0]),
                       "block2": int(r2["field"].shape[0]), "block3": int(r3["field"].shape[0])},
        "elapsed_seconds": float(time.perf_counter() - t_seed),
    }


# ---------------------------------------------------------------------------------------
# ADJUDICATION
# ---------------------------------------------------------------------------------------
def _adjudicate(seed_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Gate A0 (instrument), then Gate A1 (C1a / C1b, C2), then Gate A2 (C3..C7)."""
    gate_names = ["G1_source_adequacy", "G2_consumer_range", "G3_consumer_floor",
                  "G4_known_inverse", "G5_negative_floor", "G6_capacity_witness",
                  "G7_no_divergence", "G9_broken_access_headroom", "G10_bridgeability"]
    gate_state: Dict[str, Any] = {}
    for gname in gate_names:
        greens = [bool(sr["gates"].get(gname, {}).get("green")) for sr in seed_results]
        gate_state[gname] = {"green_on_all_seeds": all(greens), "per_seed": greens}

    sels = [sr["selection"] for sr in seed_results]
    c1b_deltas = [s.get("C1b_delta") for s in sels]
    c1a_deltas = [s.get("C1a_delta") for s in sels]
    m_stars = [s.get("M_star") for s in sels]
    m_a4s = [s.get("M_A4") for s in sels]
    a4_at_star = [s.get("A4_agreement_at_M_star") for s in sels]
    rival_at_star = [s.get("best_rival_agreement_at_M_star") for s in sels]

    # G8: degeneracy of the load-bearing readouts. The cross-seed spread of the C1b delta
    # is only defined with >= 2 seeds (a single-seed confirmer would otherwise self-refuse
    # on a one-element list); the per-seed A4-vs-rival pair is always checked.
    degen_metrics: Dict[str, Any] = {
        "A4_vs_rival_at_M_star": {"groups": [[a, b] for a, b in zip(a4_at_star, rival_at_star)
                                             if a is not None and b is not None]},
    }
    c1b_valid = [d for d in c1b_deltas if d is not None]
    if len(c1b_valid) >= 2:
        degen_metrics["C1b_delta"] = c1b_valid
    c1a_valid = [d for d in c1a_deltas if d is not None]
    if len(c1a_valid) >= 2:
        degen_metrics["C1a_delta"] = c1a_valid
    degen = check_degeneracy(degen_metrics)
    degen["n_seeds_with_C1b_delta"] = len(c1b_valid)
    gate_state["G8_degeneracy"] = {"green_on_all_seeds": bool(degen.get("non_degenerate")),
                                   "per_seed": [bool(degen.get("non_degenerate"))],
                                   "detail": degen}

    rivals_nonfunctional = [bool(s.get("rivals_nonfunctional_at_every_width")) for s in sels]
    a4_never_functional = all(m is None for m in m_a4s)
    # Refuse only when NEITHER A4 NOR any rival is functional at any width (red-team pass 2,
    # F6): rivals all below the floor while A4 clears it is a result, not a refusal.
    no_functional_arm = all(m is None for m in m_stars) and a4_never_functional

    # The VERDICT arm's readiness gate at the widths it is adjudicated on.
    a4_gate_red: List[str] = []
    for sr in seed_results:
        for w in sr["eval_widths"]:
            pw = sr["per_width"][str(w)]
            if not pw["A4_gate_green"]:
                a4_gate_red.append("seed %d @M%d" % (sr["seed"], w))

    instrument_green = (all(v["green_on_all_seeds"] for v in gate_state.values())
                        and not no_functional_arm and not a4_gate_red)

    # ---- C1b: the spec's primary estimand at M* ----------------------------------
    c1b = _paired_delta_verdict(c1b_deltas)
    c1b["per_seed_delta"] = c1b_deltas
    c1b["per_seed_M_star"] = m_stars
    c1b["best_rival_by_seed"] = {str(sr["seed"]): sr["selection"].get("best_rival_at_M_star")
                                 for sr in seed_results}

    # ---- C1a: the bandwidth regime -- a PAIRED section 1.6 contrast at M_A4 ---------
    # (red-team pass 2, F2: the ordinal M_A4 < M* crossing had no variability discipline
    # and pitted one A4 draw against a minimum over five rivals; the ordinal saving is now a
    # recorded diagnostic and the criterion is the same effect-size rule C1b uses, at the
    # narrowest width where the conditioned arm is functional.)
    savings = [s.get("bandwidth_saving") for s in sels]
    n_saving = sum(1 for s in sels if s.get("bandwidth_saving_positive"))
    c1a = _paired_delta_verdict(c1a_deltas)
    c1a.update({"per_seed_delta": c1a_deltas, "per_seed_M_A4": m_a4s,
                "per_seed_M_star": m_stars, "per_seed_bandwidth_saving": savings,
                "n_seeds_with_ordinal_saving": int(n_saving),
                "mean_bandwidth_saving": (float(np.mean([s for s in savings if s is not None]))
                                          if any(s is not None for s in savings) else None),
                "best_rival_by_seed": {str(sr["seed"]): sr["selection"].get("best_rival_at_M_A4")
                                       for sr in seed_results}})

    # ---- C2: the receiver-only control ------------------------------------------
    a7_vals = [r["consumer_use_agreement"] for sr in seed_results for r in sr["sweep_rows"]
               if r["arm_id"] == ARM_A7 and r["consumer_use_agreement"] is not None]
    c2_measured = (float(max(a7_vals)) if a7_vals else None)
    c2 = {"measured": c2_measured, "threshold": AGREEMENT_BAR, "direction": "upper",
          "passed": bool(c2_measured is not None and c2_measured < AGREEMENT_BAR),
          "n_cells": len(a7_vals), "structurally_expected_pass": True,
          "expected_reason": ("A7's information is the query and head one-hots, which can at "
                              "most reproduce the per-query majority class (~0.4); a control "
                              "with a known expected level, not a discriminating test")}

    # ---- Gate A2 at M_eval -------------------------------------------------------
    def _m_eval(sr: Dict[str, Any]) -> Optional[int]:
        s = sr["selection"]
        if c1b["passed"] and s.get("M_star") is not None:
            return int(s["M_star"])
        if s.get("M_A4") is not None:
            return int(s["M_A4"])
        return s.get("M_star")

    m_evals = [_m_eval(sr) for sr in seed_results]
    pw_eval = [(sr["per_width"].get(str(m)) if m is not None else None)
               for sr, m in zip(seed_results, m_evals)]

    state_dies = [(None if pw is None or pw["A4_agreement"] is None
                   or pw["A4_state_permuted"] is None
                   else float(pw["A4_agreement"] - pw["A4_state_permuted"])) for pw in pw_eval]
    c3v = _paired_delta_verdict(state_dies)
    c3 = {"measured": c3v["measured"], "threshold": DELTA_FLOOR, "direction": "lower",
          "passed": bool(c3v["passed"]), "per_seed_intact_minus_permuted": state_dies,
          "sd_gate_measured": c3v["sd_gate_measured"], "sd_gate_threshold": c3v["sd_gate_threshold"],
          "n_seeds_positive": c3v["n_seeds_positive"], "seeds_required": int(SEED_MAJORITY)}

    # C4: does IDENTITY conditioning by itself add use? A5 minus the best unconditional arm
    # at M_eval must stay below the floor (mean). This is the fallible form of the section
    # 2.8 routing test: A4 does not see the identity block and the heads are interchangeable
    # by construction, so delivering A4's message to a shuffled head cannot fail (red-team
    # pass 2, F5) -- that permutation is kept as a diagnostic (A4_head_permuted).
    a5_gain = [(None if pw is None else pw.get("A5_minus_unconditional")) for pw in pw_eval]
    a5g = [v for v in a5_gain if v is not None]
    c4_measured = (float(np.mean(a5g)) if a5g else None)
    head_loss = [(None if pw is None or pw["A4_agreement"] is None
                  or pw["A4_head_permuted"] is None
                  else float(pw["A4_agreement"] - pw["A4_head_permuted"])) for pw in pw_eval]
    c4 = {"measured": c4_measured, "threshold": DELTA_FLOOR, "direction": "upper",
          "passed": bool(c4_measured is not None and c4_measured < DELTA_FLOOR),
          "per_seed_A5_minus_best_unconditional": a5_gain,
          "diagnostic_A4_intact_minus_head_permuted": head_loss,
          "diagnostic_note": ("A4's message is identity-blind and the heads differ only by "
                              "initialisation, so the head-delivery permutation is a "
                              "structurally-expected ~0 and is not the criterion")}

    inv = [pw["sender_state_null"]["A1_message_cross_state_range"] for pw in pw_eval
           if pw is not None]
    c5_measured = (float(max(inv)) if inv else None)
    c5 = {"measured": c5_measured, "threshold": INVARIANCE_TOL, "direction": "upper",
          "passed": bool(c5_measured is not None and c5_measured <= INVARIANCE_TOL),
          "per_seed": inv, "is_pipeline_integrity_check": True}

    a7_high_by_seed = []
    jms = []
    for sr, pw in zip(seed_results, pw_eval):
        if pw is None:
            continue
        sb = float(sr["trivial_family_block3"]["per_query_constant"])
        a7s = [r["consumer_use_agreement"] for r in sr["sweep_rows"]
               if r["arm_id"] == ARM_A7 and r["consumer_use_agreement"] is not None]
        a7_high_by_seed.append(bool(a7s and max(a7s) >= sb + DELTA_FLOOR))
        jms.append(pw["decodings"]["joint_minus_sender"])
    jms_v = [v for v in jms if v is not None]
    c6_fail = any(h and (j is not None and j > DELTA_FLOOR) for h, j in zip(a7_high_by_seed, jms))
    c6 = {"measured": (float(max(jms_v)) if jms_v else None), "threshold": DELTA_FLOOR,
          "direction": "upper", "passed": bool(not c6_fail),
          "conjunction": "fails only if A7 high AND D_joint - D_sender > floor",
          "A7_high_by_seed": a7_high_by_seed, "joint_minus_sender_by_seed": jms}

    margins = []
    stepn = []
    cl_triv = []
    alive_n = []
    for sr, pw in zip(seed_results, pw_eval):
        if pw is None:
            continue
        ro = pw["rollouts"].get(ARM_A4, {})
        sn = ro.get("step_N_agreement")
        stepn.append(sn)
        cl_triv.append(ro.get("trivial_closed_loop"))
        alive_n.append(ro.get("alive_fraction_step_N"))
        if sn is not None and ro.get("trivial_closed_loop") is not None:
            margins.append(float(sn - ro["trivial_closed_loop"]))
    c7_measured = (float(min(margins)) if margins else None)
    c7 = {"measured": c7_measured, "threshold": 0.0, "direction": "lower",
          "passed": bool(c7_measured is not None and c7_measured >= 0.0),
          "per_seed_step_N": stepn, "per_seed_trivial_closed_loop": cl_triv,
          "per_seed_alive_fraction_step_N": alive_n, "per_seed_M_eval": m_evals,
          "denominator": "every started rollout episode; a dead episode is a non-agreement",
          "bar": "q0 closed-loop trivial on the arm's own trajectory (majority share or "
                 "previous action, whichever is larger)"}

    combination_rule = ("Gate A1 is adjudicated on ((C1b OR C1a) AND C2); C1b (paired "
                        "contrast at M*, the matched sufficient budget) and C1a (paired "
                        "contrast at M_A4, the narrowest width at which the conditioned arm is "
                        "functional) name two regimes and are reported separately. C3..C7 are "
                        "Gate A2 REFINEMENTS at M_eval (M* if C1b passed, else M_A4); they "
                        "change the LABEL, never C1's arithmetic. C2 and C5 are declared "
                        "structurally-expected guards (non-load-bearing); C5 red is an "
                        "instrument refusal.")

    g9_green = gate_state["G9_broken_access_headroom"]["green_on_all_seeds"]
    g10_green = gate_state["G10_bridgeability"]["green_on_all_seeds"]
    if not instrument_green:
        outcome = "FAIL"
        direction = "non_contributory"
        if not g9_green:
            label = "substrate_not_ready_requeue"
            summary = ("G9 red: the rotated presentation did not break the frozen consumer's "
                       "native access by at least %.2f agreement, so there is nothing for any "
                       "bridge to restore. Re-queue at a presentation that breaks access."
                       % HEADROOM_MIN)
        elif not g10_green or no_functional_arm:
            label = "instrument_refused_not_bridgeable"
            summary = ("No constrained ladder rung (G10) and/or no arm at any width in the "
                       "grid restores the frozen consumer: the conditional-vs-unconditional "
                       "question is unanswerable at admissible complexity.")
        elif a4_gate_red:
            label = "instrument_refused_verdict_arm_gate_red"
            summary = ("A4_receiver_state_cond's readiness gate is red at %s (inert "
                       "conditioning block or non-fitting bridge): refusal with a record, no "
                       "science leg adjudicated." % ", ".join(a4_gate_red))
        else:
            reds = [k for k, v in gate_state.items() if not v["green_on_all_seeds"]]
            label = "instrument_refused_gate_red"
            summary = ("Instrument gate red on %s; refusal with a record, no scientific leg "
                       "adjudicated (spec section 1.7)." % ", ".join(reds))
    elif not c2["passed"]:
        outcome, direction = "FAIL", "weakens"
        label = "receiver_only_answers_no_interface_claim"
        summary = ("A7_receiver_only clears the agreement bar: the receiver alone answers the "
                   "query, so no interface claim of any kind is licensed by this run.")
    elif not c5["passed"]:
        outcome, direction = "FAIL", "non_contributory"
        label = "instrument_refused_sender_state_leak"
        summary = ("A1_source_only's message varies with the receiver state (cross-state "
                   "range %.3g > %.1g): the state block leaked into the unconditional arm, a "
                   "pipeline defect. Refusal with a record." % (c5_measured or 0.0, INVARIANCE_TOL))
    elif c1b["passed"] or c1a["passed"]:
        outcome = "PASS"
        if c1b["passed"]:
            base = "receiver_state_conditioned_access_adds_use_at_sufficient_budget"
            direction = "supports"
            base_summary = ("C1b positive: at M* (the tightest budget at which an unconditional "
                            "or routing mechanism is functional) A4 beats its best fitted rival "
                            "by the pre-registered margin -- receiver-state-conditioned access "
                            "adds held-out consumer use even where an unconditional mechanism "
                            "suffices.")
        else:
            base = "conditional_access_buys_bandwidth_only"
            direction = "mixed"
            base_summary = ("C1b negative, C1a positive: at M* the best rival (%s) matches A4, "
                            "so the strong R3/MECH-547 reading is UNNECESSARY at sufficient "
                            "bandwidth (the T3-10 direction); but at M_A4 (the narrowest width "
                            "at which the conditioned arm is functional; ordinal saving %s "
                            "widths) A4 beats its best rival (%s) by the pre-registered margin, "
                            "so receiver-state conditioning adds use when the channel is "
                            "narrower than the unconditional map's sufficiency."
                            % (", ".join(sorted(set(v for v in c1b["best_rival_by_seed"].values()
                                                     if v))) or "none",
                               c1a["mean_bandwidth_saving"],
                               ", ".join(sorted(set(v for v in c1a["best_rival_by_seed"].values()
                                                     if v))) or "none"))
        if not c6["passed"]:
            label = "receiver_supplied_information_not_translation"
            direction = "non_contributory"
            summary = base_summary + (" HOWEVER A7 is high and D_joint exceeds D_sender by more "
                                      "than the floor: the receiver supplied the information "
                                      "and the bridge INTRODUCED rather than translated it.")
        elif not c3["passed"] and not c4["passed"]:
            label = "fixed_partner_specific_routing_not_conditioning"
            direction = "weakens"
            summary = base_summary + (" HOWEVER the gain survives receiver-state permutation "
                                      "while identity conditioning (A5) itself adds use over "
                                      "the unconditional arms: fixed partner-specific routing "
                                      "(T3-02/T3-03), not receiver conditioning. Section 2.7's "
                                      "sharper falsifier has fired.")
        elif not c3["passed"]:
            label = "advantage_not_state_specific_unattributed"
            direction = "non_contributory"
            summary = base_summary + (" HOWEVER the gain survives receiver-state permutation "
                                      "and identity conditioning adds nothing either, so it is "
                                      "attributable to neither state nor partner (a conditioned "
                                      "bridge with ANY varying auxiliary input fits better at "
                                      "this width); recorded as unattributed.")
        elif not c7["passed"]:
            label = base + "__one_step_only"
            direction = "mixed"
            summary = base_summary + (" HOWEVER closed-loop agreement at step %d falls below the "
                                      "strongest trivial predictor: one-step-only; MECH-548 "
                                      "instability blocks adoption." % N_REPEATED_USE)
        else:
            label = base
            summary = base_summary + (" The gain dies under state permutation, survives identity "
                                      "permutation, the receiver-only control is below bar, A1's "
                                      "message is state-invariant, and closed-loop use is stable "
                                      "to step %d. FORBIDDEN: any statement that a biological "
                                      "hippocampal mechanism does this." % N_REPEATED_USE)
    else:
        outcome, direction = "FAIL", "weakens"
        rivals_named = sorted(set(v for v in c1b["best_rival_by_seed"].values() if v))
        label = "conditional_access_earns_nothing_at_any_bandwidth"
        summary = ("C1b and C1a both negative: A4_receiver_state_cond beats its best fitted "
                   "rival (%s) by the pre-registered margin neither at M* nor at M_A4. "
                   "The strong R3/MECH-547 reading is UNNECESSARY at this interface at every "
                   "width in the grid -- the direction tranche 3 section 4.2(2) records the prior "
                   "as favouring (T3-10). FORBIDDEN: any statement that the interface is "
                   "unconditional in general." % (", ".join(rivals_named) or "none identified"))
        if a4_never_functional:
            summary += " NOTE: A4 never cleared the elevation floor at any width."

    return {
        "outcome": outcome, "label": label, "summary": summary,
        "evidence_direction": direction, "instrument_green": instrument_green,
        "gate_state": gate_state, "combination_rule": combination_rule,
        "M_eval_by_seed": m_evals,
        "criteria": [
            {"name": "C1b_a4_over_best_rival_at_M_star", "load_bearing": True,
             "passed": bool(c1b["passed"]), "measured": c1b["measured"],
             "threshold": c1b["threshold"], "sd_gate_measured": c1b["sd_gate_measured"],
             "sd_gate_threshold": c1b["sd_gate_threshold"],
             "n_seeds_positive": c1b["n_seeds_positive"], "seeds_required": int(SEED_MAJORITY),
             "detail": c1b},
            {"name": "C1a_a4_over_best_rival_at_M_A4", "load_bearing": True,
             "passed": bool(c1a["passed"]), "measured": c1a["measured"],
             "threshold": c1a["threshold"], "sd_gate_measured": c1a["sd_gate_measured"],
             "sd_gate_threshold": c1a["sd_gate_threshold"],
             "n_seeds_positive": c1a["n_seeds_positive"], "seeds_required": int(SEED_MAJORITY),
             "detail": c1a},
            {"name": "C2_receiver_only_below_bar", "load_bearing": False,
             "structurally_expected_pass": True,
             "passed": bool(c2["passed"]), "measured": c2["measured"],
             "threshold": c2["threshold"], "direction": c2["direction"], "detail": c2},
            {"name": "C3_advantage_dies_under_state_permutation", "load_bearing": False,
             "passed": bool(c3["passed"]), "measured": c3["measured"],
             "threshold": c3["threshold"], "direction": c3["direction"], "detail": c3},
            {"name": "C4_identity_conditioning_adds_nothing", "load_bearing": False,
             "passed": bool(c4["passed"]), "measured": c4["measured"],
             "threshold": c4["threshold"], "direction": c4["direction"], "detail": c4},
            {"name": "C5_sender_state_null_message_invariant", "load_bearing": False,
             "structurally_expected_pass": True, "passed": bool(c5["passed"]), "measured": c5["measured"],
             "threshold": c5["threshold"], "direction": c5["direction"], "detail": c5},
            {"name": "C6_receiver_contribution_bounded", "load_bearing": False,
             "passed": bool(c6["passed"]), "measured": c6["measured"],
             "threshold": c6["threshold"], "direction": c6["direction"], "detail": c6},
            {"name": "C7_repeated_use_stable", "load_bearing": False,
             "passed": bool(c7["passed"]), "measured": c7["measured"],
             "threshold": c7["threshold"], "direction": c7["direction"], "detail": c7},
        ],
        "degeneracy": degen,
    }


# ---------------------------------------------------------------------------------------
# RUN
# ---------------------------------------------------------------------------------------
def _cfg(dry_run: bool) -> Dict[str, Any]:
    if dry_run:
        return {"bc_episodes": DRY_RUN_BC_EPISODES,
                "bc_random_episodes": DRY_RUN_BC_RANDOM_EPISODES,
                "steps": DRY_RUN_STEPS, "head_passes": DRY_RUN_HEAD_PASSES,
                "bridge_passes": DRY_RUN_BRIDGE_PASSES,
                "repeated_use_steps": DRY_RUN_REPEATED_USE,
                "competence_episodes": DRY_RUN_COMPETENCE_EPISODES,
                "rollout_episodes": DRY_RUN_ROLLOUT_EPISODES,
                "n_restarts": DRY_RUN_RESTARTS,
                "m_grid": list(DRY_RUN_M_GRID), "ladder_epochs": 20}
    return {"bc_episodes": BC_EPISODES, "bc_random_episodes": BC_RANDOM_EPISODES,
            "steps": STEPS_PER_EPISODE, "head_passes": HEAD_PASSES,
            "bridge_passes": BRIDGE_PASSES, "repeated_use_steps": N_REPEATED_USE,
            "competence_episodes": 20, "rollout_episodes": ROLLOUT_EPISODES,
            "n_restarts": N_RESTARTS, "m_grid": list(M_GRID), "ladder_epochs": 200}


def _flat_scalar(adj: Dict[str, Any], seed_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """The FLAT, numeric, top-level readout the runpack converter harvests. Booleans as 0/1
    ints; non-finite and None values DROPPED (an absent key reads as unmeasured)."""
    out: Dict[str, Any] = {}

    def put(k: str, v: Any) -> None:
        if v is None:
            return
        if isinstance(v, bool):
            out[k] = int(v)
            return
        try:
            f = float(v)
        except (TypeError, ValueError):
            return
        if _math.isfinite(f):
            out[k] = f

    for c in adj["criteria"]:
        put(c["name"] + "__measured", c.get("measured"))
        put(c["name"] + "__threshold", c.get("threshold"))
        put(c["name"] + "__passed", c.get("passed"))
    put("instrument_green", adj["instrument_green"])
    for gname, gv in adj["gate_state"].items():
        put(gname + "__green", gv["green_on_all_seeds"])
    sels = [sr["selection"] for sr in seed_results]
    for key in ("M_star", "M_A4", "bandwidth_saving", "C1b_delta", "C1a_delta",
                "A4_agreement_at_M_star", "best_rival_agreement_at_M_star",
                "A4_agreement_at_M_A4", "best_rival_agreement_at_M_A4"):
        vals = [s.get(key) for s in sels if s.get(key) is not None]
        if vals:
            put("mean__" + key, float(np.mean(vals)))
    for arm_id in SWEEP_ARMS:
        for width in sorted(set(r["width"] for sr in seed_results for r in sr["sweep_rows"])):
            vals = [r["consumer_use_agreement"] for sr in seed_results for r in sr["sweep_rows"]
                    if r["arm_id"] == arm_id and r["width"] == width
                    and r["consumer_use_agreement"] is not None]
            if vals:
                put("agreement__%s__M%d" % (arm_id, width), float(np.mean(vals)))
    a0 = [sr["a0_row"]["consumer_use_agreement"] for sr in seed_results
          if sr["a0_row"]["consumer_use_agreement"] is not None]
    if a0:
        put("agreement__" + ARM_A0, float(np.mean(a0)))
    for name in ("random", "A0_native_canonical", "A0_native_under_broken_frame",
                 ARM_A4, ARM_A1, ARM_A2):
        vals = [pw["rollouts"][name]["foraging_competence"]
                for sr in seed_results for pw in sr["per_width"].values()
                if name in pw["rollouts"] and pw["rollouts"][name]["foraging_competence"] is not None]
        if vals:
            put("closed_loop_foraging__" + name, float(np.mean(vals)))
    put("trivial_predictor_agreement",
        float(np.mean([sr["trivial_predictor_agreement"] for sr in seed_results])))
    put("n_seeds", len(seed_results))
    return out


def run_experiment(seeds: List[int], dry_run: bool = False,
                   scratch_out: Optional[str] = None) -> Dict[str, Any]:
    t0 = time.perf_counter()
    cfg = _cfg(dry_run)
    env_kwargs = x734._env_kwargs_for_rung(RUNG)

    assert_no_structurally_unsatisfiable_gate(PRECONDITION_SPECS, _arm_contexts())

    seed_results = [_run_seed(s, env_kwargs, cfg) for s in seeds]
    adj = _adjudicate(seed_results)

    all_gates = [g for sr in seed_results for g in sr["arm_gates"]]
    per_arm_gate = aggregate_arm_gates(all_gates) if all_gates else {
        "non_degenerate": False, "degeneracy_reason": "no evaluation width selected",
        "adjudication_preconditions": [], "per_arm_gate": {}, "green_arms": [], "red_arms": []}

    arm_results: List[Dict[str, Any]] = []
    for sr in seed_results:
        arm_results.extend(sr["sweep_rows"])
        arm_results.extend(sr["control_rows"])

    # Per-criterion non-degeneracy, keyed to the owning arm's gate at the adjudicated widths
    # and to the numeric degeneracy check -- never a copy of instrument_green.
    def _arm_green_at_eval(arm_id: str, use_star: bool) -> bool:
        ok = True
        for sr in seed_results:
            w = sr["selection"].get("M_star" if use_star else "M_A4")
            if w is None:
                return False
            g = next((gg for gg in sr["arm_gates"] if gg["arm_id"] == arm_id and gg["width"] == w),
                     None)
            ok = ok and bool(g and g["gate_green"])
        return ok

    degen_ok = bool(adj["degeneracy"].get("non_degenerate"))
    a4_star = _arm_green_at_eval(ARM_A4, True)
    a4_a4 = _arm_green_at_eval(ARM_A4, False)
    a1_any = _arm_green_at_eval(ARM_A1, True) or _arm_green_at_eval(ARM_A1, False)
    a7_ok = _arm_green_at_eval(ARM_A7, True) or _arm_green_at_eval(ARM_A7, False)
    non_degen = {
        "C1b_a4_over_best_rival_at_M_star": bool(a4_star and degen_ok),
        "C1a_a4_over_best_rival_at_M_A4": bool(a4_a4 and degen_ok),
        "C2_receiver_only_below_bar": bool(a7_ok),
        "C3_advantage_dies_under_state_permutation": bool(a4_star or a4_a4),
        "C4_identity_conditioning_adds_nothing": bool(a4_star or a4_a4),
        "C5_sender_state_null_message_invariant": bool(a1_any),
        "C6_receiver_contribution_bounded": bool(a7_ok),
        "C7_repeated_use_stable": bool(a4_star or a4_a4),
    }

    full_config = {
        "rung_id": RUNG_ID, "env_kwargs": env_kwargs, "steps_per_episode": cfg["steps"],
        "bc_episodes": cfg["bc_episodes"], "bc_random_episodes": cfg["bc_random_episodes"],
        "head_passes": cfg["head_passes"], "bridge_passes": cfg["bridge_passes"],
        "m_grid": cfg["m_grid"], "n_restarts": cfg["n_restarts"],
        "bridge_weight_decay": BRIDGE_WEIGHT_DECAY,
        "objective": "end_to_end_consumer_ce_through_frozen_heads",
        "cond_dim": COND_DIM, "k_heads": K_HEADS, "k_queries": K_QUERIES,
        "k_selector": K_SELECTOR, "consumer_rung": CONSUMER_RUNG,
        "gate_a1_frame": GATE_A1_FRAME, "broken_frame": BROKEN_FRAME,
        "frame_levels": list(FRAME_IDS), "block_fracs": list(BLOCK_FRACS),
        "agreement_bar": AGREEMENT_BAR, "agreement_elevation_min": AGREEMENT_ELEVATION_MIN,
        "delta_floor": DELTA_FLOOR, "equivalence_band": EQUIVALENCE_BAND,
        "headroom_min": HEADROOM_MIN, "n_repeated_use": cfg["repeated_use_steps"],
        "rollout_episodes": cfg["rollout_episodes"], "ladder_l4_bottleneck": LADDER_L4_BOTTLENECK,
        "seeds": list(seeds), "dry_run": bool(dry_run),
    }

    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    manifest: Dict[str, Any] = {
        "run_id": "%s_%s_v3" % (EXPERIMENT_TYPE, ts),
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "architecture_epoch": ARCHITECTURE_EPOCH,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "sleep_driver_pattern": SLEEP_DRIVER_PATTERN,
        "claim_ids": list(CLAIM_IDS),
        "bears_on": list(BEARS_ON),
        "outcome": adj["outcome"],
        "evidence_direction": adj["evidence_direction"],
        "timestamp_utc": ts,
        "dry_run": bool(dry_run),
        "interpretation": {
            "label": adj["label"], "summary": adj["summary"],
            "combination_rule": adj["combination_rule"],
            "preconditions": per_arm_gate.get("adjudication_preconditions", []),
            "criteria_non_degenerate": non_degen,
            "M_eval_by_seed": adj["M_eval_by_seed"],
        },
        "criteria": adj["criteria"],
        "combination_rule": adj["combination_rule"],
        "instrument_gates": {sr["seed"]: sr["gates"] for sr in seed_results},
        "per_arm_gate": per_arm_gate.get("per_arm_gate", {}),
        "arm_results": arm_results,
        "per_seed_results": [
            {k: v for k, v in sr.items() if k not in ("sweep_rows", "control_rows", "arm_gates")}
            for sr in seed_results],
        "diagnostics": {
            "degeneracy": adj["degeneracy"],
            "rot90_field_permutation": ROT90_PERM,
            "rot90_action_map": {str(k): v for k, v in ROT90_ACTION.items()},
            "label_balance": {str(sr["seed"]): sr["label_balance"] for sr in seed_results},
            "deferred_stages": ["Gate A3 frame x query factorial with the compositional "
                                "holdout", "ws250_pca32 replication of Gate A1"],
        },
        "readout": {},
    }
    manifest.update({k: v for k, v in adj["degeneracy"].items()
                     if k.startswith("non_degenerate") or k in ("degeneracy_reason",
                                                                "degenerate_metrics")})
    manifest["readout"] = _flat_scalar(adj, seed_results)

    out_path = write_flat_manifest(
        manifest, out_dir=scratch_out, dry_run=dry_run, config=full_config,
        seeds=list(seeds), script_path=Path(__file__), started_at=t0,
    )
    return {"outcome": adj["outcome"], "manifest_path": out_path,
            "label": adj["label"], "summary": adj["summary"]}


def _self_test() -> int:
    """Design-time arithmetic that must hold before any compute is spent."""
    fails: List[str] = []
    x = torch.randn(7, RESOURCE_FIELD_DIM)
    if not torch.allclose(_invert_frame(_apply_frame(x, FRAME_ROT90), FRAME_ROT90), x):
        fails.append("rot90 is not exactly invertible")
    if sorted(ROT90_PERM) != list(range(RESOURCE_FIELD_DIM)):
        fails.append("rot90 field map is not a permutation")
    if sorted(ROT90_ACTION.values()) != sorted(ROT90_ACTION.keys()):
        fails.append("rot90 action map is not a bijection")
    if GATE_A1_FRAME == FRAME_CANONICAL:
        fails.append("Gate A1 must present the BROKEN frame, or there is nothing to translate")
    widths = {a: int(_cond_mask(a).shape[0]) for a in ARM_IDS}
    if len(set(widths.values())) != 1:
        fails.append("conditioning width differs across arms: %r" % widths)
    if float(_cond_mask(ARM_A1).sum()) != 0.0:
        fails.append("A1_source_only must see a constant-zero conditioning vector")
    if float(_cond_mask(ARM_A4)[COND_SLICE_STATE[0]:COND_SLICE_STATE[1]].sum()) != K_QUERIES:
        fails.append("A4 must see the whole state block")
    if float(_cond_mask(ARM_A4)[COND_SLICE_IDENT[0]:COND_SLICE_IDENT[1]].sum()) != 0.0:
        fails.append("A4 must NOT see the identity block")
    if float(_cond_mask(ARM_A5)[COND_SLICE_IDENT[0]:COND_SLICE_IDENT[1]].sum()) != K_HEADS:
        fails.append("A5 must see the whole identity block")
    if ARM_A3 in VARYING_COND_ARMS:
        fails.append("A3's conditioning is a CONSTANT at one frame; it must not be declared varying")
    g = torch.Generator().manual_seed(11)
    f = torch.rand(400, RESOURCE_FIELD_DIM, generator=g)
    for qi in range(K_QUERIES):
        lab = _query_labels(f, qi)
        if int(torch.unique(lab).numel()) < 2:
            fails.append("query %s is CONSTANT -- it cannot discriminate" % QUERY_IDS[qi])
    # Budget match: every bridge arm has the identical parameter count at every width.
    for w in M_GRID:
        caps = {a: _n_params([_make_bridge(1, w)]) for a in BRIDGE_ARMS}
        if len(set(caps.values())) != 1:
            fails.append("bridge arms not capacity-matched at M=%d: %r" % (w, caps))
    if list(M_GRID) != sorted(set(M_GRID)) or min(M_GRID) < 1:
        fails.append("M_GRID must be sorted, unique and >= 1")
    if not set(DRY_RUN_M_GRID) <= set(M_GRID):
        fails.append("DRY_RUN_M_GRID must be a subset of M_GRID")
    # Every (query, identity) cell is populated by the row-index assignment.
    n_probe = K_QUERIES * K_HEADS * 4
    q = torch.arange(n_probe) % K_QUERIES
    h = (torch.arange(n_probe) // K_QUERIES) % K_HEADS
    for qi in range(K_QUERIES):
        for hi in range(K_HEADS):
            if int(((q == qi) & (h == hi)).sum()) != 4:
                fails.append("cell (q=%d,h=%d) is not equally populated" % (qi, hi))
    # Leave-one-episode-out retrieval never returns the query's own episode.
    key = torch.randn(30, RESOURCE_FIELD_DIM)
    ep = torch.arange(30) // 10
    got = _nn_retrieve(key, key, ep.reshape(-1, 1).float().repeat(1, RESOURCE_FIELD_DIM),
                       ep, ep)
    if bool((got[:, 0].long() == ep).any()):
        fails.append("leave-one-episode-out retrieval returned the query's own episode")
    # k-means returns K_SELECTOR centroids; the codes are pairwise distinct.
    cent = _fit_kmeans(torch.randn(50, RESOURCE_FIELD_DIM), K_SELECTOR, 1, iters=3)
    if tuple(cent.shape) != (K_SELECTOR, RESOURCE_FIELD_DIM):
        fails.append("k-means did not return K_SELECTOR centroids")
    codes = _selector_codes(1)
    if float(torch.cdist(codes, codes).masked_fill(torch.eye(K_SELECTOR, dtype=torch.bool), 1.0).min()) <= 0:
        fails.append("selector codes are not pairwise distinct")
    if VERDICT_ARM in RIVAL_ARMS:
        fails.append("the verdict arm must not be in the best-rival set")
    if ARM_A0 in RIVAL_ARMS or ARM_A7 in RIVAL_ARMS:
        fails.append("A0 (floor) and A7 (control) must not be in the best-rival set")
    # Driver-stratified blocks: every block carries every driver (red-team pass 2, F1).
    toy = [{"obs": [], "labels": [0], "prev_actions": [-1], "driver": "oracle"} for _ in range(12)]
    toy += [{"obs": [], "labels": [0], "prev_actions": [-1], "driver": "random"} for _ in range(6)]
    try:
        bl = _blocks(toy)
        for k, v in bl.items():
            if set(e["driver"] for e in v) != {"oracle", "random"}:
                fails.append("block %s is not driver-stratified" % k)
    except AssertionError as exc:
        fails.append("block stratification assertion fired on a toy split: %s" % exc)
    # G4's inverse route module is exactly the inverse permutation.
    inv = _LinearRoute(RESOURCE_FIELD_DIM)
    with torch.no_grad():
        inv.down.weight.copy_(torch.eye(RESOURCE_FIELD_DIM)[ROT90_INV_PERM])
        inv.up.weight.copy_(torch.eye(RESOURCE_FIELD_DIM))
        inv.up.bias.zero_()
        if not torch.allclose(inv(_apply_frame(x, FRAME_ROT90)), x, atol=1e-6):
            fails.append("G4 inverse route module does not invert the presentation")
    # An unconditional arm's message is exactly state-invariant (C5 holds by construction).
    torch.manual_seed(3)
    br = _make_bridge(5, 4)
    rows = {"query": torch.arange(9) % 3, "head": torch.zeros(9, dtype=torch.long),
            "frame": torch.full((9,), FRAME_IDS.index(GATE_A1_FRAME), dtype=torch.long)}
    fa = _FittedArm(ARM_A1, 4, {}, {}, lambda r, p, so, io: br(p, _cond_vector(r, ARM_A1, so, io)))
    if _state_invariance_range(fa, rows, torch.randn(9, RESOURCE_FIELD_DIM)) != 0.0:
        fails.append("A1's message is not exactly state-invariant")
    for f_ in fails:
        print("  [self-test] FAIL %s" % f_, flush=True)
    if not fails:
        print("  [self-test] PASS all design-time checks", flush=True)
    return 1 if fails else 0


def main() -> Dict[str, Any]:
    ap = argparse.ArgumentParser(description="V3-EXQ-1044 hippocampal assay A")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    ap.add_argument("--scratch-out", type=str, default=None,
                    help="AUTHORING-TIME CONFIRMER ONLY: write the manifest to this directory "
                         "instead of evidence/experiments/ and write no runner sentinel. Never "
                         "set by the runner.")
    args = ap.parse_args()

    if args.self_test:
        sys.exit(_self_test())

    seeds = args.seeds if args.seeds else (DRY_RUN_SEEDS if args.dry_run else SEEDS)
    if _self_test() != 0:
        raise RuntimeError("design-time self-test failed; refusing to spend compute")

    result = run_experiment(seeds, dry_run=args.dry_run, scratch_out=args.scratch_out)
    print("")
    print("outcome: %s" % result["outcome"], flush=True)
    print("label: %s" % result["label"], flush=True)
    print("summary: %s" % result["summary"], flush=True)
    print("manifest: %s" % result["manifest_path"], flush=True)
    result["dry_run"] = bool(args.dry_run)
    result["scratch_out"] = args.scratch_out
    return result


if __name__ == "__main__":
    _result = main()
    _outcome_raw = str(_result["outcome"]).upper()
    if _result.get("scratch_out"):
        print("[scratch] authoring-time confirmer: no runner sentinel written", flush=True)
        sys.exit(0)
    emit_outcome(
        outcome=(_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL"),
        manifest_path=_result["manifest_path"],
        queue_id=QUEUE_ID,
        dry_run=_result["dry_run"],
    )
