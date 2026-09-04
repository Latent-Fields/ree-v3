# ==============================================================================
# REPAIR RECORD -- this driver went through TWO red-team passes before it was queued.
#
# PASS 1 (fable, 2026-09-04): BLOCKING. The then-criterion (AGREEMENT_BAR 0.50,
#   elevation over the MAJORITY class) was clearable without reading any
#   resource-field information: LocalViewGreedyPolicy repeats its previous action
#   ~57% of the time (0.5693/0.5837/0.5755 held-out, seeds 42/43/44) against a
#   ~0.249 majority baseline. REPAIRED: elevation is now measured over
#   max(majority, repeat-previous-executed-action), a `zworld_untrained` negative
#   control was added, and the bar was raised. The leak pathway that pass named
#   (body_state[5:9] one-hot last action -> z_self_init -> z_beta -> world_topdown)
#   is REAL but was measured INERT by pass 2 (0.677 -> 0.677 with body_state[5:9]
#   scrambled), and the untrained control carries it identically, so it cancels.
#
# PASS 2 (2026-09-04, CONTESTED, 5 findings; REE_assembly
#   evidence/planning/exq1002_redteam_findings_20260904.md). ALL FIVE APPLIED:
#   F1 the 0.70 bar was calibrated against a LINEAR untrained readout (0.590) while
#      the criterion uses the 2x128 MLP, which reaches 0.681-0.695 on the untrained
#      latent -- the bar sat ON the shortcut. FIXED: bar recalibrated against the
#      MLP figure (see AGREEMENT_BAR) and the effective threshold now stated.
#   F2 the verdict grid labelled a beats-untrained-but-below-bar result as H-C.
#      FIXED: `beats_untrained` and `untrained_clears` are carried into the grid;
#      H-C now requires NOT beats_untrained (see `_adjudicate`, contract-tested by
#      `--self-test`).
#   F3 headroom gate passes by 0.02-0.03. NOT silently loosened -- the measured
#      ratio is now stated in the check's own description.
#   F4 the differential criterion never consulted its COMPARATOR arm's gate.
#      Fixed in `non_degenerate_flags` -- but see PASS 3, which found that repair
#      INCOMPLETE and closed it properly.
#   F5 ARM_FEATURE_DIVERGENCE_EPS 1e-6 was below the latent's per-dim std. MOOT:
#      the `zworld_on` arm it served was dropped (see the ARMS section).
#
# PASS 3 (sonnet, 2026-09-04, MANDATORY cross-model re-review after the pass-2
# repairs): CONTESTED, no BLOCKING. It independently re-verified passes 1 and 2 at
# source (bar arithmetic exact; the Finding-2 grid confirmed by the boolean-cube
# invariant; F3 disclosed not loosened; F5 moot, confirmed by grep that no live path
# references the dropped arm) and raised ONE structural finding, APPLIED here:
#   P3-F1 the pass-2 F4 repair fixed the reported FLAG, not the OUTCOME it was meant
#      to protect. `beats_untrained` is computed from paired agreements alone, and
#      `_adjudicate` gated only on gate_green/verdict_arm_green -- so a COLLAPSED
#      untrained control could still score low, hand the verdict arm a free margin,
#      and resolve to PASS / H-B-consumer-learning while the co-emitted flag read
#      False. A reader taking `hypothesis_verdict` at face value would never see it.
#      FIXED STRUCTURALLY: `comparator_green` is now a THIRD readiness gate inside
#      `_adjudicate`, routing a red comparator to substrate_not_ready_requeue in
#      EITHER direction (it must not manufacture H-B via a free margin, nor H-C via
#      a vacuous `not beats_untrained`), and the non-degeneracy flag now reads the
#      SAME quantity so flag and verdict cannot disagree. Two named self-test rows
#      plus a second whole-cube invariant pin it.
#   P3-F2 (informational, NOT fixed, disclosed instead) the leak-inertness evidence
#      (body_state[5:9] scramble, 0.677 -> 0.677) was measured at DRY-SCALE warmup
#      (seed 42, 21 train eps), not at this run's 60+200+90 budget, and the live run
#      does not re-measure it. Accepted as-is: the pathway is architectural rather
#      than something training targets, and the untrained control carries the
#      identical pathway so it cancels in the differential. Named here so a future
#      reader does not mistake an imported offline probe for a run-scale measurement.
#
# The earlier INTEGRITY NOTE (feature-standardisation code appearing mid-review from
# an unattributed second agent) is DISCHARGED: that code was read line by line and
# re-authored by the redesign session campaign-c1a-redesign-20260904, which owns it.
# ==============================================================================
"""V3-EXQ-1002 -- z_world ACTOR-ADEQUACY LOCUS: can a CAPACITY-MATCHED supervised adapter
reproduce the local_view_greedy oracle's actions from the FROZEN z_world V3-EXQ-978 left?

Question (hypothesis_space_registry.v1.json qid `zworld_actor_adequacy_locus`):
  H-B-consumer-learning   the directional information in z_world is in a usable form and the
                          RL consumer simply failed to learn the mapping -> a simple SUPERVISED
                          reader CAN reproduce oracle actions.
  H-C-geometry-mismatch   the information is recoverable but the geometry does not make the
                          relationships action selection needs accessible -> a simple SUPERVISED
                          reader CANNOT reproduce oracle actions despite a strong generic decode.

Both are pre-registered ALIVE (Mode A, 2026-09-03). This run is their declared `live_gate`.

Claims: NONE. See "CLAIM LAYER" below -- this is deliberate, not an omission.

EXPERIMENT_PURPOSE = "diagnostic"

SLEEP DRIVER: not applicable -- no sleep flag is set by this driver (the x734 all-ON stack does
not enable use_sleep_loop / sws_enabled / rem_enabled / use_sleep_aggregation_cluster at this
rung). Recorded as sleep_driver_pattern="none" rather than omitted, so a reader can tell "no
sleep" from "nobody declared".

red-team (fable): see the queue entry note for V3-EXQ-1002.

=== WHY THIS RUN, AND WHY IT IS NOT A THIRD REPETITION ===

Routed by the confirmed autopsy `failure_autopsy_V3-EXQ-978_2026-09-03.md` sections 5-7 (a
GOV-FANOUT-1 `fanout_recommendation`), ratified by governance-20260903T2013 with one mandatory
amendment (see "THE CAPACITY MATCH" below). Chip: chip-20260903-exq978-oracle-adapter-discriminator.

V3-EXQ-978 established that the 25-dim `resource_field_view` IS linearly decodable from frozen
z_world -- held-out r2 0.70991 on the sense path, 0.85838 on the encoder path, in the arm whose
directional loss was OFF. That weakens the simple information-loss reading. It does NOT show the
information sits in a geometry a policy can USE. The user's thought document written against that
run states it verbatim: "V3-EXQ-978 does not distinguish B from C."

The frozen-latent reader has now been run TWICE with a PPO reader and failed both times
(V3-EXQ-948 `ppo_ree_latent` 0.5 res/ep; V3-EXQ-978 OFF arm 0.267 res/ep, both 0/3 against the
1.0 floor). Both failures are confounded with RL credit assignment. A BEHAVIOUR CLONE removes
exactly that confound: the supervision signal is the oracle's action at every visited state, so
there is no credit-assignment problem left to fail at. That is what makes this a DISCRIMINATOR
and not a third repetition, and it is why the re-derive brake (which FIRED on 978 -- MECH-457
carries 13 prior ceiling readings, INV-088 2) does not refuse it: the brake explicitly permits
"a redesign under a new EXQ number testing a different question", and this skill's own Step 2.5b
exempts "a `diagnostic` whose purpose is to discriminate WHY the ceiling holds".

It is also strictly cheaper than SD-018's pre-declared shape (b) fallback: no `ree_core` change
at all, hence no contract-gate exposure. And shape (b) PRESUPPOSES H-B -- if H-C holds it may
well raise the score while banking a confident-but-wrong localisation, which is the failure
GOV-FANOUT-1 exists to prevent. Discriminate first, build second.

=== CLAIM LAYER -- claim_ids IS DELIBERATELY EMPTY ===

V3-EXQ-978 tagged INV-088 and MECH-457 and its confirmed autopsy section 3 recorded BOTH as
PERIPHERAL CO-TAGS, `non_contributory`, with `recommended_epistemic_category_per_claim` set so
the re-derive-brake counter does not increment either claim from that run. This run tests neither
claim's mechanism either: it does not exercise MECH-457's actor-critic substrate, and INV-088's
own stated re-check condition is gated on a different substrate route
(SD-e1-rollout-consistency-training ITEM 2). Re-attaching them here would repeat exactly the
mis-tagging that autopsy corrected, and would increment two brake counters on a run that does not
bear on either claim.

So `claim_ids = []`. What this run adjudicates is a HYPOTHESIS-SPACE question, not a claim:
`zworld_actor_adequacy_locus`, whose two legs are named in `interpretation.question`. Governance
reads that, not a claim tag. `experiment_purpose = "diagnostic"` additionally excludes the run
from confidence/conflict scoring, which is the correct treatment for a discriminator.

=== ARMS (3) -- one positive control, one negative control, one verdict-bearing ===

  rawfield_ceiling  adapter reads the RAW 25-dim `resource_field_view`. POSITIVE CONTROL and
                    achievable ceiling. Gates the run (see PRECONDITIONS). NOT verdict-bearing.
  zworld_untrained  adapter reads z_world from the SAME agent construction with the warmup
                    SKIPPED ENTIRELY -- a random-projection latent. NEGATIVE CONTROL. NOT
                    verdict-bearing, but the verdict arm is scored AGAINST it.
  zworld_off        adapter reads frozen z_world from 978's OFF-arm warmup
                    (zworld_p0_resource_field_weight = 0.0). ** VERDICT-BEARING. **

WHY `zworld_off` CARRIES THE VERDICT. The puzzle this run exists to resolve was created by the
OFF arm: it is the arm whose absolute decode r2 (0.70991 / 0.85838) drove the autopsy's routing
finding, and it is the direct 948/813 replication anchor. Pinning the verdict to one
pre-registered arm also avoids a two-arms-two-chances multiplicity: "the adapter succeeded in
EITHER arm" would give H-B two independent shots at the bar.

WHY THE `zworld_on` ARM WAS DROPPED (2026-09-04, redesign decision -- it was present in the
first draft of this driver). It would have read the frozen z_world from 978's ON-arm warmup
(P0a resource-field weight 0.5), as a reported secondary asking whether SD-018's directional
supervision changed the geometry's ACCESSIBILITY even though 978 showed it changed neither
competence nor the decode r2. V3-EXQ-978 ALREADY ANSWERED THE PRECURSOR OF THAT QUESTION, in
the direction that makes the arm not worth a third full warmup family. Its confirmed autopsy
(`failure_autopsy_V3-EXQ-978_2026-09-03.md` section 2) records, on a fully green 8/8 gate with
the loss demonstrably applied and trained (`used_resource_field_head` true 3/3 at weight 0.5,
P0a held-out field r2 0.678 / 0.627 / 0.653):

    delta sense-path decode r2   -0.00063
    delta encoder-path decode r2 +0.00038
    delta participation ratio    +0.0029

-- "all two to three orders below the within-arm seed spread (~0.099 r2, ~1.61 PR) and
sign-inconsistent across seeds". The ON leg moves the latent by a quantity the seed noise
swamps. Three 350-episode warmups (60 P0a + 200 P0 + 90 P1, x 3 seeds) to re-measure a
foregone conclusion is exactly the compute the wave plan refuses, and the criterion the arm
fed (`C_sd018_supervision_changes_accessibility`) was reported-only and never gating, so
nothing load-bearing depends on it. The second red-team pass's Finding 5 -- that the arm's
own non-degeneracy epsilon (1e-6) sat below the latent's per-dimension std (0.005-0.010) and
so could not fire -- is a symptom of the same fact and is discharged by the drop rather than
by tuning the epsilon.

WHAT THE DROP COSTS, stated rather than glossed: if SD-018's supervision DID change
accessibility without changing decode r2 or competence, this run cannot see it. That
possibility is not refuted here -- it is deferred, and it is cheap to test later against THIS
run's banked OFF-arm cells (the config_slice is emitted with
`include_driver_script_in_hash=False` precisely so a different driver can reuse them).

=== THE CAPACITY MATCH (governance amendment, MANDATORY) ===

Red-team amendment 5 of the 2026-09-03 governance cycle: "978's oracle-adapter probe must be
capacity-matched to the policy head or it cannot separate H-B from H-C." The reasoning is exact.
H-B is a claim about what THE CONSUMER'S READOUT could have learned. A large MLP succeeding would
say only that SOME function of z_world reproduces the oracle -- which the r2 0.71 decode already
half-says -- and would mis-route H-B.

So the adapter is not a look-alike of the consumer's policy head; it IS the same class at the same
width, `x734.PPOPolicyNet`, the exact class V3-EXQ-978 instantiated as its reader
(`v3_exq_978...py`: `net = x734.PPOPolicyNet(in_dim=z_dim, action_dim=action_dim)`). The match is
therefore by CONSTRUCTION -- one import, one constant (`x734.PPO_TRUNK_HIDDEN`) -- not by an
arithmetic I could get wrong or that could silently drift when x734 changes.

  ARITHMETIC, at this rung (z_dim = 32, action_dim = 5, PPO_TRUNK_HIDDEN = 128):

    trunk  Linear(32, 128)    32*128 + 128  =  4,224
           Tanh
           Linear(128, 128)  128*128 + 128  = 16,512
           Tanh                               ------
                                   trunk    = 20,736
    policy_head Linear(128, 5)  128*5 + 5   =    645
                                              ------
      ACTION PATH (trunk + policy_head)      = 21,381   <- the mapping under test
    value_head  Linear(128, 1)  128*1 + 1   =    129    <- not part of the action mapping
                                              ------
                              module total   = 21,510

  978's consumer:  21,381 action-path parameters.
  zworld_off / zworld_untrained adapters:  21,381 -- IDENTICAL, same class, in_dim, width.
  rawfield_ceiling adapter:  in_dim = 25, so Linear(25,128) = 3,328 and the action path is
     3,328 + 16,512 + 645 = 20,485 -- 896 parameters (4.19%) FEWER than the z_world arms.

  That last asymmetry is forced by the narrower input and is in the CONSERVATIVE direction: the
  positive control is very slightly UNDER-powered relative to the arms it certifies, so a strong
  ceiling there is not bought with extra capacity. Verified empirically at authoring time and
  re-measured into the manifest at run time (`capacity_match` block) rather than trusted.

The value head receives no gradient: the adapter is trained by cross-entropy on `logits` alone.
It is left in place so the module is bit-identically the consumer's, rather than a trimmed
variant whose parameter count I would then have to argue about.

NO LARGER-CAPACITY ARM IS RUN. The amendment permits one as a REPORTED secondary that is never
verdict-bearing; it is omitted here because it cannot change the H-B/H-C routing (H-B is a claim
about the consumer's readout capacity specifically) and would add a third full warmup family.

=== WHAT IS HELD FIXED, AND WHY THAT IS THE WHOLE DESIGN ===

"Freeze z_world exactly as V3-EXQ-978 left it" is implemented by REPRODUCING 978's warmup rather
than loading a checkpoint -- 978 saved none. Every budget, env kwarg, rung, weighting, seed and
P0a field weight is IMPORTED from the same modules 978 imported (x734 / x808 / x724), never
redefined here, so this driver cannot drift from the run whose latent it is meant to freeze. The
warmup call is `x734._train_all_on_agent(...)` with the identical
`zworld_p0_resource_field_weight` split (0.0 OFF / 0.5 ON) and the identical
`zworld_p0_episodes` / `p0_episodes` / `p1_episodes`.

WHAT IS DROPPED: the 1000-episode PPO reader and the fishtank episode log. Neither is part of the
latent; both were 978's consumer and observable. Dropping the PPO stage IS the manipulation.

=== THE DV, AND WHY IT IS AGREEMENT RATHER THAN COMPETENCE ===

DV: `oracle_action_agreement` -- the fraction of HELD-OUT states on which the adapter's argmax
equals the action `local_view_greedy` took. This is the declared null's own statistic ("the
adapter cannot reproduce oracle ACTIONS above a pre-registered floor"), and it is the right one:
it measures the representation-to-action MAPPING directly, with no rollout dynamics, no
compounding error and no credit assignment in between. Rolled-out foraging competence of the
cloned policy is ALSO measured and reported (it is what connects the result to the 1.0 floor and
the oracle's 45.75 res/ep), but it is NOT verdict-bearing: a clone can have a high per-state
agreement and still drift off-distribution, and that would be a fact about rollout robustness,
not about whether the geometry supports the mapping.

DATA. Per seed, the oracle drives `BC_EPISODES` episodes and the FULL observation sequence is
stored. Every arm's features are then extracted from THE SAME STORED STEPS -- the raw field
directly, and z_world by replaying each stored episode through that arm's warmed agent from a
fresh `agent.reset()`. So the three arms are paired step-for-step by construction; nothing
depends on two separately-driven rollouts happening to coincide.

SPLIT: by EPISODE, not by step. Consecutive steps in a grid world are strongly correlated, so a
step-level (or prefix) split leaks the test set into training and inflates agreement. The first
`BC_TRAIN_FRAC` of episodes train; the remainder is held out and never touched.

SECONDARY STATE DISTRIBUTION. Agreement is ALSO measured on `BC_RANDOM_EPISODES` episodes driven
by a RANDOM policy (labels still the oracle's action at each state). This is the state
distribution 978's own decode probe used -- deliberately, so its r2 0.71 is directly comparable
-- and it reports whether any success is confined to the demonstrator's own visitation.

FEATURE STANDARDISATION (pre-registered; applied IDENTICALLY to all three arms). Each arm fits a
per-dimension mean/std ON ITS OWN TRAINING SPLIT and applies it to that arm's train, held-out and
random-state splits. Fitting on the training split alone is what keeps the held-out split held
out. The reason it is here at all: measured at authoring time, the frozen z_world's per-dimension
std is 0.006-0.012 while the raw resource_field_view's is 0.135-0.233, a ~15-20x scale gap. One
global `ADAPTER_LR` across two feature spaces that differ in scale by that much is not one
instrument, and THE POSITIVE CONTROL CANNOT DETECT THIS -- its own input is the well-scaled raw
field, so a green control certifies the instrument at the raw field's scale and says nothing
about the latent's. An unstandardised gradient adapter could therefore under-fit z_world for a
purely optimisation reason and the design would route that shortfall to H-C, which is exactly the
inference this run exists to make trustworthy. The correction is affine and per-dimension: it
cannot create linear decodability that was not already present, so it does not inflate H-B; it
only removes an alternative explanation for a null. UNSTANDARDISED agreement is retrained and
reported per arm as a SECONDARY readout (`oracle_action_agreement_unstandardised`), never entered
into any criterion -- the gap between the two IS the measured size of the scaling effect, which
makes the instrument change auditable rather than assumed benign. The rolled-out competence
secondary wraps the adapter in its fitted standardiser, so it is evaluated on the input
distribution it was trained on. `feature_standardisation` is in the cell config_slice: a cell
minted with it is not interchangeable with one minted without it.

=== PRE-REGISTERED BAR, AND THE RANGE IT SITS INSIDE (dv_headroom) ===

Six of the seven 2026-09-03 pending-review runs passed every precondition and still could not
discriminate, because the registered threshold lay outside the range the configuration could
produce. This driver declares the `dv_headroom` precondition class that landed for that
(ree-v3 8e133d26ed; `_metrics.dv_headroom_check`).

THE ACHIEVABLE RANGE, measured at authoring time from 978's own env and oracle (rung
D3_hazard_free, seeds 42/43/44, 12,000 labelled steps per state distribution):

    oracle action marginal, oracle-driven states:  {N .2533, S .2556, W .2434, E .2452, stay .0024}
    oracle action marginal, random-driven states:  {N .2672, S .2781, W .1941, E .2598, stay .0008}
    STATE-BLIND majority-class baseline:  0.2556 (oracle-driven) / 0.2781 (random-driven)
    uniform baseline:                     0.2000

THE MAJORITY-CLASS BASELINE IS NOT THE TRIVIAL CEILING, and reading it as one is the mistake
this driver was re-designed to remove. Re-measured 2026-09-04 at full BC scale on the SAME rung
and seeds (~2,000 held-out labelled steps per seed):

    majority-class baseline, held-out:            0.250 / 0.248 / 0.246
    REPEAT PREVIOUS EXECUTED ACTION, held-out:    0.568 / 0.582 / 0.573
    oracle label autocorrelation P(y_t=y_{t-1}):  0.582 / 0.571 / 0.566

CALIBRATE AGAINST THE INSTRUMENT THE CRITERION ACTUALLY USES. This is red-team pass 2's
Finding 1 and it is the single most important number on this page. An earlier revision of this
driver quoted the untrained-latent shortcut as 0.590 -- but that figure was measured with a
LINEAR readout, while the criterion is evaluated with `_make_adapter` (= `x734.PPOPolicyNet`, a
2x128 tanh MLP) trained for `ADAPTER_PASSES`. Re-measured with THE RUN'S OWN ADAPTER, at full
BC scale, on this rung and these seeds:

    UNTRAINED z_world -> oracle action, THIS ADAPTER, held-out:  0.688 / 0.681 / 0.695
    ...same, TURN states only:                                   0.609 / 0.616 / 0.623
    ...same, random-driven states:                               0.613 / 0.636 / 0.631
    ...same, TRAIN split:                                        0.828 / 0.828 / 0.821
    UNTRAINED z_world, LINEAR readout (the STALE figure):        0.590
    raw field, THIS ADAPTER, held-out / train:      0.985/0.997  0.980/0.992  0.973/0.986

A bar of 0.70 would therefore have sat ON the untrained shortcut, not above it, and would have
carried no shortcut protection at all.

The oracle is a local greedy walker, so its action PERSISTS while it walks toward a resource.
Two consequences, both of which invalidate an absolute bar set against the majority class:
(1) "repeat the previous executed action" scores ~0.57 with ZERO state-reading, and z_world
carries the previous action through the body/topdown path, so an adapter can clear a 0.50 bar
by proprioception alone (that pathway is real but MEASURED INERT here -- scrambling
`body_state[5:9]` moves held-out agreement 0.677 -> 0.677 -- and the untrained control carries
it identically, so it cancels in the differential); (2) even with that shortcut removed by
construction (scoring only the ~43% of held-out steps where the oracle TURNS), an UNTRAINED
random-projection z_world still reaches 0.61-0.62 under this adapter, because a random linear
projection preserves the field's decodable content.

THE CRITERIA THAT FOLLOW FROM THIS:
  * `AGREEMENT_BAR = 0.80`, calibrated as the MEASURED untrained-MLP ceiling (0.695, worst
    seed, this adapter, this dataset) plus `UNTRAINED_CONTROL_MARGIN`, rounded up. It sits
    above every measured shortcut -- trivial 0.582, untrained-MLP 0.695 -- and well below the
    raw-field control's 0.973-0.985 with this same adapter, so it stays inside a demonstrated
    achievable range.
  * `AGREEMENT_ELEVATION_MIN` is measured against `max(majority, repeat-previous-action)` -- the
    STRONGEST trivial predictor -- not against the majority class alone. Elevation over majority
    is retained as a secondary readout so the two remain comparable.
  * The `zworld_untrained` NEGATIVE CONTROL: same construction, warmup skipped, same data, same
    adapter, same standardiser, so it carries the same shortcuts and the verdict arm must beat
    it by `UNTRAINED_CONTROL_MARGIN`. This converts an absolute-threshold design -- which
    requires knowing every shortcut's ceiling in advance -- into a DIFFERENTIAL one, which does
    not, and it makes the design robust to this run's control landing away from 0.695.
  * A shortcut-free secondary, `oracle_action_agreement_turn_states`, is reported per arm.

THE EFFECTIVE PASS THRESHOLD, STATED HONESTLY. The three conjuncts are ANDed, so what the
verdict arm must actually reach on a seed is

    max( AGREEMENT_BAR,  trivial_baseline + AGREEMENT_ELEVATION_MIN,
                         untrained_control + UNTRAINED_CONTROL_MARGIN )

At the authoring-time measurements that is max(0.80, 0.57+0.20 = 0.77, 0.695+0.10 = 0.795)
= 0.80 per seed -- roughly 56% of the way from the trivial baseline (0.58) to the raw-field
ceiling (0.98). It is NOT "0.80 by a comfortable margin over a 0.57 shortcut"; the three
conjuncts nearly coincide at this operating point, BY DESIGN. The absolute bar's job is to hold
the floor if this run's untrained control comes out anomalously LOW (a low control would
otherwise hand the verdict arm a free margin); the differential's job is to raise the bar if the
control comes out HIGH. Each covers the other's failure mode, and neither is decorative. The run
computes this quantity per seed and records it as
`interpretation.effective_pass_threshold_per_seed`, so no reader has to reconstruct it.

These are CONSTANTS fixed before the run. The run re-measures the trivial baseline in its own
data and feeds it to `dv_headroom_check` as the control value, so a configuration whose oracle
marginal turned out skewed would fail the headroom precondition rather than pass a vacuous bar.

HEADROOM-GATE MARGIN, ALSO STATED HONESTLY (red-team pass 2, Finding 3). The elevation
headroom check declares `margin=2.0`, i.e. it requires 1.0 - trivial_baseline >= 0.40. The
measured headroom is 0.418 / 0.420 / 0.434 -- a ratio of 1.05-1.09, so the gate PASSES BY
0.02-0.03. A seed whose held-out split happened to realise persistence >= 0.60 would route the
whole run to `substrate_not_ready_requeue` on a property of the oracle's walk rather than of
any representation. That is an honest refusal, not a misattribution -- but the margin is
satisfied only nominally and the run is one unlucky split away from answering nothing. The
margin is deliberately NOT lowered to buy comfort: moving a pre-registered threshold to obtain
a pass is the failure the `dv_headroom` class exists to prevent.

=== NULL TABLE (declared up front -- BOTH directions are informative) ===

This is the property that makes the design a discriminator rather than a probe with one
interesting outcome, and it is the answer to GOV-FANOUT-1's objection to a single sequential leg:
the two live hypotheses are mutually exclusive readings of ONE measurement, and the measurement's
two outcomes map onto them symmetrically.

The grid is a function of THREE seed-majority booleans (>= SEED_MAJORITY of 3), evaluated
under THREE readiness gates (gate_green, verdict_arm_green, comparator_green) any one of which
routes the run to substrate_not_ready_requeue with no hypothesis verdict:

    off_clears        `zworld_off` reaches AGREEMENT_BAR *and* AGREEMENT_ELEVATION_MIN over its
                      own seed's strongest trivial predictor
    beats_untrained   `zworld_off` exceeds the PAIRED `zworld_untrained` agreement on the same
                      seed by >= UNTRAINED_CONTROL_MARGIN
    untrained_clears  `zworld_untrained` itself reaches the bar and the elevation

It is implemented as the pure function `_adjudicate()` and CONTRACT-TESTED by `--self-test`,
which pushes named synthetic rows (including red-team pass 2's own worked example,
off=0.74 / untrained=0.69 / trivial=0.57) through it and asserts the label. A verdict grid that
can only be exercised by a 3-hour run is a grid nobody checks.

- Gate not green (positive control below floor, or the DV has no headroom)
      -> `substrate_not_ready_requeue`, `undetermined`. NOT an H-C verdict. It says the
      instrument (this capacity, this dataset, this optimiser) cannot learn the mapping even
      from the raw field, so a z_world null would say nothing. Re-queue at a larger dataset or
      more adapter passes.
- `off_clears` AND `beats_untrained`
      -> **H-B**, `zworld_supports_oracle_mapping_h_b_consumer_learning`. The frozen latent DOES
      support the representation-to-action mapping at the consumer's own capacity, and the
      warmup is what put it there; the deficit is in the RL consumer's LEARNING. SD-018's shape
      (b) (side-channelling the raw field past z_world) is then addressing the wrong locus, and
      the next lever is the consumer/optimiser, not the representation.
- `off_clears` AND NOT `beats_untrained`
      -> `zworld_supports_mapping_but_warmup_non_contributory`, hypothesis verdict
      `H-B-leaning-warmup-non-contributory`. This cell is NOT a bare requeue and NOT
      `undetermined`, and red-team pass 2 Finding 2 is why. The REGISTERED live_gate for this
      question (hypothesis_space_registry `zworld_actor_adequacy_locus`) reads "CANNOT reproduce
      -> H-C; CAN reproduce -> H-B" -- and here the adapter DID reproduce the oracle from the
      frozen latent, at >= 0.80, far above every measured shortcut. So H-C is DISCONFIRMED and
      the H-C corroborator is NOT owed. What is NOT established is ATTRIBUTION: an untrained
      random projection of the same observation does about as well, so the actionability cannot
      be credited to SD-018's warmup. That is a real finding about the warmup, and the honest
      next step is an attribution question about SD-018, not a geometry probe. The
      pre-registered load-bearing criterion is still recorded as unmet (its differential
      conjunct failed), so `outcome` is FAIL -- outcome tracks the criterion, the interpretation
      carries the science, and this driver will not launder a partial result into a PASS.
- NOT `off_clears` AND `beats_untrained`
      -> `zworld_partially_supports_oracle_mapping_below_bar`, hypothesis verdict
      `undetermined-partial-support`. The warmup DID add actionability the untrained control
      does not have, but the absolute level falls short of the registered floor. This is the
      cell red-team pass 2 found the previous grid absorbing into H-C: a geometry that
      demonstrably improved is not a geometry that BLOCKS the mapping, and labelling it H-C
      would send the H-C corroborator after an effect pointing the other way. Neither
      hypothesis is adjudicated; the graded reading (how far short, against the raw-field
      ceiling) is what to report.
- NOT `off_clears` AND NOT `beats_untrained` AND `untrained_clears`
      -> `untrained_projection_clears_warmed_arm_does_not`, hypothesis verdict `undetermined`.
      Rare and worth its own cell: an untrained random projection of this observation reaches
      the bar while 978's warmed latent does not. H-C is REFUTED for the observation at this
      dimensionality (a 32-dim projection plainly supports the mapping) and the finding is that
      the warmup DEGRADED actionability -- a statement about SD-018's recipe, not about
      geometry in general. Do not queue the H-C corroborator on this cell.
- NOT `off_clears` AND NOT `beats_untrained` AND NOT `untrained_clears`
      -> **H-C**, `zworld_geometry_blocks_oracle_mapping_h_c_geometry_mismatch`. The information
      is present (978: r2 0.71) and linearly decodable, and a reader with the consumer's exact
      capacity cannot turn it into the oracle's actions from the SAME states -- with credit
      assignment removed, with the latent's ~15-20x scale disadvantage removed by train-split
      z-scoring, and with NEITHER the warmed nor the untrained latent clearing the bar while the
      raw field reaches 0.97+. The last conjunct is what makes this attributable to the geometry
      rather than to "nothing at 32 dimensions works": it is reported, and the H-C label is
      withheld without it. If the SECONDARY unstandardised agreement is far below the primary on
      this arm, say so in the write-up: it means the scale gap was doing real work and the
      pre-fix design would have mis-routed. The H-C corroborator (an information-preserving
      rotation/reweighting) THEN BECOMES OWED -- it is deliberately NOT queued by this session,
      per the autopsy's routing and the campaign brief.

A null here is therefore a result, not a wasted run, on every branch.

=== DV-SYMMETRY INVARIANCE (declared per arm, per the skill's mandatory check) ===

DV = the mean over held-out states of an indicator, 1 iff argmax(adapter logits at that state)
equals the oracle's action at that state. Its symmetry group is: permutation of the held-out
states (the mean is a symmetric function of per-state indicators), and any relabelling of the
action index set applied CONSISTENTLY to both the adapter's output and the oracle's label.

- rawfield_ceiling: the manipulation is the FEATURE VECTOR the adapter reads (raw 25-dim field).
  It is not a broadcast constant added to the logits (which an argmax would annihilate), not a
  monotone rescaling of a ranked quantity, and not a permutation of interchangeable units: it
  changes the learned per-state function, hence the per-state argmax, hence the indicator. Path
  open; not invariant. (This arm is a positive control and is not scored for a verdict.)
- zworld_off: same structure -- the feature vector is that arm's frozen z_world. The manipulation
  relative to the control is a change of INPUT REPRESENTATION, which cannot be absorbed by an
  argmax because it re-orders the logits state-by-state rather than shifting them uniformly.
  Path open; not invariant.
- zworld_untrained: identical to zworld_off, differing only in whether the encoder ever received
  a gradient. The warmup changes `world_encoder` WEIGHTS, hence z_world, hence the adapter's
  per-state input, hence its argmax. Path open; not invariant. (Negative control; not scored for
  a verdict, but it IS the comparator of a load-bearing criterion, so its path must be open too
  -- a comparator whose manipulation the DV could not see would make the differential vacuous.)

Corollary (per the same rule -- state what each readiness precondition actually certifies):
`adapter_capacity_sufficient_on_raw_field` certifies the ADAPTER + DATASET + OPTIMISER, i.e. the
instrument, for every arm; it says nothing about any encoder. `zworld_encoder_trained_in_p0` and
`zworld_not_collapsed` certify only the z_world arms and are scoped OUT of the raw-field arm,
which has no encoder at all. `oracle_labels_non_degenerate` and `heldout_split_sufficient`
certify the shared dataset and therefore speak for all three arms.

=== KNOWN OPEN SUBSTRATE DEFECTS OVERLAPPING THIS DRIVER (skill Step 2.5c) ===

SD-018 itself is `amend_implemented_pending_validation`, severity DEGRADING, with
`substrate_paths` ree_core/latent/stack.py + zworld_p0.py + agent.py::compute_resource_proximity_loss
-- all exercised by the warmup. Recorded, not blocking (degrading), and note that reproducing
978's warmup DEFECTS INCLUDED is the design requirement here, not a hazard to be avoided: the
whole point is that this is the latent 978 produced. Several open CORRUPTING entries list
`ree_core/agent.py` or `ree_core/utils/config.py` (mode-governance-engagement, SD-082), which
every driver in the corpus imports; each names a mechanism -- affinity-input clamping in the
salience coordinator, a lateral-PFC bias head -- that this run's DV does not read, since the DV
is a supervised decode of a frozen latent and never consults action selection. Held constant
across all three arms by construction.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from experiment_protocol import emit_outcome  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402
from experiments._metrics import (  # noqa: E402
    P0NotReady,
    dv_headroom_check,
    p0_readiness_gate,
)
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from experiments._lib.capability_eval import (  # noqa: E402
    COMPETENCE_RESOURCE_FLOOR,
    LocalViewGreedyPolicy,
    Policy,
    RandomPolicy,
    evaluate_seed,
)
from experiments._lib.precondition_gate import (  # noqa: E402
    PreconditionSpec,
    aggregate_arm_gates,
    arm_criteria_non_degenerate,
    assert_no_structurally_unsatisfiable_gate,
    evaluate_arm_gate,
)
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator  # noqa: E402
from experiments._lib.zworld_encoder_guard import (  # noqa: E402
    latent_stack_snapshot,
    latent_stack_weight_delta,
)
import experiments.v3_exq_724_competence_localization_diagnostic as x724  # noqa: E402
import experiments.v3_exq_734_env_difficulty_competence_recovery_sweep as x734  # noqa: E402
import experiments.v3_exq_737_ree_latent_policy_head_competence_probe as x737  # noqa: E402
import experiments.v3_exq_808_return_decomposition_objective_misspecification as x808  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_1002_zworld_actor_adequacy_oracle_adapter"
QUEUE_ID = "V3-EXQ-1002"
EXPERIMENT_PURPOSE = "diagnostic"
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"

# Deliberately EMPTY -- see the docstring's CLAIM LAYER section. This run adjudicates a
# hypothesis-space question, not a claim; INV-088 / MECH-457 were recorded as peripheral
# co-tags by the V3-EXQ-978 autopsy and re-attaching them here would repeat that error.
CLAIM_IDS: List[str] = []

HYPOTHESIS_QID = "zworld_actor_adequacy_locus"

DEVICE = torch.device("cpu")

# Same seeds as 978 / 948 / 813, so each z_world arm is a per-seed reproduction of that lineage.
# Not a reef config at this rung, so the seed-44 reef-instability rule does not apply.
SEEDS: List[int] = [42, 43, 44]

# ---- IMPORTED, NEVER REDEFINED: this is what makes the frozen latent 978's latent ----------
ZWORLD_P0_EPISODES = x734.ZWORLD_P0_EPISODES        # 60
P0_WARMUP_EPISODES = x734.P0_WARMUP_EPISODES        # 200
P1_REINFORCE_EPISODES = x734.P1_REINFORCE_EPISODES  # 90
EVAL_EPISODES = x734.EVAL_EPISODES                  # 20
STEPS_PER_EPISODE = x734.STEPS_PER_EPISODE          # 200

RUNG = x734.DIFFICULTY_RUNGS[-1]                    # D3_hazard_free
RUNG_ID = RUNG["rung_id"]
_W3 = next(w for w in x808.WEIGHTINGS if w["id"] == "W3_survival_zeroed")
LEVEL_ID = str(_W3["id"])

# 978's ON-arm P0a directional-field weight. 0.0 on the OFF arm. Imported semantics, same value.
P0A_FIELD_WEIGHT = 0.5
ZWORLD_DELTA_FLOOR = x808.ZWORLD_DELTA_FLOOR        # 1e-6
PARTICIPATION_RATIO_FLOOR = 2.0

LOCALFIELD_KEY = "resource_field_view"
RESOURCE_FIELD_DIM = 25

ARM_RAW = "rawfield_ceiling"
ARM_UNTRAINED = "zworld_untrained"   # NEGATIVE CONTROL -- see the docstring's ARMS section
ARM_OFF = "zworld_off"
# NOTE: a `zworld_on` arm (978's ON-arm warmup, P0a field weight 0.5) was present in the first
# draft and was DROPPED 2026-09-04. 978's confirmed autopsy section 2 measured the ON leg's
# effect on the latent at two to three orders BELOW the within-arm seed spread and
# sign-inconsistent across seeds; three more 350-episode warmups for that is compute the wave
# plan refuses. Full rationale, and what the drop costs, in the docstring's ARMS section.
ARM_IDS = [ARM_RAW, ARM_UNTRAINED, ARM_OFF]
VERDICT_ARM = ARM_OFF

# --------------------------------------------------------------------------------------
# PRE-REGISTERED THRESHOLDS (constants; never derived from this run's own statistics)
# --------------------------------------------------------------------------------------
# Held-out top-1 agreement the verdict arm must reach.
#
# RE-CALIBRATED TWICE. Read both steps -- the second one is the load-bearing correction.
#
# STEP 1 (2026-09-04, red-team pass 1): the ORIGINAL 0.50 sat BELOW two measured shortcuts:
#   (1) "repeat the previous executed action" scores 0.568-0.582 held-out -- the oracle is a
#       local greedy walker, so its actions are strongly autocorrelated. The state-blind
#       MAJORITY-CLASS baseline (0.246-0.250) does NOT detect this: it understates the trivial
#       predictor ceiling by more than 30 points.
#   (2) an UNTRAINED z_world (random-init encoder, zero warmup) supports 0.590 held-out.
# That step raised the bar to 0.70 and added the elevation + untrained-control criteria.
#
# STEP 2 (2026-09-04, red-team pass 2, Finding 1 -- THE CORRECTION): shortcut (2)'s 0.590 was
# measured with a LINEAR readout, but this criterion is evaluated with `_make_adapter` (=
# `x734.PPOPolicyNet`, a 2x128 tanh MLP) trained for ADAPTER_PASSES. Re-measured with THE RUN'S
# OWN ADAPTER at full BC scale on this rung and these seeds, the untrained latent supports
#   0.688 / 0.681 / 0.695  held-out
# -- so a 0.70 bar sat ON the shortcut, not above it, and carried NO shortcut protection.
#
# 0.80 = the measured untrained-MLP ceiling (0.695, worst seed) + UNTRAINED_CONTROL_MARGIN,
# rounded up. It sits above every measured shortcut (trivial 0.582, untrained-MLP 0.695) and
# well below the raw-field control's 0.973-0.985 WITH THIS SAME ADAPTER, so it stays inside a
# demonstrated achievable range. See the docstring's "THE EFFECTIVE PASS THRESHOLD, STATED
# HONESTLY" block for what the three ANDed conjuncts actually require per seed (~0.80), and why
# they nearly coincide at this operating point by design.
AGREEMENT_BAR = 0.80

# ...AND it must beat this run's OWN strongest TRIVIAL predictor by this margin. The baseline is
# max(state-blind majority class, repeat-previous-action) -- NOT majority class alone, which is
# what made shortcut (1) invisible. Measured trivial ceiling 0.566-0.582, so 0.20 over it is
# 0.766-0.782 -- just BELOW the 0.80 absolute bar at this operating point, which is why the
# absolute bar is the binding conjunct here and the elevation is the guard against a run whose
# trivial baseline came out unexpectedly HIGH.
AGREEMENT_ELEVATION_MIN = 0.20

# ...AND it must beat the UNTRAINED negative-control arm by this margin, PAIRED PER SEED. The
# negative control is the same agent construction with the warmup SKIPPED, so the difference
# isolates what the warmup actually added to the latent's actionability. This is the criterion
# that makes the design robust to THIS run's untrained control landing away from the
# authoring-time 0.688-0.695: the absolute bar holds the floor if the control comes out LOW, the
# margin raises it if the control comes out HIGH. It is ALSO carried into the verdict grid on
# the FAIL side (red-team pass 2, Finding 2): H-C now REQUIRES `not beats_untrained`, so a
# verdict arm that demonstrably beat its own untrained control is never labelled
# "the geometry blocks the mapping".
UNTRAINED_CONTROL_MARGIN = 0.10

# POSITIVE CONTROL floor. The same capacity-matched adapter, same dataset, same optimiser, fed
# the raw 25-dim field, must reach this. Below it the INSTRUMENT is inadequate and a z_world
# null is uninterpretable -> substrate_not_ready_requeue, never an H-C verdict.
RAW_FIELD_CONTROL_FLOOR = 0.60

# The oracle's own action marginal must not be dominated by one action, or agreement is a
# measure of label skew rather than of state-dependence. CEILING (direction "upper").
# Measured at authoring time: 0.2556 oracle-driven / 0.2781 random-driven.
ORACLE_MAJORITY_CEILING = 0.60

# Minimum held-out labelled steps, per seed, for the agreement estimate to mean anything.
HELDOUT_MIN_STEPS = 500

# NOTE: `ARM_FEATURE_DIVERGENCE_EPS` (1e-6) was REMOVED with the `zworld_on` arm. It gated the
# non-degeneracy flag on `C_sd018_supervision_changes_accessibility`, and red-team pass 2's
# Finding 5 measured it BELOW the latent's own per-dimension std (0.005-0.010), so the check
# could not fire -- an OFF/ON delta of 5.4e-5 (0.8% of one feature std) with
# `mean_argmax_disagreement_frac = 0.0` still read as non-degenerate. Both the criterion and its
# epsilon are gone with the arm; see the docstring's ARMS section for why the arm was dropped.

# ---- behaviour-cloning dataset and adapter schedule ----------------------------------------
BC_EPISODES = 40            # oracle-driven episodes per seed (the demonstrator distribution)
BC_RANDOM_EPISODES = 20     # random-driven episodes per seed (the 978 decode-probe distribution)
BC_TRAIN_FRAC = 0.7         # split by EPISODE, never by step (correlation leak)
ADAPTER_PASSES = 60         # full passes over the training split
ADAPTER_BATCH = 256
ADAPTER_LR = 1e-3           # supervised CE; the consumer's PPO_LR (3e-4) is an RL setting and
                            # the optimiser difference IS the manipulation under test
SEED_MAJORITY = 2           # of 3 seeds

# ---- feature standardisation (TRAIN-SPLIT z-scoring, applied IDENTICALLY to every arm) ------
# Measured at authoring time on this rung: the frozen z_world's per-dimension std is 0.006-0.012
# while the raw resource_field_view's is 0.135-0.233 -- a ~15-20x scale gap. A single global
# ADAPTER_LR against a gradient-trained adapter therefore does NOT mean the same effective step
# size in the two feature spaces, and the positive control CANNOT catch it: the control's own
# input is the well-scaled raw field, so it certifies the instrument only at that scale. Without
# this, a pure OPTIMISATION shortfall on a badly-scaled input would be indistinguishable from the
# representational shortfall H-C asserts, and the run would route a scaling artefact to H-C.
# The fix is affine and per-dimension, so it is information-preserving by construction (it cannot
# create linear decodability that was not already there) and it does not weaken the H-C claim:
# it removes the one alternative explanation that a null would otherwise carry.
STANDARDISE_FEATURES = True
STANDARDISER_EPS = 1e-6     # floor on a fitted per-dim std, so a constant dimension is passed
                            # through as (x - mean) rather than amplified to +-inf

DRY_RUN_SEEDS = [42]
DRY_RUN_ZWORLD_P0 = x734.DRY_RUN_ZWORLD_P0
DRY_RUN_P0 = 2
DRY_RUN_P1 = 2
DRY_RUN_EVAL = 2
DRY_RUN_STEPS = 20
DRY_RUN_BC_EPISODES = 6
DRY_RUN_BC_RANDOM_EPISODES = 4
DRY_RUN_ADAPTER_PASSES = 3

# Each cell builds a fresh agent, so a run-level list would keep every arm x seed agent alive
# until the last cell finished. The accumulator reads the counters at observe() time.
_ZG = ZGoalStreamAccumulator()


# --------------------------------------------------------------------------------------
# PRE-REGISTERED PRECONDITIONS
# --------------------------------------------------------------------------------------
def _arm_contexts() -> List[Dict[str, Any]]:
    """Arm context consumed by the precondition specs' `applies_to`.

    `trained_encoder` is deliberately distinct from `has_encoder`: the NEGATIVE CONTROL arm
    HAS an encoder (it is the same agent construction) but never receives a gradient, so the
    P0-weight-delta precondition would fail it BY CONSTRUCTION. That is disposition (a) in the
    skill's precondition-gate rule -- the precondition is NOT MEANINGFUL for that regime, so it
    is scoped out; the arm stays fully scorable, which it must be, since the verdict arm is
    scored AGAINST it.
    """
    return [{"id": aid,
             "has_encoder": (aid != ARM_RAW),
             "trained_encoder": (aid not in (ARM_RAW, ARM_UNTRAINED))}
            for aid in ARM_IDS]


PRECONDITION_SPECS = [
    PreconditionSpec(
        name="adapter_capacity_sufficient_on_raw_field",
        description=(
            "POSITIVE CONTROL. The capacity-matched adapter, on the same dataset with the same "
            "optimiser and passes, must reproduce the oracle from the RAW 25-dim field. This is "
            "the readiness assert for the load-bearing criterion and it reports the SAME "
            "statistic that criterion routes on (held-out top-1 agreement), on the worst seed. "
            "Below floor the instrument cannot learn the mapping from information handed to it "
            "directly, so a z_world null is uninterpretable and the run self-routes "
            "substrate_not_ready_requeue rather than claiming H-C."),
        control="rawfield_ceiling worst-seed held-out oracle_action_agreement",
        threshold=float(RAW_FIELD_CONTROL_FLOOR),
        direction="lower",
        kind="readiness",
        structural_max=lambda ctx: 1.0,
    ),
    PreconditionSpec(
        name="oracle_labels_non_degenerate",
        description=(
            "The oracle's action marginal must not be dominated by a single action, or "
            "agreement measures label skew rather than state-dependence and a state-blind "
            "predictor could clear the bar. CEILING: measured is the worst (largest) "
            "majority-class share across seeds. Authoring-time measurement 0.2556."),
        control="worst-seed majority-class share of the oracle's action on the training split",
        threshold=float(ORACLE_MAJORITY_CEILING),
        direction="upper",
        kind="readiness",
        structural_min=lambda ctx: 0.2,
    ),
    PreconditionSpec(
        name="heldout_split_sufficient",
        description=(
            "Enough held-out labelled steps, on the worst seed, for the agreement estimate to "
            "resolve the pre-registered elevation. Guards the case where short episodes (early "
            "termination) shrink the test split below usefulness."),
        control="worst-seed held-out labelled step count on the oracle-driven split",
        threshold=float(HELDOUT_MIN_STEPS),
        direction="lower",
        kind="readiness",
    ),
    PreconditionSpec(
        name="zworld_encoder_trained_in_p0",
        description=(
            "The P0a SD-070 recipe must actually move split_encoder.world_encoder, or the "
            "'frozen z_world' the adapter reads is a random projection and the run measures "
            "nothing about 978's latent. Same guard and floor the family uses."),
        control="latent_stack weight delta over the warmup, vs the family's guard floor",
        threshold=float(ZWORLD_DELTA_FLOOR),
        direction="lower",
        kind="readiness",
        applies_to=lambda ctx: bool(ctx["trained_encoder"]),
        applies_note=(
            "rawfield_ceiling reads the raw observation channel and warms no encoder at all, "
            "so it has no P0 weight delta to certify; zworld_untrained is the NEGATIVE CONTROL "
            "whose encoder is deliberately never trained, so a trained-encoder precondition is "
            "not meaningful for it (scoping the PRECONDITION out, not the arm)"),
    ),
    PreconditionSpec(
        name="zworld_not_collapsed",
        description=(
            "z_world must retain more than one effective dimension. SD-018's own ML note names "
            "25-dim-target-dominates-P0 collapse as its hazard and the substrate has measured "
            "participation ratio 9.21 -> 1.06 under a related recipe. A collapsed arm is an "
            "operating-point failure, NOT evidence that the geometry is unusable -- which is "
            "exactly the H-C conclusion this run must not reach by accident."),
        control="participation ratio of frozen z_world over that arm's own BC feature matrix",
        threshold=float(PARTICIPATION_RATIO_FLOOR),
        direction="lower",
        kind="readiness",
        applies_to=lambda ctx: bool(ctx["has_encoder"]),
        applies_note="rawfield_ceiling has no latent whose dimensionality could collapse",
    ),
    PreconditionSpec(
        name="d3_local_view_greedy_clears_floor",
        description=(
            "The demonstrator must be worth cloning: local_view_greedy must clear the 1.0 "
            "competence floor FROM THE SAME 5x5 field the adapter reads, closing the "
            "V3-EXQ-732a privileged-oracle confound. 978 measured 45.75 / 49.7 / 48.7."),
        control="local_view_greedy worst-seed mean resources/ep vs the 1.0 competence floor",
        threshold=float(COMPETENCE_RESOURCE_FLOOR),
        direction="lower",
        kind="readiness",
    ),
]


# --------------------------------------------------------------------------------------
# HELPERS
# --------------------------------------------------------------------------------------
def _mean(vals: List[float]) -> float:
    return float(sum(vals) / len(vals)) if vals else 0.0


def _worst_cell(rows: List[Dict[str, Any]], key: str, mode: str = "min"
                ) -> Tuple[float, Optional[str]]:
    """Worst value of `key` across rows, plus the offending cell id.

    The indexer recomputes `met` from the reported number, so a precondition whose `met` is a
    worst-case claim MUST report the worst cell, not the mean -- an in-band mean masks an
    out-of-band cell (V3-EXQ-779b `tonic_axis_live`).
    """
    vals = [(float(r[key]), r.get("cell_id")) for r in rows if r.get(key) is not None]
    if not vals:
        return (0.0, None)
    return min(vals) if mode == "min" else max(vals)


def _localfield_vector(obs_dict: Dict[str, Any]) -> torch.Tensor:
    """The 25-dim agent-centred resource gradient, shaped [1, 25].

    Fails LOUDLY: a missing channel (use_proxy_fields=False) would silently turn the whole
    probe into a zero-information control whose null would read as an H-C verdict.
    """
    v = obs_dict.get(LOCALFIELD_KEY)
    if v is None:
        raise KeyError(
            "obs_dict has no %r -- this driver requires use_proxy_fields=True. Without it the "
            "oracle degrades to random and every readout is vacuous." % (LOCALFIELD_KEY,))
    t = v if isinstance(v, torch.Tensor) else torch.as_tensor(v)
    return t.reshape(1, -1).float().to(DEVICE)


def _participation_ratio(z: torch.Tensor) -> float:
    """PR = (sum eig)^2 / sum(eig^2) over the covariance spectrum. 0.0 on degenerate input,
    which correctly FAILS the floor rather than raising."""
    if z.ndim != 2 or z.shape[0] < 2:
        return 0.0
    zc = z - z.mean(dim=0, keepdim=True)
    cov = (zc.T @ zc) / float(zc.shape[0] - 1)
    eig = torch.linalg.eigvalsh(cov).clamp(min=0.0)
    s1 = float(eig.sum())
    s2 = float((eig ** 2).sum())
    if s2 <= 0.0 or s1 <= 0.0:
        return 0.0
    return float((s1 * s1) / s2)


def _make_agent(env):
    """The x734 all-ON stack, built EXACTLY as V3-EXQ-978 built it.

    EVERY z_world arm CONSTRUCTS the directional head, at the same P0a loss weight 978 used --
    978's own F1 correction, and it must be preserved here or the frozen latents stop being
    978's. (The `zworld_off` arm then trains with a P0a field weight of 0.0 and the untrained
    control never trains at all; the head's EXISTENCE, not its weight, is what this is about.) Building `resource_field_head` consumes torch RNG draws and SplitEncoder builds it
    BEFORE world_topdown / self_topdown, so an arm pair differing in whether the head EXISTS
    would also differ in the random init of every module built after it -- modules that are
    never trained yet feed sense()-time z_world, which is precisely the latent this adapter
    reads.
    """
    kwargs = x724._base_config_kwargs(env)
    kwargs.update(x724._all_on_extra_kwargs())
    kwargs["use_resource_field_head"] = True
    kwargs["resource_field_dim"] = RESOURCE_FIELD_DIM
    kwargs["resource_field_weight"] = P0A_FIELD_WEIGHT
    cfg = x724.REEConfig.from_dims(**kwargs)
    return x724.REEAgent(cfg)


def _off_path_config_slice(dry_run: bool, zworld_p0: int, p0: int, p1: int, steps: int,
                           bc_eps: int, bc_rand: int, passes: int,
                           eval_eps: int) -> Dict[str, Any]:
    """The resolved config a cell's computation reads.

    Declared explicitly (config_slice_declared=True) AND emitted with
    include_driver_script_in_hash=False, so a future consumer -- the H-C corroborator is the
    obvious one, since it wants THIS frozen latent -- can reconstruct it from a DIFFERENT
    driver and match this mint's fingerprint.

    IT MUST BE A CONSERVATIVE SUPERSET OF WHAT THE CELL READS, and that is why the adapter and
    dataset schedule are in here alongside the warmup schedule. Reuse is WHOLE-CELL: a consumer
    that hit one of these cells would read back its agreement readouts, and those depend on
    ADAPTER_LR / ADAPTER_BATCH / the pass count / the BC episode counts / the split fraction
    just as much as on the encoder warmup. Omitting them would be a false-cache-HIT bug
    (arm_reuse_fingerprint_plan.md 7b, confirmed instance V3-EXQ-798) -- a false MISS only
    wastes compute, a false HIT corrupts a conclusion.

    Acceptance thresholds (AGREEMENT_BAR, AGREEMENT_ELEVATION_MIN, RAW_FIELD_CONTROL_FLOOR,
    ORACLE_MAJORITY_CEILING, HELDOUT_MIN_STEPS, SEED_MAJORITY) are deliberately EXCLUDED: they
    are applied to the readouts after the cell has computed them and change no recorded value.
    """
    return {
        "env_kwargs": x734._env_kwargs_for_rung(RUNG),
        "rung_id": RUNG_ID,
        "level_id": LEVEL_ID,
        "zworld_p0_episodes": int(zworld_p0),
        "p0_warmup_episodes": int(p0),
        "p1_reinforce_episodes": int(p1),
        "steps_per_episode": int(steps),
        "p0a_field_weight_on": float(P0A_FIELD_WEIGHT),
        "use_resource_field_head": True,
        "resource_field_dim": int(RESOURCE_FIELD_DIM),
        # dataset the adapter is fitted and scored on
        "bc_episodes": int(bc_eps),
        "bc_random_episodes": int(bc_rand),
        "bc_train_frac": float(BC_TRAIN_FRAC),
        # adapter optimisation -- every recorded agreement depends on these
        "adapter_passes": int(passes),
        "adapter_batch": int(ADAPTER_BATCH),
        "adapter_lr": float(ADAPTER_LR),
        "adapter_trunk_hidden": int(x734.PPO_TRUNK_HIDDEN),
        # feature preprocessing -- changes every recorded agreement, so it is a cache key.
        # A consumer that reused a cell minted WITHOUT standardisation against a driver that
        # standardises would be reading agreements computed on a different input distribution:
        # a false HIT, the corrupting direction (arm_reuse_fingerprint_plan.md 7b).
        "feature_standardisation": ("train_split_zscore" if STANDARDISE_FEATURES else "none"),
        "standardiser_eps": float(STANDARDISER_EPS),
        # rolled-out competence secondary
        "eval_episodes": int(eval_eps),
        "dry_run": bool(dry_run),
    }


# --------------------------------------------------------------------------------------
# FEATURE STANDARDISATION -- fitted on the TRAIN SPLIT ONLY, applied identically to all arms
# --------------------------------------------------------------------------------------
def _fit_standardiser(x_tr: torch.Tensor) -> Dict[str, Any]:
    """Fit per-dimension mean/std ON THE TRAINING SPLIT ONLY.

    Fitting on the training split alone is what keeps the held-out split held out: a
    standardiser fitted on train+test would leak the test set's first two moments into the
    features the adapter is scored on. The SAME procedure runs for every arm -- it is not a
    z_world-only correction -- so no arm gains a treatment the others did not get, and the
    between-arm comparison stays a comparison of representations rather than of preprocessing.

    Returns the fitted statistics plus the raw per-dimension spread that motivated the fix, so
    the scale gap is a recorded number in the manifest rather than an authoring-time claim.
    """
    n = int(x_tr.shape[0])
    if n == 0:
        return {"fitted": False, "mean": None, "std": None, "n_train_rows": 0,
                "raw_per_dim_std_min": None, "raw_per_dim_std_median": None,
                "raw_per_dim_std_max": None, "n_dims_at_eps_floor": None,
                "eps": float(STANDARDISER_EPS)}
    mu = x_tr.mean(dim=0, keepdim=True)
    sd_raw = x_tr.std(dim=0, unbiased=False, keepdim=True)
    sd = torch.clamp(sd_raw, min=float(STANDARDISER_EPS))
    flat = sd_raw.reshape(-1)
    return {
        "fitted": True,
        "mean": mu,
        "std": sd,
        "n_train_rows": n,
        "n_dims": int(x_tr.shape[1]),
        "raw_per_dim_std_min": float(flat.min().item()),
        "raw_per_dim_std_median": float(flat.median().item()),
        "raw_per_dim_std_max": float(flat.max().item()),
        "n_dims_at_eps_floor": int((flat < float(STANDARDISER_EPS)).sum().item()),
        "eps": float(STANDARDISER_EPS),
    }


def _apply_standardiser(x: torch.Tensor, st: Dict[str, Any]) -> torch.Tensor:
    """Apply a fitted standardiser. A no-op when nothing was fitted (empty train split)."""
    if not st.get("fitted") or st.get("mean") is None:
        return x
    if int(x.shape[0]) == 0:
        return x
    return (x - st["mean"]) / st["std"]


def _standardiser_report(st: Dict[str, Any]) -> Dict[str, Any]:
    """The manifest-safe view of a fitted standardiser (drops the tensors)."""
    return {k: v for k, v in st.items() if k not in ("mean", "std")}


class _StandardisedNet:
    """The adapter with its train-split standardiser welded on the front.

    The rolled-out competence secondary hands the policy RAW features at every step (both
    `RawFieldAdapterPolicy` and `x737.LatentPPOEvalPolicy` compute the feature vector from the
    live observation), so an adapter trained on standardised features MUST be wrapped before it
    is rolled out or it would be evaluated on an input distribution it never saw -- which would
    corrupt the secondary readout while the primary one stayed correct, the hardest kind of
    inconsistency to notice. Duck-typed on the `(logits, value)` call signature the two eval
    policies use; nothing else about them needs to know.
    """

    def __init__(self, net, st: Dict[str, Any]) -> None:
        self.net = net
        self._st = st

    def __call__(self, x: torch.Tensor):
        return self.net(_apply_standardiser(x, self._st))


# --------------------------------------------------------------------------------------
# THE ADAPTER -- capacity-matched to 978's consumer BY CONSTRUCTION
# --------------------------------------------------------------------------------------
def _make_adapter(in_dim: int, action_dim: int):
    """`x734.PPOPolicyNet` -- literally the class V3-EXQ-978 instantiated as its reader.

    Not a look-alike. Using the same class at the same `x734.PPO_TRUNK_HIDDEN` makes the
    capacity match hold by construction and keeps holding if x734 ever changes, rather than
    depending on an arithmetic transcribed into this file. The value head is left in place and
    simply receives no gradient (training is cross-entropy on `logits` only), so the module is
    bit-identically the consumer's rather than a trimmed variant.
    """
    return x734.PPOPolicyNet(in_dim=int(in_dim), action_dim=int(action_dim)).to(DEVICE)


def _capacity_report(net, in_dim: int, action_dim: int) -> Dict[str, Any]:
    """Measure the capacity match into the manifest rather than asserting it in prose."""
    trunk = int(sum(p.numel() for p in net.trunk.parameters()))
    phead = int(sum(p.numel() for p in net.policy_head.parameters()))
    vhead = int(sum(p.numel() for p in net.value_head.parameters()))
    return {
        "module_class": type(net).__name__,
        "in_dim": int(in_dim),
        "action_dim": int(action_dim),
        "trunk_hidden": int(x734.PPO_TRUNK_HIDDEN),
        "trunk_params": trunk,
        "policy_head_params": phead,
        "value_head_params_unused": vhead,
        "action_path_params": trunk + phead,
        "total_params": trunk + phead + vhead,
    }


def _train_adapter(x_tr: torch.Tensor, y_tr: torch.Tensor, action_dim: int,
                   passes: int, seed: int, arm_id: str) -> Tuple[Any, Dict[str, Any]]:
    """Behaviour-clone the oracle: cross-entropy on the adapter's policy logits.

    NOT reinforcement learning, and that is the entire point of the run -- the frozen-latent
    reader has already failed twice under PPO and both failures are confounded with credit
    assignment. Here the target is the oracle's action at the very state being read, so there
    is no credit-assignment problem left for the reader to fail at. If it still cannot fit the
    mapping, the shortfall is in the representation.
    """
    net = _make_adapter(int(x_tr.shape[1]), action_dim)
    opt = torch.optim.Adam(net.parameters(), lr=ADAPTER_LR)
    n = int(x_tr.shape[0])
    losses: List[float] = []
    for p in range(int(passes)):
        perm = torch.randperm(n)
        total, nb = 0.0, 0
        for i in range(0, n, ADAPTER_BATCH):
            idx = perm[i:i + ADAPTER_BATCH]
            logits, _v = net(x_tr[idx])
            loss = F.cross_entropy(logits, y_tr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item())
            nb += 1
        losses.append(total / max(nb, 1))
        if (p + 1) % 10 == 0 or (p + 1) == int(passes):
            # Deliberately NOT the "ep N/M" shape -- that pattern is the runner's episode
            # progress channel and its denominator belongs to _train_all_on_agent.
            print("  [adapter] %s seed=%d pass %d of %d ce_loss=%.4f"
                  % (arm_id, seed, p + 1, int(passes), losses[-1]), flush=True)
    return net, {"final_ce_loss": (losses[-1] if losses else None),
                 "first_ce_loss": (losses[0] if losses else None),
                 "n_passes": int(passes), "n_train_steps": n}


def _agreement(net, x: torch.Tensor, y: torch.Tensor) -> Optional[float]:
    """Top-1 agreement between the adapter's argmax and the oracle's action."""
    if int(x.shape[0]) == 0:
        return None
    with torch.no_grad():
        logits, _v = net(x)
        pred = torch.argmax(logits, dim=-1)
    return float((pred == y).float().mean().item())


def _state_blind_agreement(y_tr: torch.Tensor, y_te: torch.Tensor,
                           action_dim: int) -> Tuple[Optional[float], float]:
    """(test agreement of the majority-class predictor, train majority share).

    The honest state-blind control: pick the single most frequent oracle action ON THE TRAINING
    SPLIT and predict it everywhere on test. Its test agreement is what a reader that ignores
    the state achieves, and it is the baseline the elevation criterion is measured against.
    """
    if int(y_tr.shape[0]) == 0:
        return (None, 0.0)
    counts = torch.bincount(y_tr, minlength=int(action_dim)).float()
    maj_share = float((counts.max() / counts.sum()).item())
    maj_action = int(torch.argmax(counts).item())
    if int(y_te.shape[0]) == 0:
        return (None, maj_share)
    return (float((y_te == maj_action).float().mean().item()), maj_share)


def _unstandardised_secondary(x_tr: torch.Tensor, y_tr: torch.Tensor,
                              x_te: torch.Tensor, y_te: torch.Tensor,
                              action_dim: int, passes: int, seed: int,
                              arm_id: str) -> Dict[str, Any]:
    """SECONDARY READOUT, never verdict-bearing: the same adapter on UNSTANDARDISED features.

    Kept because the standardisation is a change to the instrument, and an instrument change
    should be measurable rather than assumed benign. The gap between this and the primary
    readout IS the size of the optimisation-scaling effect on this arm: near-zero says the
    scale gap never mattered here, large says it did -- and in the latter case the primary
    (standardised) number is the one that answers the actual question, because it is the one
    with the scaling explanation removed. Reported per arm; never entered into any criterion.
    """
    net_raw, stats = _train_adapter(x_tr, y_tr, action_dim, passes, seed,
                                    arm_id + ":unstd")
    return {
        "oracle_action_agreement_unstandardised": _agreement(net_raw, x_te, y_te),
        "oracle_action_agreement_train_unstandardised": _agreement(net_raw, x_tr, y_tr),
        "final_ce_loss_unstandardised": stats.get("final_ce_loss"),
    }


# --------------------------------------------------------------------------------------
# EVAL POLICIES (rolled-out competence -- REPORTED SECONDARY, never verdict-bearing)
# --------------------------------------------------------------------------------------
class RawFieldAdapterPolicy(Policy):
    """Greedy (argmax) rollout of an adapter that reads the raw resource_field_view."""

    name = "rawfield_adapter"

    def __init__(self, net) -> None:
        self.net = net

    def act(self, env: Any, obs_dict: Dict[str, Any]) -> int:
        with torch.no_grad():
            logits, _v = self.net(_localfield_vector(obs_dict))
        if not torch.isfinite(logits).all():
            return int(np.random.randint(0, int(env.action_dim)))
        return int(torch.argmax(logits.reshape(-1), dim=-1).item())


# --------------------------------------------------------------------------------------
# DATASET -- collected ONCE per seed, shared by every arm
# --------------------------------------------------------------------------------------
def _collect_episodes(seed: int, env_kwargs: Dict[str, Any], driver: str,
                      n_episodes: int, steps: int) -> List[Dict[str, Any]]:
    """Roll out `n_episodes` and STORE the full observation sequence plus the ORACLE's action.

    The label is ALWAYS the oracle's action at that state, whichever policy is driving. With
    driver="oracle" the oracle also acts, giving the demonstrator's own visitation (the natural
    behaviour-cloning distribution). With driver="random" a random policy acts, giving the
    state distribution 978's decode probe used -- so that arm's agreement is directly
    comparable to its r2 0.71.

    Storing the observations (rather than re-driving per arm) is what makes the three arms
    paired STEP-FOR-STEP: every arm's features are extracted from these same stored steps, so
    nothing depends on two separately-driven rollouts happening to coincide.
    """
    oracle = LocalViewGreedyPolicy(seed)
    rnd = RandomPolicy(seed + 991)
    episodes: List[Dict[str, Any]] = []
    for ep in range(int(n_episodes)):
        env = x734._make_env(seed * 1000 + ep, env_kwargs)
        _flat, obs = env.reset()
        oracle.reset(env)
        rnd.reset(env)
        obs_seq: List[Dict[str, Any]] = []
        labels: List[int] = []
        prev_actions: List[int] = []
        # PREVIOUS EXECUTED ACTION, recorded here rather than recovered from body_state at read
        # time. It is the strongest TRIVIAL predictor of the oracle's label (0.568-0.582
        # held-out, measured), so it must be a first-class recorded quantity, not an inference
        # about an observation-vector layout that could drift. -1 at t=0 (no predecessor); it
        # never equals a real action, so the first step counts as a turn state.
        last_executed = -1
        for _t in range(int(steps)):
            a_oracle = int(oracle.act(env, obs))
            obs_seq.append(obs)
            labels.append(a_oracle)
            prev_actions.append(int(last_executed))
            a_step = a_oracle if driver == "oracle" else int(rnd.act(env, obs))
            _f, _h, done, _i, obs = env.step(a_step)
            last_executed = int(a_step)
            if done:
                break
        if labels:
            episodes.append({"obs": obs_seq, "labels": labels,
                             "prev_actions": prev_actions})
    return episodes


def _prev_action_vector(episodes: List[Dict[str, Any]]) -> torch.Tensor:
    """The previous-executed-action column, in the SAME row order the feature extractors use.

    Both `_rawfield_features` and `_zworld_features` walk episodes in order and steps in order,
    so this parallel walk is aligned with their rows by construction.
    """
    out: List[int] = []
    for ep in episodes:
        out.extend(int(a) for a in ep["prev_actions"])
    if not out:
        return torch.zeros(0, dtype=torch.long)
    return torch.tensor(out, dtype=torch.long)


def _split_episodes(episodes: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]],
                                                             List[Dict[str, Any]]]:
    """Train/test split BY EPISODE. Never by step: consecutive grid-world steps are strongly
    correlated, so a step-level or prefix split leaks the test set into training."""
    n = len(episodes)
    n_tr = max(1, int(n * BC_TRAIN_FRAC)) if n > 1 else n
    return episodes[:n_tr], episodes[n_tr:]


def _rawfield_features(episodes: List[Dict[str, Any]]) -> Tuple[torch.Tensor, torch.Tensor]:
    xs, ys = [], []
    for ep in episodes:
        for obs, lab in zip(ep["obs"], ep["labels"]):
            xs.append(_localfield_vector(obs).reshape(-1))
            ys.append(int(lab))
    if not xs:
        return torch.zeros(0, RESOURCE_FIELD_DIM), torch.zeros(0, dtype=torch.long)
    return torch.stack(xs), torch.tensor(ys, dtype=torch.long)


def _zworld_features(agent, episodes: List[Dict[str, Any]]
                     ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Replay each stored episode through the warmed agent, recording sense()-time z_world.

    SENSE-time, not encoder-path: sense-time z_world is
    `(world_encoder(w) + world_topdown(beta_to_split(z_beta))) * prec` and is exactly what a
    reader consumes -- it is the quantity 978's PPO reader read and the quantity H-B/H-C are
    about. `agent.reset()` per episode because sense() advances the agent's recurrent state,
    so a replay must start each episode from the same place the original rollout did.
    """
    xs, ys = [], []
    for ep in episodes:
        agent.reset()
        for obs, lab in zip(ep["obs"], ep["labels"]):
            xs.append(x737._agent_zworld(agent, obs).reshape(-1).detach().cpu())
            ys.append(int(lab))
    if not xs:
        return torch.zeros(0, 1), torch.zeros(0, dtype=torch.long)
    return torch.stack(xs), torch.tensor(ys, dtype=torch.long)


# --------------------------------------------------------------------------------------
# CELLS
# --------------------------------------------------------------------------------------
def _score_cell(net, x_tr, y_tr, x_te, y_te, xr_te, yr_te, action_dim,
                prev_te: Optional[torch.Tensor] = None) -> Dict[str, Any]:
    """Every agreement readout for one trained adapter.

    `agreement_elevation` is measured against the STRONGEST TRIVIAL predictor, which on this
    task is NOT the majority class. See AGREEMENT_BAR's comment: the oracle is a local greedy
    walker, so "repeat the previous executed action" scores ~0.57 while majority class scores
    ~0.25. Eleveating over the weaker of the two is what let a pure-autocorrelation adapter look
    like a representation-to-action mapping.
    """
    sb_test, maj_share = _state_blind_agreement(y_tr, y_te, action_dim)
    agree_te = _agreement(net, x_te, y_te)

    prev_acc: Optional[float] = None
    turn_agree: Optional[float] = None
    n_turn = 0
    if prev_te is not None and int(y_te.shape[0]) > 0:
        prev_acc = float((y_te == prev_te).float().mean().item())
        # TURN STATES: the oracle's action differs from the previous executed action, so the
        # repeat-previous shortcut is wrong by construction (it scores exactly 0.0 here).
        # Reported as a SHORTCUT-FREE secondary; not itself verdict-bearing.
        turn = (y_te != prev_te)
        n_turn = int(turn.sum().item())
        if n_turn > 0 and net is not None:
            with torch.no_grad():
                lg, _v = net(x_te)
                pred = torch.argmax(lg, dim=-1)
            turn_agree = float((pred[turn] == y_te[turn]).float().mean().item())

    trivial = sb_test
    if trivial is None:
        trivial = prev_acc
    elif prev_acc is not None:
        trivial = max(trivial, prev_acc)

    return {
        "oracle_action_agreement": agree_te,
        "oracle_action_agreement_train": _agreement(net, x_tr, y_tr),
        "oracle_action_agreement_random_states": _agreement(net, xr_te, yr_te),
        "oracle_action_agreement_turn_states": turn_agree,
        "state_blind_agreement": sb_test,
        "prev_action_agreement": prev_acc,
        "trivial_baseline": trivial,
        "trivial_baseline_source": ("prev_action"
                                    if (prev_acc is not None and sb_test is not None
                                        and prev_acc >= sb_test)
                                    else "state_blind_majority"),
        "train_majority_class_share": maj_share,
        "agreement_elevation": ((agree_te - trivial)
                                if (agree_te is not None and trivial is not None) else None),
        "agreement_elevation_over_majority_only": (
            (agree_te - sb_test) if (agree_te is not None and sb_test is not None) else None),
        "n_train_steps": int(x_tr.shape[0]),
        "n_heldout_steps": int(x_te.shape[0]),
        "n_heldout_steps_turn": n_turn,
        "n_heldout_steps_random_states": int(xr_te.shape[0]),
    }


def _run_rawfield_cell(seed: int, data: Dict[str, Any], action_dim: int, passes: int,
                       eval_eps: int, steps: int, env_kwargs: Dict[str, Any],
                       cfg_slice: Dict[str, Any], dry_run: bool) -> Dict[str, Any]:
    """POSITIVE CONTROL: the same adapter on the raw field. Warms no encoder -- cheap, and it
    gates the expensive z_world arms."""
    print("Seed %d Condition %s:%s" % (seed, RUNG_ID, ARM_RAW), flush=True)
    with arm_cell(seed, config_slice=cfg_slice, script_path=Path(__file__),
                  config_slice_declared=True,
                  include_driver_script_in_hash=False) as cell:
        x_tr, y_tr = _rawfield_features(data["train"])
        x_te, y_te = _rawfield_features(data["test"])
        xr_te, yr_te = _rawfield_features(data["random"])
        # Train-split z-scoring, fitted here and applied to every split of THIS arm -- the
        # identical procedure every z_world arm runs. See _fit_standardiser.
        st = _fit_standardiser(x_tr) if STANDARDISE_FEATURES else {"fitted": False}
        xs_tr = _apply_standardiser(x_tr, st)
        xs_te = _apply_standardiser(x_te, st)
        xsr_te = _apply_standardiser(xr_te, st)
        net, train_stats = _train_adapter(xs_tr, y_tr, action_dim, passes, seed, ARM_RAW)
        row = {
            "cell_id": "%s|seed%d" % (ARM_RAW, seed),
            "arm_id": ARM_RAW,
            "seed": int(seed),
            "feature_dim": int(x_tr.shape[1]) if int(x_tr.shape[0]) else RESOURCE_FIELD_DIM,
            "capacity_match": _capacity_report(net, int(xs_tr.shape[1]), action_dim),
            "adapter_training": train_stats,
            "feature_standardisation": _standardiser_report(st),
            "zworld_participation_ratio": None,
            "zworld_weight_delta": None,
        }
        row.update(_score_cell(net, xs_tr, y_tr, xs_te, y_te, xsr_te, yr_te, action_dim,
                               prev_te=_prev_action_vector(data["test"])))
        row.update(_unstandardised_secondary(x_tr, y_tr, x_te, y_te, action_dim,
                                             passes, seed, ARM_RAW))
        eval_row = evaluate_seed(RawFieldAdapterPolicy(_StandardisedNet(net, st)),
                                 x734._make_env(seed, env_kwargs), eval_eps, steps)
        row.update({
            "cloned_foraging_competence": float(eval_row["foraging_competence"]),
            "cloned_competence_supra_floor": bool(eval_row["competence_supra_floor"]),
            "cloned_survival_horizon": float(eval_row["survival_horizon"]),
            "cloned_death_rate": float(eval_row["death_rate"]),
            "cloned_per_episode_resources": list(eval_row["per_episode_resources"]),
        })
        cell.stamp(row)
    print("verdict: %s" % ("PASS" if (row["oracle_action_agreement"] or 0.0)
                           >= RAW_FIELD_CONTROL_FLOOR else "FAIL"), flush=True)
    return row


def _run_zworld_cell(arm_id: str, seed: int, data: Dict[str, Any], action_dim: int,
                     zworld_p0: int, p0: int, p1: int, steps: int, passes: int,
                     eval_eps: int, env_kwargs: Dict[str, Any], cfg_slice: Dict[str, Any],
                     dry_run: bool) -> Dict[str, Any]:
    """Reproduce 978's warmup for this arm, freeze, then behaviour-clone from the latent.

    `arm_id == ARM_UNTRAINED` is the NEGATIVE CONTROL: identical agent CONSTRUCTION, warmup
    SKIPPED entirely. Everything downstream -- feature extraction, standardiser, adapter,
    dataset, scoring -- is bit-identical to the verdict arm's, so the only difference is whether
    the encoder ever received a gradient. It is what makes the verdict differential rather than
    absolute; see UNTRAINED_CONTROL_MARGIN.
    """
    untrained = (arm_id == ARM_UNTRAINED)
    # 978's OFF-arm warmup ran at P0a field weight 0.0. The ON arm (weight 0.5) was dropped, so
    # this is now a constant -- kept as an explicit config_slice key rather than deleted, because
    # a cell minted at weight 0.0 must never cache-HIT a consumer that warms at 0.5.
    with arm_cell(seed, config_slice=dict(cfg_slice, arm_p0a_field_weight=0.0,
                                          arm_warmup_skipped=bool(untrained)),
                  script_path=Path(__file__), config_slice_declared=True,
                  include_driver_script_in_hash=False) as cell:
        if untrained:
            # No warmup runs on this arm, so do NOT print a ":warmup" line for it -- a reader
            # scanning the log for warmup phases would otherwise count one that never happened.
            print("Seed %d Condition %s:%s:no-warmup" % (seed, RUNG_ID, arm_id), flush=True)
        else:
            print("Seed %d Condition %s:%s:warmup" % (seed, RUNG_ID, arm_id), flush=True)
        warm_env = x734._make_env(seed, env_kwargs)
        agent = _make_agent(warm_env)
        before = latent_stack_snapshot(agent)
        if untrained:
            # No gradient ever reaches this encoder. The warmup is skipped, NOT shortened: a
            # shortened warmup would be a weak treatment arm, whereas the control has to be a
            # genuine zero so that "beat it by a margin" means "the warmup added this much".
            stats = {}
        else:
            stats = x734._train_all_on_agent(
                agent, warm_env, seed=seed, p0_episodes=p0, p1_episodes=p1,
                steps_per_episode=steps, rung_id=RUNG_ID, total_denominator=(p0 + p1),
                zworld_p0_episodes=zworld_p0,
                zworld_p0_env=(x734._make_env(seed, env_kwargs) if zworld_p0 > 0 else None),
                zworld_p0_dry_run=dry_run,
                zworld_p0_resource_field_weight=0.0,   # 978's OFF arm; the ON arm was dropped
            )
        guard = latent_stack_weight_delta(agent, before)
        p0a = (stats or {}).get("zworld_p0", {}) or {}

        print("Seed %d Condition %s:%s" % (seed, RUNG_ID, arm_id), flush=True)
        # ENCODER FROZEN from here: every z_world read is under torch.no_grad (via
        # _agent_zworld) and no optimiser ever touches the agent again.
        x_tr, y_tr = _zworld_features(agent, data["train"])
        x_te, y_te = _zworld_features(agent, data["test"])
        xr_te, yr_te = _zworld_features(agent, data["random"])
        # Train-split z-scoring -- the same call the positive control makes, which is the
        # point: the correction is a property of the PROCEDURE, not of this arm.
        st = _fit_standardiser(x_tr) if STANDARDISE_FEATURES else {"fitted": False}
        xs_tr = _apply_standardiser(x_tr, st)
        xs_te = _apply_standardiser(x_te, st)
        xsr_te = _apply_standardiser(xr_te, st)
        net, train_stats = _train_adapter(xs_tr, y_tr, action_dim, passes, seed, arm_id)

        row = {
            "cell_id": "%s|seed%d" % (arm_id, seed),
            "arm_id": arm_id,
            "seed": int(seed),
            "warmup_skipped": bool(untrained),
            "feature_dim": int(x_tr.shape[1]) if int(x_tr.shape[0]) else 0,
            "capacity_match": _capacity_report(net, int(xs_tr.shape[1]), action_dim),
            "adapter_training": train_stats,
            "feature_standardisation": _standardiser_report(st),
            "zworld_weight_delta": guard,
            # PARTICIPATION RATIO ON THE RAW LATENT, deliberately: it is a geometric property of
            # z_world itself, and z-scoring per dimension would change it into a property of the
            # preprocessing. The scale gap it would mask is reported in feature_standardisation.
            "zworld_participation_ratio": _participation_ratio(x_tr),
            "p0a": {
                "ran": bool(p0a.get("p0a_ran")),
                "resource_field_weight": p0a.get("p0a_resource_field_weight"),
                "used_resource_field_head": bool(p0a.get("p0a_used_resource_field_head")),
                "resource_field_holdout": p0a.get("p0a_resource_field_holdout"),
            },
        }
        row.update(_score_cell(net, xs_tr, y_tr, xs_te, y_te, xsr_te, yr_te, action_dim,
                               prev_te=_prev_action_vector(data["test"])))
        row.update(_unstandardised_secondary(x_tr, y_tr, x_te, y_te, action_dim,
                                             passes, seed, arm_id))
        eval_row = evaluate_seed(x737.LatentPPOEvalPolicy(_StandardisedNet(net, st), agent),
                                 x734._make_env(seed, env_kwargs), eval_eps, steps)
        row.update({
            "cloned_foraging_competence": float(eval_row["foraging_competence"]),
            "cloned_competence_supra_floor": bool(eval_row["competence_supra_floor"]),
            "cloned_survival_horizon": float(eval_row["survival_horizon"]),
            "cloned_death_rate": float(eval_row["death_rate"]),
            "cloned_per_episode_resources": list(eval_row["per_episode_resources"]),
        })
        _ZG.observe(agent)   # AFTER stepping -- reads the counters at call time
        cell.stamp(row)
    agree = row["oracle_action_agreement"] or 0.0
    elev = row["agreement_elevation"] or 0.0
    print("verdict: %s" % ("PASS" if (agree >= AGREEMENT_BAR
                                      and elev >= AGREEMENT_ELEVATION_MIN) else "FAIL"),
          flush=True)
    return row


# --------------------------------------------------------------------------------------
# THE VERDICT GRID -- a PURE function of three seed-majority booleans, so it is testable
# without a 3-hour run. `--self-test` pushes named synthetic rows through it and asserts the
# label; a grid that can only be exercised by the full run is a grid nobody checks.
# --------------------------------------------------------------------------------------
def _adjudicate(gate_green: bool, verdict_arm_green: bool, comparator_green: bool,
                off_clears: bool, beats_untrained: bool,
                untrained_clears: bool) -> Tuple[str, str, str]:
    """(outcome, label, hypothesis_verdict). See the docstring's NULL TABLE for the full text.

    RED-TEAM PASS 2, FINDING 2 is the reason this is not the earlier three-branch ladder. That
    ladder sent EVERY sub-bar result to H-C regardless of the untrained control, so a verdict
    arm that beat its own control by the margin but fell short of the absolute bar was labelled
    "the geometry blocks the oracle mapping" -- and the H-C corroborator would then have been
    queued against an effect pointing the other way. `beats_untrained` and `untrained_clears`
    are now both carried into the label, and H-C REQUIRES `not beats_untrained`.

    RED-TEAM PASS 3, FINDING 1 is why `comparator_green` is a separate gate rather than a
    reported flag. Pass 2's Finding-4 repair wired the untrained arm's own precondition gate
    into `criteria_non_degenerate` only -- the DIAGNOSTIC. `beats_untrained` itself is computed
    from paired per-seed agreements with no reference to that gate, so a COLLAPSED untrained
    projection scores low, hands the verdict arm a free margin, and could still resolve to
    PASS / H-B-consumer-learning while the co-emitted flag read False. A reader taking
    `hypothesis_verdict` at face value -- which this driver's own docstring tells them to do --
    would never see it. The whole H-B/H-C separation rests on the comparator, so a red
    comparator licenses NO hypothesis verdict in EITHER direction: it must not manufacture H-B
    through a free margin, and it must not manufacture H-C through a vacuous
    `not beats_untrained`. It routes to substrate_not_ready_requeue, exactly as a red verdict
    arm does.
    """
    if not gate_green or not verdict_arm_green or not comparator_green:
        return ("FAIL", "substrate_not_ready_requeue", "undetermined")
    if off_clears and beats_untrained:
        return ("PASS", "zworld_supports_oracle_mapping_h_b_consumer_learning",
                "H-B-consumer-learning")
    if off_clears and not beats_untrained:
        # The registered live_gate ("CAN reproduce -> H-B") IS satisfied: the adapter reproduced
        # the oracle from the frozen latent at >= AGREEMENT_BAR, far above every measured
        # shortcut. So H-C is disconfirmed and the H-C corroborator is NOT owed. What is not
        # established is ATTRIBUTION to the warmup. Outcome stays FAIL because a pre-registered
        # load-bearing conjunct is unmet -- outcome tracks the criterion, the label carries the
        # science.
        return ("FAIL", "zworld_supports_mapping_but_warmup_non_contributory",
                "H-B-leaning-warmup-non-contributory")
    if beats_untrained:
        # Short of the bar, but the warmup demonstrably ADDED actionability the untrained
        # control lacks. A geometry that improved is not a geometry that BLOCKS; labelling this
        # H-C is exactly the Finding-2 error.
        return ("FAIL", "zworld_partially_supports_oracle_mapping_below_bar",
                "undetermined-partial-support")
    if untrained_clears:
        # Rare: an untrained random projection clears the bar while 978's warmed latent does
        # not. H-C is REFUTED for the observation at this dimensionality; the finding is that
        # the warmup DEGRADED actionability. Do not queue the H-C corroborator on this cell.
        return ("FAIL", "untrained_projection_clears_warmed_arm_does_not", "undetermined")
    return ("FAIL", "zworld_geometry_blocks_oracle_mapping_h_c_geometry_mismatch",
            "H-C-geometry-mismatch")


# The contract for the grid. Row 2 is red-team pass 2's OWN worked example (Finding 2's
# off=0.74 / untrained=0.69 / trivial=0.57): at AGREEMENT_BAR 0.80 it clears neither the bar
# (0.74 < 0.80) nor the elevation (0.74 - 0.57 = 0.17 < 0.20) nor the margin
# (0.74 - 0.69 = 0.05 < 0.10), and the untrained control does not clear either -- so it lands
# in H-C, and that IS the right answer at 0.74 against a raw-field ceiling of 0.97: passing a
# quarter of the achievable range through z_world is the geometry failing to carry the mapping.
# What Finding 2 correctly demanded, and what rows 3-5 pin, is that the OTHER sub-bar cells no
# longer collapse into that same label.
_SELF_TEST_ROWS: List[Dict[str, Any]] = [
    {"name": "gate red -> requeue, no hypothesis verdict",
     "in": dict(gate_green=False, verdict_arm_green=True, comparator_green=True, off_clears=True,
                beats_untrained=True, untrained_clears=False),
     "label": "substrate_not_ready_requeue", "verdict": "undetermined", "outcome": "FAIL"},
    {"name": "red-team worked example off=0.74 untrained=0.69 trivial=0.57 -> H-C",
     "in": dict(gate_green=True, verdict_arm_green=True, comparator_green=True, off_clears=False,
                beats_untrained=False, untrained_clears=False),
     "label": "zworld_geometry_blocks_oracle_mapping_h_c_geometry_mismatch",
     "verdict": "H-C-geometry-mismatch", "outcome": "FAIL"},
    {"name": "beats untrained but below bar -> partial, NOT H-C (Finding 2)",
     "in": dict(gate_green=True, verdict_arm_green=True, comparator_green=True, off_clears=False,
                beats_untrained=True, untrained_clears=False),
     "label": "zworld_partially_supports_oracle_mapping_below_bar",
     "verdict": "undetermined-partial-support", "outcome": "FAIL"},
    {"name": "clears bar but warmup non-contributory -> H-B-leaning, NOT bare requeue",
     "in": dict(gate_green=True, verdict_arm_green=True, comparator_green=True, off_clears=True,
                beats_untrained=False, untrained_clears=True),
     "label": "zworld_supports_mapping_but_warmup_non_contributory",
     "verdict": "H-B-leaning-warmup-non-contributory", "outcome": "FAIL"},
    {"name": "untrained clears, warmed arm does not -> warmup degraded, NOT H-C",
     "in": dict(gate_green=True, verdict_arm_green=True, comparator_green=True, off_clears=False,
                beats_untrained=False, untrained_clears=True),
     "label": "untrained_projection_clears_warmed_arm_does_not",
     "verdict": "undetermined", "outcome": "FAIL"},
    {"name": "comparator red + would-have-beaten -> requeue, NOT H-B (pass-3 Finding 1)",
     "in": dict(gate_green=True, verdict_arm_green=True, comparator_green=False,
                off_clears=True, beats_untrained=True, untrained_clears=False),
     "label": "substrate_not_ready_requeue", "verdict": "undetermined", "outcome": "FAIL"},
    {"name": "comparator red + sub-bar -> requeue, NOT H-C (pass-3 Finding 1)",
     "in": dict(gate_green=True, verdict_arm_green=True, comparator_green=False,
                off_clears=False, beats_untrained=False, untrained_clears=False),
     "label": "substrate_not_ready_requeue", "verdict": "undetermined", "outcome": "FAIL"},
    {"name": "full H-B",
     "in": dict(gate_green=True, verdict_arm_green=True, comparator_green=True, off_clears=True,
                beats_untrained=True, untrained_clears=False),
     "label": "zworld_supports_oracle_mapping_h_b_consumer_learning",
     "verdict": "H-B-consumer-learning", "outcome": "PASS"},
]


def _run_self_test() -> int:
    """Push the synthetic rows through _adjudicate. Returns a process exit code."""
    n_fail = 0
    for row in _SELF_TEST_ROWS:
        got = _adjudicate(**row["in"])
        want = (row["outcome"], row["label"], row["verdict"])
        ok = (got == want)
        n_fail += 0 if ok else 1
        print("[self-test] %-4s %s" % ("OK" if ok else "FAIL", row["name"]), flush=True)
        if not ok:
            print("            got  %r" % (got,), flush=True)
            print("            want %r" % (want,), flush=True)
    # Every H-C label must require NOT beats_untrained -- the Finding-2 invariant, asserted
    # over the whole boolean cube rather than only the named rows above.
    for og in (True, False):
        for vg in (True, False):
            for cg in (True, False):
                for oc in (True, False):
                    for bu in (True, False):
                        for uc in (True, False):
                            _o, lab, _v = _adjudicate(og, vg, cg, oc, bu, uc)
                            if lab.endswith("h_c_geometry_mismatch") and bu:
                                print("[self-test] FAIL H-C label emitted with "
                                      "beats_untrained=True (gate=%s verdict_arm=%s "
                                      "comparator=%s off=%s untrained_clears=%s)"
                                      % (og, vg, cg, oc, uc), flush=True)
                                n_fail += 1
                            # Pass-3 Finding 1: a red comparator licenses NO hypothesis
                            # verdict, in either direction.
                            if (not cg) and lab != "substrate_not_ready_requeue":
                                print("[self-test] FAIL comparator red emitted %r "
                                      "(gate=%s verdict_arm=%s off=%s beats=%s "
                                      "untrained_clears=%s)"
                                      % (lab, og, vg, oc, bu, uc), flush=True)
                                n_fail += 1
    print("[self-test] %d failure(s)" % n_fail, flush=True)
    return 1 if n_fail else 0


# --------------------------------------------------------------------------------------
def run_experiment(seeds: List[int], dry_run: bool = False) -> Dict[str, Any]:
    zworld_p0 = DRY_RUN_ZWORLD_P0 if dry_run else ZWORLD_P0_EPISODES
    p0 = DRY_RUN_P0 if dry_run else P0_WARMUP_EPISODES
    p1 = DRY_RUN_P1 if dry_run else P1_REINFORCE_EPISODES
    eval_eps = DRY_RUN_EVAL if dry_run else EVAL_EPISODES
    steps = DRY_RUN_STEPS if dry_run else STEPS_PER_EPISODE
    bc_eps = DRY_RUN_BC_EPISODES if dry_run else BC_EPISODES
    bc_rand = DRY_RUN_BC_RANDOM_EPISODES if dry_run else BC_RANDOM_EPISODES
    passes = DRY_RUN_ADAPTER_PASSES if dry_run else ADAPTER_PASSES

    # Design-time refusal BEFORE compute: a precondition no arm could satisfy is a design bug,
    # and the remedy is scoping the gate, never lowering a pre-registered threshold.
    assert_no_structurally_unsatisfiable_gate(PRECONDITION_SPECS, _arm_contexts())

    env_kwargs = x734._env_kwargs_for_rung(RUNG)
    cfg_slice = _off_path_config_slice(dry_run, zworld_p0, p0, p1, steps,
                                       bc_eps, bc_rand, passes, eval_eps)
    probe_env = x734._make_env(seeds[0], env_kwargs)
    action_dim = int(probe_env.action_dim)

    # ---- shared dataset + demonstrator anchor, once per seed --------------------------
    per_seed_data: Dict[int, Dict[str, Any]] = {}
    anchor_rows: List[Dict[str, Any]] = []
    for s in seeds:
        torch.manual_seed(s)
        np.random.seed(s)
        oracle_eps = _collect_episodes(s, env_kwargs, "oracle", bc_eps, steps)
        rand_eps = _collect_episodes(s, env_kwargs, "random", bc_rand, steps)
        tr, te = _split_episodes(oracle_eps)
        per_seed_data[s] = {"train": tr, "test": te, "random": rand_eps}
        row = evaluate_seed(LocalViewGreedyPolicy(s),
                            x734._make_env(s, env_kwargs), eval_eps, steps)
        anchor_rows.append({
            "cell_id": "local_view_greedy|seed%d" % s,
            "anchor_id": "local_view_greedy",
            "seed": int(s),
            "foraging_competence": float(row["foraging_competence"]),
            "competence_supra_floor": bool(row["competence_supra_floor"]),
            "n_train_episodes": len(tr), "n_test_episodes": len(te),
            "n_random_episodes": len(rand_eps),
        })

    # ---- ARM 1: the positive control, which gates the rest ----------------------------
    raw_rows = [_run_rawfield_cell(s, per_seed_data[s], action_dim, passes, eval_eps, steps,
                                   env_kwargs, cfg_slice, dry_run) for s in seeds]

    # ---- run-level DV headroom, measured from THIS run's own control values -----------
    # Run-level, not per-arm: it is a property of the DV and the dataset, which all three arms
    # share. ANDing it with the per-arm aggregate below is therefore NOT the forbidden
    # whole-run AND of per-arm gates (V3-EXQ-785) -- that failure is one ARM's precondition
    # vacating another ARM's result, and this check belongs to no arm.
    # The TRIVIAL baseline, not the majority-class one: on this task "repeat the previous
    # executed action" is ~0.57 against majority class's ~0.25, so headroom computed off the
    # majority class alone overstates the room the DV really has by more than 30 points.
    state_blind_vals = [float(r["trivial_baseline"]) for r in raw_rows
                        if r.get("trivial_baseline") is not None]
    raw_agree_vals = [float(r["oracle_action_agreement"]) for r in raw_rows
                      if r.get("oracle_action_agreement") is not None]
    dv_checks = []
    if state_blind_vals:
        dv_checks.append(dv_headroom_check(
            "dv_headroom_agreement_elevation",
            dv_name="oracle_action_agreement_elevation_over_state_blind",
            criterion_threshold=float(AGREEMENT_ELEVATION_MIN),
            control_values=state_blind_vals, statistic="ceiling_headroom",
            dv_bounds=(0.0, 1.0), margin=2.0,
            control=("strongest TRIVIAL predictor's held-out agreement, per seed -- "
                     "max(state-blind majority class, repeat-previous-executed-action); "
                     "the achievable elevation is what is left between it and 1.0"),
            description=("The elevation criterion must be reachable: 1.0 minus the worst "
                         "(largest) trivial baseline is the room the DV actually has above a "
                         "reader that ignores the state. margin 2.0 because the criterion must "
                         "RESOLVE the effect, not merely touch the bound. STATED HONESTLY "
                         "(red-team pass 2, Finding 3): at authoring-time measurements this "
                         "gate PASSES BY 0.02-0.03 -- measured headroom 0.418/0.420/0.434 "
                         "against the 0.40 required, a ratio of 1.05-1.09. A seed whose "
                         "held-out split realises repeat-previous-action >= 0.60 routes the "
                         "WHOLE run to substrate_not_ready_requeue on a property of the "
                         "oracle's walk rather than of any representation. That is an honest "
                         "refusal, not a misattribution -- but the margin is satisfied only "
                         "nominally and the run is one unlucky split from answering nothing. "
                         "The margin is deliberately NOT lowered to buy comfort: moving a "
                         "pre-registered threshold to obtain a pass is the failure this "
                         "precondition class exists to prevent.")))
    if raw_agree_vals:
        dv_checks.append(dv_headroom_check(
            "dv_headroom_agreement_absolute",
            dv_name="oracle_action_agreement",
            criterion_threshold=float(AGREEMENT_BAR),
            control_values=raw_agree_vals, statistic="max_abs", margin=1.0,
            control=("rawfield_ceiling held-out agreement per seed -- what this DV "
                     "demonstrably reaches at this capacity on this dataset"),
            description=("The absolute bar must sit inside what the DV is shown to reach. NOT "
                         "implied by adapter_capacity_sufficient_on_raw_field: that certifies "
                         "the instrument against a 0.60 FLOOR, while this asks whether the "
                         "0.80 BAR is reachable -- a different question, and the bar now sits "
                         "above the floor, so the implication runs the wrong way. Measured at "
                         "authoring time the raw-field control reaches 0.973-0.985 with this "
                         "same adapter, so the ratio is ~1.22 and the bar is comfortably in "
                         "range -- unlike the elevation check above, this one is NOT marginal.")))
    try:
        dv_preconditions = p0_readiness_gate(dv_checks) if dv_checks else []
        dv_gate_green = True
        dv_gate_reason = ""
    except P0NotReady as e:
        dv_preconditions = list(e.preconditions)
        dv_gate_green = False
        dv_gate_reason = "dv_headroom unmet: " + ", ".join(
            str(p.get("name")) for p in dv_preconditions if not p.get("met"))

    raw_worst, raw_worst_cell = _worst_cell(raw_rows, "oracle_action_agreement", "min")
    maj_worst, maj_worst_cell = _worst_cell(raw_rows, "train_majority_class_share", "max")
    nte_worst, nte_worst_cell = _worst_cell(raw_rows, "n_heldout_steps", "min")
    lvg_worst, lvg_worst_cell = _worst_cell(anchor_rows, "foraging_competence", "min")

    instrument_ready = bool(raw_worst >= RAW_FIELD_CONTROL_FLOOR) and dv_gate_green

    # ---- ARMS 2 and 3: the expensive warmups, only once the instrument is certified ----
    # `or dry_run`: a --dry-run smoke MUST execute every arm. At dry-run scale the positive
    # control cannot reach its floor (3 adapter passes over 6 short episodes), so without this
    # the gate short-circuits and the smoke silently never touches the warmup, the z_world
    # feature replay, the latent adapter or the rolled-out eval -- the exact blind-smoke
    # failure that let V3-EXQ-591g burn 5h46m behind a short-circuit. The real run's semantics
    # are unchanged: `instrument_ready` alone still decides the outcome LABEL below, so a
    # dry-run's arms are exercised but never certify anything.
    zw_rows: List[Dict[str, Any]] = []
    if instrument_ready or dry_run:
        for s in seeds:
            for aid in (ARM_UNTRAINED, ARM_OFF):
                zw_rows.append(_run_zworld_cell(
                    aid, s, per_seed_data[s], action_dim, zworld_p0, p0, p1, steps,
                    passes, eval_eps, env_kwargs, cfg_slice, dry_run))
    else:
        # Do NOT burn the warmups on an uninterpretable measurement. The two z_world arms are
        # emitted as UNRUN so a reader cannot mistake their absence for a measured null.
        for s in seeds:
            for aid in (ARM_UNTRAINED, ARM_OFF):
                print("Seed %d Condition %s:%s" % (s, RUNG_ID, aid), flush=True)
                print("  [skip] instrument not certified; z_world arm not run", flush=True)
                print("verdict: FAIL", flush=True)

    all_rows = raw_rows + zw_rows

    def _arm_summary(aid: str) -> Dict[str, Any]:
        rows = [r for r in all_rows if r["arm_id"] == aid]
        agr = [r["oracle_action_agreement"] for r in rows
               if r.get("oracle_action_agreement") is not None]
        elev = [r["agreement_elevation"] for r in rows
                if r.get("agreement_elevation") is not None]
        n_clear = int(sum(1 for r in rows
                          if (r.get("oracle_action_agreement") or 0.0) >= AGREEMENT_BAR
                          and (r.get("agreement_elevation") or 0.0) >= AGREEMENT_ELEVATION_MIN))
        return {
            "arm_id": aid,
            "ran": bool(rows),
            "n_seeds": len(rows),
            "n_seeds_clearing_bar": n_clear,
            "majority_clears_bar": bool(rows and n_clear >= SEED_MAJORITY),
            "mean_oracle_action_agreement": (_mean(agr) if agr else None),
            "per_seed_oracle_action_agreement": [r.get("oracle_action_agreement") for r in rows],
            "per_seed_oracle_action_agreement_train": [
                r.get("oracle_action_agreement_train") for r in rows],
            "per_seed_oracle_action_agreement_random_states": [
                r.get("oracle_action_agreement_random_states") for r in rows],
            "mean_agreement_elevation": (_mean(elev) if elev else None),
            "per_seed_agreement_elevation": [r.get("agreement_elevation") for r in rows],
            "per_seed_state_blind_agreement": [r.get("state_blind_agreement") for r in rows],
            "mean_cloned_foraging_competence": _mean(
                [r["cloned_foraging_competence"] for r in rows
                 if r.get("cloned_foraging_competence") is not None]),
            "per_seed_cloned_foraging_competence": [
                r.get("cloned_foraging_competence") for r in rows],
            "per_seed_cloned_death_rate": [r.get("cloned_death_rate") for r in rows],
            "per_seed_participation_ratio": [r.get("zworld_participation_ratio") for r in rows],
            "per_seed_final_ce_loss": [
                (r.get("adapter_training") or {}).get("final_ce_loss") for r in rows],
            "action_path_params": (rows[0]["capacity_match"]["action_path_params"]
                                   if rows else None),
        }

    per_arm = {aid: _arm_summary(aid) for aid in ARM_IDS}

    # ---- per-arm precondition gates (never AND'd across arms) -------------------------
    arm_gates = []
    for aid in ARM_IDS:
        rows = [r for r in all_rows if r["arm_id"] == aid]
        ctx = {"id": aid,
               "has_encoder": (aid != ARM_RAW),
               "trained_encoder": (aid not in (ARM_RAW, ARM_UNTRAINED))}
        measured = {
            "adapter_capacity_sufficient_on_raw_field": raw_worst,
            "oracle_labels_non_degenerate": maj_worst,
            "heldout_split_sufficient": nte_worst,
            "d3_local_view_greedy_clears_floor": lvg_worst,
        }
        if aid != ARM_RAW:
            dmin, _dc = _worst_cell(
                [{"cell_id": r["cell_id"],
                  "d": float((r.get("zworld_weight_delta") or {}).get(
                      "world_encoder_max_abs_delta", 0.0) or 0.0)} for r in rows],
                "d", "min") if rows else (0.0, None)
            prmin, _pc = _worst_cell(rows, "zworld_participation_ratio", "min") if rows \
                else (0.0, None)
            # The negative control's encoder is deliberately never trained, so the P0 delta is
            # 0 BY CONSTRUCTION -- the precondition is scoped out for it (see _arm_contexts)
            # and the measurement is not supplied, so a scoped-out spec cannot be read.
            if ctx["trained_encoder"]:
                measured["zworld_encoder_trained_in_p0"] = dmin
            # zworld_not_collapsed DOES apply to the control: a collapsed random projection
            # would make the differential comparison meaningless in the other direction.
            measured["zworld_not_collapsed"] = prmin
        arm_gates.append(evaluate_arm_gate(aid, ctx, PRECONDITION_SPECS, measured))
    gate = aggregate_arm_gates(arm_gates)

    verdict_arm_green = bool(VERDICT_ARM in (gate.get("green_arms") or []))
    gate_green = bool(gate["non_degenerate"]) and dv_gate_green

    # ---- pre-registered criteria -----------------------------------------------------
    off_clears = bool(per_arm[ARM_OFF]["majority_clears_bar"])
    untrained_clears = bool(per_arm[ARM_UNTRAINED]["majority_clears_bar"])
    raw_clears = bool(raw_worst >= RAW_FIELD_CONTROL_FLOOR)

    # ---- NEGATIVE-CONTROL DIFFERENTIAL (per seed, paired) ----------------------------
    # Measured at authoring time with THIS RUN'S OWN ADAPTER: an UNTRAINED z_world already
    # reaches 0.681-0.695 held-out (0.609-0.623 even on shortcut-free turn states), because a
    # random projection of the observation preserves the field's decodable content. An ABSOLUTE
    # bar alone therefore cannot separate "the WARMED latent supports the mapping" from "any
    # projection of this observation does". Pairing per seed (not comparing arm means) keeps the
    # comparison matched on the dataset, the standardiser procedure and the adapter, so the only
    # difference is the warmup.
    _untrained_by_seed = {r["seed"]: r.get("oracle_action_agreement")
                          for r in all_rows if r["arm_id"] == ARM_UNTRAINED}
    untrained_margins: List[Dict[str, Any]] = []
    for r in [x for x in all_rows if x["arm_id"] == ARM_OFF]:
        a_off = r.get("oracle_action_agreement")
        a_unt = _untrained_by_seed.get(r["seed"])
        # The EFFECTIVE per-seed pass threshold: what the three ANDed conjuncts actually
        # require. Recorded per seed so no reader has to reconstruct it (red-team Finding 1:
        # the constants block alone reads as "a 0.80 bar over a 0.57 shortcut", which
        # understates it).
        _triv = r.get("trivial_baseline")
        _eff_parts = [float(AGREEMENT_BAR)]
        if _triv is not None:
            _eff_parts.append(float(_triv) + float(AGREEMENT_ELEVATION_MIN))
        if a_unt is not None:
            _eff_parts.append(float(a_unt) + float(UNTRAINED_CONTROL_MARGIN))
        untrained_margins.append({
            "seed": int(r["seed"]),
            "verdict_arm_agreement": a_off,
            "untrained_control_agreement": a_unt,
            "trivial_baseline": _triv,
            "effective_pass_threshold": max(_eff_parts),
            "margin": ((a_off - a_unt) if (a_off is not None and a_unt is not None) else None),
            "clears": bool(a_off is not None and a_unt is not None
                           and (a_off - a_unt) >= UNTRAINED_CONTROL_MARGIN),
        })
    n_beats_untrained = int(sum(1 for m in untrained_margins if m["clears"]))
    beats_untrained = bool(untrained_margins and n_beats_untrained >= SEED_MAJORITY)

    # RED-TEAM PASS 3, FINDING 1. The comparator arm's READINESS, as a GATE on the verdict --
    # not merely as a reported non-degeneracy flag. `beats_untrained` above is built from
    # paired agreements alone; nothing in it consults whether ARM_UNTRAINED's own precondition
    # gate was green. A collapsed untrained projection scores low and hands the verdict arm a
    # free margin. Because the entire H-B/H-C separation is differential, a red comparator
    # licenses no hypothesis verdict in either direction, so this routes to
    # substrate_not_ready_requeue alongside a red verdict arm rather than being reported after
    # the fact. See _adjudicate's docstring.
    comparator_green = bool(
        (ARM_UNTRAINED in (gate.get("green_arms") or []))
        and any(m["untrained_control_agreement"] is not None for m in untrained_margins))

    criteria = [
        {"name": "C_zworld_adapter_reproduces_oracle", "load_bearing": True,
         "passed": bool(off_clears and beats_untrained and gate_green and verdict_arm_green),
         "description": (
             "On %s (the verdict arm), a capacity-matched supervised adapter reads the FROZEN "
             "z_world and reproduces local_view_greedy's action on >= %d of %d seeds, at "
             "held-out agreement >= %.2f AND >= %.2f above that seed's own STRONGEST TRIVIAL "
             "predictor -- max(state-blind majority class, repeat-previous-executed-action) -- "
             "AND >= %.2f above the paired UNTRAINED negative control on that same seed. All "
             "three conjuncts are required: the absolute bar alone was clearable by an "
             "autocorrelation shortcut (measured 0.57) and by an untrained random projection "
             "(measured 0.59). PASS -> H-B (the representation supports the mapping; the RL "
             "consumer failed to learn it). FAIL, with the positive control green -> H-C (the "
             "geometry does not make the mapping accessible)."
             % (VERDICT_ARM, SEED_MAJORITY, len(seeds), AGREEMENT_BAR,
                AGREEMENT_ELEVATION_MIN, UNTRAINED_CONTROL_MARGIN))},
        {"name": "C_verdict_arm_beats_untrained_control", "load_bearing": True,
         "passed": beats_untrained,
         "description": (
             "%s exceeds the paired %s negative control by >= %.2f on >= %d of %d seeds. The "
             "control is the SAME agent construction with the warmup SKIPPED, scored on the "
             "same data with the same adapter, so it carries the SAME shortcuts (label "
             "autocorrelation, random-projection-preserved field content) and the difference "
             "isolates what the warmup added to the latent's actionability. Reported as its "
             "own criterion, not folded into the one above, so a manifest shows WHICH conjunct "
             "failed. %d of %d seeds cleared."
             % (VERDICT_ARM, ARM_UNTRAINED, UNTRAINED_CONTROL_MARGIN, SEED_MAJORITY,
                len(seeds), n_beats_untrained, len(untrained_margins)))},
        {"name": "C_positive_control_learns_from_raw_field", "load_bearing": False,
         "passed": raw_clears,
         "description": (
             "The SAME adapter, dataset, optimiser and passes reach >= %.2f agreement from the "
             "raw 25-dim field on every seed. This is what makes a z_world null attributable "
             "to the representation rather than to the instrument; it is a criterion AND a "
             "precondition because its failure changes the outcome LABEL, not just a number."
             % RAW_FIELD_CONTROL_FLOOR)},
        {"name": "C_untrained_control_below_bar", "load_bearing": False,
         "passed": bool(not untrained_clears),
         "description": (
             "REPORTED, never gating -- it is what makes an H-C label attributable to THIS "
             "latent's geometry rather than to '32 dimensions of anything cannot do it'. True "
             "(the expected case) means the untrained random projection does NOT itself clear "
             "the bar+elevation. FALSE while the verdict arm also fails is a distinct and "
             "interesting outcome -- a random projection outperforming 978's warmed latent -- "
             "and the grid gives it its own label rather than folding it into H-C.")},
    ]
    combination_rule = (
        "The VERDICT is C_zworld_adapter_reproduces_oracle alone, on the pre-registered arm %s. "
        "C_positive_control_learns_from_raw_field is a gate on INTERPRETABILITY, not a "
        "conjunct: its failure routes the run to substrate_not_ready_requeue rather than "
        "turning a pass into a fail. C_untrained_control_below_bar is reported only, and is "
        "what separates an attributable H-C from 'nothing at 32 dimensions works'. The FIVE "
        "outcome cells are a function of three seed-majority booleans -- off_clears, "
        "beats_untrained, untrained_clears -- under three readiness gates -- gate_green, "
        "verdict_arm_green and comparator_green -- all computed by `_adjudicate()` and "
        "contract-tested by `--self-test`. H-C REQUIRES `not beats_untrained`: a verdict arm "
        "that beat its own untrained control by the margin is never labelled 'the geometry "
        "blocks the mapping'. A RED COMPARATOR ARM licenses no hypothesis verdict in either "
        "direction and routes to substrate_not_ready_requeue: the separation is differential, "
        "so a collapsed untrained control can neither hand the verdict arm a free margin (a "
        "spurious H-B) nor supply a vacuous `not beats_untrained` (a spurious H-C)."
        % VERDICT_ARM)
    overall_pass = bool(off_clears and beats_untrained and gate_green and verdict_arm_green
                        and comparator_green)

    # ---- interpretation grid (pure function; see _adjudicate + --self-test) ----------
    outcome, label, hypothesis_verdict = _adjudicate(
        gate_green=gate_green, verdict_arm_green=verdict_arm_green,
        comparator_green=comparator_green,
        off_clears=off_clears, beats_untrained=beats_untrained,
        untrained_clears=untrained_clears)

    non_degenerate_flags = arm_criteria_non_degenerate(
        {ARM_OFF: ["C_zworld_adapter_reproduces_oracle",
                   "C_verdict_arm_beats_untrained_control"],
         ARM_RAW: ["C_positive_control_learns_from_raw_field"],
         ARM_UNTRAINED: ["C_untrained_control_below_bar"]},
        gate,
    )
    # The differential criterion is degenerate if the negative control produced no scorable
    # agreement to compare against (an unrun or empty control arm reads as "beaten" trivially).
    #
    # RED-TEAM PASS 2, FINDING 4: it is ALSO degenerate when the CONTROL arm's own precondition
    # gate is red. arm_criteria_non_degenerate files this criterion under ARM_OFF, so its flag
    # tracked the VERDICT arm's gate and said nothing about the comparator's. A COLLAPSED
    # untrained projection (participation ratio below the floor) scores low, hands the verdict
    # arm a free 0.10 margin, and would have read as non-degenerate. The comparator's gate is
    # now ANDed in explicitly.
    #
    # RED-TEAM PASS 3, FINDING 1: that repair fixed the FLAG only. `comparator_green` is now
    # ALSO a gate inside _adjudicate, so the flag and the verdict are driven by ONE quantity
    # and cannot disagree.
    non_degenerate_flags["C_verdict_arm_beats_untrained_control"] = bool(
        non_degenerate_flags.get("C_verdict_arm_beats_untrained_control", True)
        and comparator_green)

    metrics = {
        "agreement_bar": float(AGREEMENT_BAR),
        "agreement_elevation_min": float(AGREEMENT_ELEVATION_MIN),
        "raw_field_control_floor": float(RAW_FIELD_CONTROL_FLOOR),
        "competence_floor": float(COMPETENCE_RESOURCE_FLOOR),
        "verdict_arm": VERDICT_ARM,
        "rawfield_worst_seed_agreement": raw_worst,
        "rawfield_worst_cell": raw_worst_cell,
        "zworld_off_mean_agreement": per_arm[ARM_OFF]["mean_oracle_action_agreement"],
        "zworld_off_mean_elevation": per_arm[ARM_OFF]["mean_agreement_elevation"],
        "zworld_off_n_seeds_clearing_bar": per_arm[ARM_OFF]["n_seeds_clearing_bar"],
        "zworld_untrained_n_seeds_clearing_bar": per_arm[ARM_UNTRAINED]["n_seeds_clearing_bar"],
        "untrained_control_margin_min": float(UNTRAINED_CONTROL_MARGIN),
        "zworld_untrained_mean_agreement": per_arm[ARM_UNTRAINED][
            "mean_oracle_action_agreement"],
        "n_seeds_verdict_arm_beats_untrained": n_beats_untrained,
        "worst_seed_trivial_baseline": (max(state_blind_vals) if state_blind_vals
                                        else None),
        "worst_seed_state_blind_agreement": (
            max([float(r["state_blind_agreement"]) for r in raw_rows
                 if r.get("state_blind_agreement") is not None] or [0.0])
            if raw_rows else None),
        "worst_seed_oracle_majority_share": maj_worst,
        "local_view_greedy_worst_seed_competence": lvg_worst,
        "zworld_off_mean_cloned_competence": per_arm[ARM_OFF]["mean_cloned_foraging_competence"],
        "zworld_untrained_mean_cloned_competence": per_arm[ARM_UNTRAINED]["mean_cloned_foraging_competence"],
        "rawfield_mean_cloned_competence": per_arm[ARM_RAW]["mean_cloned_foraging_competence"],
        "n_seeds": len(seeds),
    }

    interpretation = {
        "label": label,
        "question_id": HYPOTHESIS_QID,
        "hypothesis_verdict": hypothesis_verdict,
        "question": (
            "Does the FROZEN z_world V3-EXQ-978 left support the representation-to-action "
            "mapping, read by a capacity-matched supervised adapter with credit assignment "
            "removed? The registered live_gate is 'CANNOT reproduce -> H-C; CAN reproduce -> "
            "H-B'. The untrained negative control is an ATTRIBUTION axis layered on top: it "
            "says whether any actionability found is creditable to SD-018's warmup, and it "
            "gates the H-C label so a verdict arm that beat its own control is never called "
            "geometry-blocked."),
        "hypotheses": {
            "H-B-consumer-learning": (
                "z_world supports the representation-to-action mapping at the consumer's own "
                "capacity; the deficit is in the RL consumer's learning."),
            "H-C-geometry-mismatch": (
                "The information is present and linearly decodable (978: r2 0.71) but the "
                "geometry does not make the mapping accessible; the H-C corroborator (an "
                "information-preserving rotation/reweighting) becomes OWED on this branch and "
                "is deliberately NOT queued by this run."),
        },
        "effective_pass_threshold_per_seed": [
            {"seed": m["seed"], "effective_pass_threshold": m["effective_pass_threshold"],
             "trivial_baseline": m["trivial_baseline"],
             "untrained_control_agreement": m["untrained_control_agreement"]}
            for m in untrained_margins],
        "combination_rule": combination_rule,
        "preconditions": list(gate["adjudication_preconditions"]) + list(dv_preconditions),
        "per_arm_gate": gate,
        "dv_headroom_gate_green": dv_gate_green,
        "dv_headroom_reason": dv_gate_reason,
        "criteria_non_degenerate": non_degenerate_flags,
        "criteria": criteria,
        "untrained_control_differential": {
            "margin_required": float(UNTRAINED_CONTROL_MARGIN),
            "n_seeds_clearing": n_beats_untrained,
            "per_seed": untrained_margins,
            "note": ("verdict-arm minus paired untrained-control held-out agreement, per seed. "
                     "Authoring-time measurement WITH THIS RUN'S OWN ADAPTER (not a linear "
                     "readout -- that stale 0.590 figure is what red-team pass 2 Finding 1 "
                     "corrected): an untrained z_world reaches 0.681-0.695 held-out, "
                     "0.609-0.623 on shortcut-free turn states. AGREEMENT_BAR is calibrated "
                     "against THAT figure plus this margin."),
        },
        "null_reading": {
            "substrate_not_ready_requeue": (
                "ONE OF TWO, and the manifest's criteria list says which: (a) the positive "
                "control did not learn the mapping from the RAW field, or (b) the DV had no "
                "headroom for its own threshold. NEITHER is an H-C verdict and neither may be "
                "read as one -- both say the INSTRUMENT is uninterpretable here. Re-queue with "
                "more BC episodes or adapter passes. NOTE the 'cleared the bar but did not "
                "beat the untrained control' case is NO LONGER routed here: it has its own "
                "label below, because the registered live_gate is satisfied on that branch."),
            "zworld_supports_mapping_but_warmup_non_contributory": (
                "the adapter DID reproduce the oracle from the frozen latent at >= the bar, so "
                "the registered live_gate's H-B leg is satisfied and H-C is DISCONFIRMED -- the "
                "H-C corroborator is NOT owed. What is unestablished is ATTRIBUTION: an "
                "untrained random projection of the same observation does about as well, so "
                "the actionability is not creditable to SD-018's warmup. The next question is "
                "an attribution question about the warmup, not a geometry probe. Outcome is "
                "FAIL only because a pre-registered load-bearing conjunct is unmet."),
            "zworld_partially_supports_oracle_mapping_below_bar": (
                "the warmup DID add actionability the untrained control lacks (paired margin "
                "cleared) but the absolute level falls short of the registered floor. Neither "
                "hypothesis is adjudicated. NOT H-C: a geometry that demonstrably improved is "
                "not a geometry that blocks the mapping, and queueing the H-C corroborator "
                "here would chase an effect pointing the other way. Report the graded reading "
                "-- how far short, against the raw-field ceiling."),
            "untrained_projection_clears_warmed_arm_does_not": (
                "an UNTRAINED random projection of this observation clears the bar while "
                "978's warmed latent does not. H-C is REFUTED for the observation at this "
                "dimensionality; the finding is that the warmup DEGRADED actionability -- a "
                "statement about SD-018's recipe, not about geometry in general. Do NOT queue "
                "the H-C corroborator on this branch."),
            "zworld_geometry_blocks_oracle_mapping_h_c_geometry_mismatch": (
                "a reader with the consumer's EXACT capacity, trained by supervision on the "
                "oracle's own visited states with credit assignment removed, still cannot "
                "reproduce the oracle from frozen z_world -- while the same reader can from "
                "the raw field at 0.97+. AND the untrained control fails too, which is what "
                "makes this attributable to the geometry rather than to 'nothing at 32 "
                "dimensions works'; that conjunct is reported as "
                "C_untrained_control_below_bar and the label is withheld without it. H-C; the "
                "H-C corroborator becomes owed."),
        },
    }

    # Table rows built by ITERATING ARM_IDS rather than by positional %-substitution: the
    # 2026-09-04 negative-control amend added a fourth arm, and a hand-positioned format string
    # silently mis-labels every row when the arm list changes.
    _rows_md = []
    for aid in ARM_IDS:
        pa = per_arm[aid]
        _a = pa["mean_oracle_action_agreement"]
        _rows_md.append("| %s | %s | %s | %d/%d | %.4f |" % (
            aid, pa["action_path_params"],
            ("%.4f" % _a) if _a is not None else "not run",
            pa["n_seeds_clearing_bar"], pa["n_seeds"],
            pa["mean_cloned_foraging_competence"]))

    summary_markdown = """# %s -- z_world actor-adequacy locus (H-B vs H-C)

Outcome: **%s** (%s) -- hypothesis verdict: **%s**

| arm | action-path params | mean held-out agreement | seeds clearing bar | mean cloned res/ep |
|---|---|---|---|---|
%s

Strongest TRIVIAL predictor -- max(state-blind majority class, repeat-previous-executed-action)
-- worst seed: %s. Bar: agreement >= %.2f AND elevation >= %.2f over that trivial baseline AND
>= %.2f over the paired `%s` negative control, on >= %d of %d seeds. **Effective per-seed pass
threshold** (what those three ANDed conjuncts actually require) -- worst seed: %s; see
`interpretation.effective_pass_threshold_per_seed`. Verdict arm beat the untrained control on
%d of %d seeds. Demonstrator anchor local_view_greedy worst seed = %.2f res/ep against the 1.0
floor (cell %s).

%s

The adapter IS `x734.PPOPolicyNet`, the exact class V3-EXQ-978 used as its reader, so the
capacity match to the consumer's policy head holds by construction (governance amendment 5,
2026-09-03). The manipulation relative to 978 is the OBJECTIVE -- cross-entropy on the oracle's
action instead of PPO -- which removes the credit-assignment confound that both prior
frozen-latent readings (948, 978 OFF) were confounded with.

The `%s` arm is the NEGATIVE CONTROL: identical agent construction, warmup skipped. It exists
because the pre-amend criteria were clearable without reading the state -- "repeat the previous
executed action" scores 0.57 held-out on this oracle, and an untrained z_world scores
0.681-0.695 UNDER THIS RUN'S OWN ADAPTER (the stale 0.59 figure was a LINEAR readout; red-team
pass 2 Finding 1). Neither the positive control nor the state-blind majority baseline can see
either shortcut, which is why AGREEMENT_BAR is calibrated against the untrained-MLP figure.

The `zworld_on` arm (978's ON-arm warmup) was DROPPED: 978's confirmed autopsy measured the
SD-018 ON leg moving the latent two to three orders below the within-arm seed spread and
sign-inconsistently across seeds, so three more 350-episode warmups would re-measure a foregone
conclusion. What that costs is stated in the docstring's ARMS section.
""" % (QUEUE_ID, outcome, label, hypothesis_verdict,
       "\n".join(_rows_md),
       ("%.4f" % max(state_blind_vals)) if state_blind_vals else "n/a",
       AGREEMENT_BAR, AGREEMENT_ELEVATION_MIN, UNTRAINED_CONTROL_MARGIN, ARM_UNTRAINED,
       SEED_MAJORITY, len(seeds),
       ("%.4f" % max([m["effective_pass_threshold"] for m in untrained_margins]))
       if untrained_margins else "n/a",
       n_beats_untrained, len(untrained_margins),
       lvg_worst, lvg_worst_cell, combination_rule, ARM_UNTRAINED)

    return {
        "status": outcome,
        "outcome": outcome,
        "overall_pass": overall_pass,
        "metrics": metrics,
        "interpretation": interpretation,
        "summary_markdown": summary_markdown,
        "claim_ids": CLAIM_IDS,
        "evidence_direction": "unknown",
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "experiment_type": EXPERIMENT_TYPE,
        "sleep_driver_pattern": "none",
        "arm_results": all_rows,
        "per_arm": per_arm,
        "anchor_results": anchor_rows,
        "rung_id": RUNG_ID,
        "level_id": LEVEL_ID,
        "capacity_match": {
            "requirement": (
                "governance-20260903 red-team amendment 5: the adapter must be "
                "capacity-matched to the policy head the consumer used, or it cannot "
                "separate H-B from H-C."),
            "consumer_reference": (
                "V3-EXQ-978 instantiated x734.PPOPolicyNet(in_dim=z_dim, "
                "action_dim=action_dim) as its z_world reader; this driver instantiates the "
                "SAME CLASS at the same x734.PPO_TRUNK_HIDDEN, so the match holds by "
                "construction rather than by a transcribed arithmetic."),
            "trunk_hidden": int(x734.PPO_TRUNK_HIDDEN),
            "per_arm_action_path_params": {
                aid: per_arm[aid]["action_path_params"] for aid in ARM_IDS},
            "note": (
                "Both z_world arms match the consumer exactly. rawfield_ceiling reads a "
                "25-dim input rather than 32, so its first layer is smaller and its action "
                "path is slightly SMALLER -- the conservative direction for a positive "
                "control, which is therefore not bought with extra capacity."),
        },
        "frozen_latent_source": {
            "run_id": "v3_exq_978_sd018_directional_field_fishtank_20260903T111718Z_v3",
            "queue_id": "V3-EXQ-978",
            "reproduced_not_loaded": (
                "V3-EXQ-978 saved no checkpoint, so the latent is reproduced by re-running its "
                "warmup with every budget, env kwarg, rung, weighting, seed and P0a field "
                "weight imported from the same modules it imported."),
            "off_arm_field_decode_r2_sense_path": 0.70991,
            "off_arm_field_decode_r2_encoder_path": 0.85838,
        },
        "supersedes": None,
    }


if __name__ == "__main__":
    import argparse
    import time
    from datetime import datetime

    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--self-test", action="store_true",
                        help="Push the pre-registered synthetic rows through the verdict grid "
                             "and exit. No agent, no env, no warmup -- seconds, not hours.")
    args = parser.parse_args()

    if args.self_test:
        sys.exit(_run_self_test())

    t0 = time.perf_counter()
    seeds = args.seeds if args.seeds else (DRY_RUN_SEEDS if args.dry_run else SEEDS)
    result = run_experiment(seeds=seeds, dry_run=args.dry_run)

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    result["run_timestamp"] = ts
    result["timestamp_utc"] = ts
    result["run_id"] = "%s_%s_v3" % (EXPERIMENT_TYPE, ts)
    result["architecture_epoch"] = ARCHITECTURE_EPOCH
    result["queue_id"] = QUEUE_ID

    full_config = {
        "rung": RUNG,
        "level_id": LEVEL_ID,
        "env_kwargs": x734._env_kwargs_for_rung(RUNG),
        "zworld_p0_episodes": (DRY_RUN_ZWORLD_P0 if args.dry_run else ZWORLD_P0_EPISODES),
        "p0_warmup_episodes": (DRY_RUN_P0 if args.dry_run else P0_WARMUP_EPISODES),
        "p1_reinforce_episodes": (DRY_RUN_P1 if args.dry_run else P1_REINFORCE_EPISODES),
        "eval_episodes": (DRY_RUN_EVAL if args.dry_run else EVAL_EPISODES),
        "steps_per_episode": (DRY_RUN_STEPS if args.dry_run else STEPS_PER_EPISODE),
        "bc_episodes": (DRY_RUN_BC_EPISODES if args.dry_run else BC_EPISODES),
        "bc_random_episodes": (DRY_RUN_BC_RANDOM_EPISODES if args.dry_run
                               else BC_RANDOM_EPISODES),
        "bc_train_frac": BC_TRAIN_FRAC,
        "adapter_passes": (DRY_RUN_ADAPTER_PASSES if args.dry_run else ADAPTER_PASSES),
        "adapter_batch": ADAPTER_BATCH,
        "adapter_lr": ADAPTER_LR,
        "adapter_class": "experiments.v3_exq_734...PPOPolicyNet",
        "adapter_trunk_hidden": int(x734.PPO_TRUNK_HIDDEN),
        "p0a_field_weight_on_arm": P0A_FIELD_WEIGHT,
        "agreement_bar": AGREEMENT_BAR,
        "agreement_elevation_min": AGREEMENT_ELEVATION_MIN,
        "raw_field_control_floor": RAW_FIELD_CONTROL_FLOOR,
        "oracle_majority_ceiling": ORACLE_MAJORITY_CEILING,
        "heldout_min_steps": HELDOUT_MIN_STEPS,
        "participation_ratio_floor": PARTICIPATION_RATIO_FLOOR,
        "seed_majority": SEED_MAJORITY,
        "dry_run": bool(args.dry_run),
    }
    # write_flat_manifest stamps the always-core (recording_schema / substrate_hash / machine /
    # machine_class / elapsed_seconds / config / seeds) itself, hoisting substrate_hash from the
    # per-cell arm_fingerprints in arm_results.
    out_path = write_flat_manifest(
        result, None, dry_run=args.dry_run,
        config=full_config, seeds=seeds, script_path=Path(__file__),
        started_at=t0, z_goal_stream_stats=_ZG.stats(),
    )

    print("outcome: %s (%s)" % (result["outcome"], result["interpretation"]["label"]),
          flush=True)

    _outcome_raw = str(result["outcome"]).upper()
    emit_outcome(
        outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=out_path,
        dry_run=args.dry_run,
    )
