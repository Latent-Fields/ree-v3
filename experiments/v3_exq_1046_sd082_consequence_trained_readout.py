"""V3-EXQ-1046 -- SD-082: the first CONSEQUENCE-TRAINED rule readout.

The successor that `failure_autopsy_V3-EXQ-1027-1029-cluster_2026-09-14` (status
`confirmed`, applied by governance 2026-09-15, REE_assembly `beb47bca09`) routes off
V3-EXQ-1029, whose P1 budget `failure_autopsy_V3-EXQ-1028_2026-09-15` supplies.

EXPERIMENT_PURPOSE = "diagnostic"

SLEEP DRIVER: not applicable -- no sleep flag is set (this configuration enables no sleep
loop). Recorded as sleep_driver_pattern="none".

red-team (Step 4.5): pass 1 (opus) BLOCKING -> criteria redesigned; pass 2 (opus) CONTESTED
-> every finding fixed or dismissed in writing (RED-TEAM RECORD at the end of this
docstring, and the V3-EXQ-1046 queue entry note).

=== THE QUESTION ===

Every SD-082 run to date trained the `lateral_pfc` rule->bias readout in a regime where
its output could not reach E3's selection. V3-EXQ-1029 measured that directly: with
modulatory selection authority OFF, the init-head readout changes E3's argmin on
**0 of 1694** raw-identical co-fresh ticks and the sampled action on **0 of 13857**, at
native magnitude. The cluster autopsy's reading, verbatim: to measurement precision the
lineage's P1 REINFORCE surrogate is *advantage-weighted imitation of E3*, not
consequence-based reinforcement -- so the credit-assignment fix (1027) could not matter
and the learning-signal legs (1020/1028) measured the consistency of an imitation
gradient, not of the reinforcement SD-082's design names.

This run is the first in which the SD-082 consumer is engaged
(`lateral_pfc_rule_readout_consumer=True`) **and** the readout's output actually REACHES
selection during P1 (`use_modulatory_selection_authority=True`), so that the head's
output enters the selection whose outcome REINFORCE credits, and registry leg
**H1-trained-discriminating-readout** is testable at all.

    QUESTION: after P1 under an open consequence loop, does the TRAINED readout change
    E3's TOP PREFERENCE (the argmin of post-bias scores) MORE than the same substrate's
    FROZEN-INIT readout does, tick for tick on the same yoked observation stream?

"Top preference", not "committed preference": E3 never commits in this configuration
(V3-EXQ-1029 measured `committed_fraction` 0.0 on every runner; the scale probe below
reproduces it), so every behavioural selection is a multinomial draw and the argmin is
E3's ranking, not its executed action. Sampled-action divergence is recorded per pair
(`action_divergence_cofresh_raw_identical`) and is NOT a criterion.

=== WHY A NEW EXQ NUMBER AND WHY NO `supersedes` ===

The cluster autopsy's `re_derive_brake.release_basis` is explicit on both points:

  "REFUSED: V3-EXQ-1029a, a MAG_GAIN re-sweep of this init-head design (the magnitude
   question is closed from above by the substrate bound), and any further lettered
   iteration of 822/1020. LICENSED: the successor recorded in
   successor_experiment_recommendation (new EXQ number; a new adjudicating run for the
   alive registry leg H1 under a changed training regime)."

So `1029a` is refused BY NAME and this run takes a new number. And it sets **no
`supersedes`**: under the EXQ Versioning and Supersession Policy, supersession is for a
lettered iteration whose predecessor's *bug invalidated its scientific result*.
V3-EXQ-1029 is a `PASS` whose result is CONFIRMED and load-bearing -- it is what
established H-selection-authority-bounded and what this design is built on. Stamping
`supersedes: V3-EXQ-1029` would set `evidence_direction: "superseded"` on a confirmed
finding and delete it from the record. Nothing here invalidates 1029, so governance is
owed no supersession flag for it; the only bookkeeping either autopsy names is an AMEND
to the SD-082 substrate_queue entry, which is governance's own step and is NOT performed
by this driver.

=== THE RE-DERIVE BRAKE (skill Step 2.5b), WORKED RATHER THAN ASSERTED ===

The consumer predicate counts 3 for SD-082 (822b/822c/822d), over threshold 2. Both
producer artifacts nonetheless stamp `re_derive_brake.fired = false` with
`literal_count_meets_threshold = true` -- an EXPLICIT PRODUCER RELEASE, which is the
consumer predicate's clause (3). The 1028 artifact states the basis: not one of the
counted readings is a `substrate_ceiling` (each reaches the count only through the
direction fallback on `non_contributory`); `granularity_debt_cluster.py` reports SD-082
alignment `unclear=8` with ZERO `weakened`; SD-082 has never had a ceiling reading in its
history; and NO substrate build is owed by either autopsy (both amends are bookkeeping;
`severity` stays `corrupting` and `substrate_paths` stays `[]`, emptied by governance on
2026-09-09 on purpose).

Three of the skill's own "not braked" exemptions also apply independently: this is a new
EXQ NUMBER, not a lettered iteration; the manipulation (training regime) is CHANGED, so it
is not the same granularity against the same substrate; and it is a `diagnostic` whose
purpose is to discriminate WHY the ceiling held.

=== P1 BUDGET = 300 UPDATES -- PROVISIONAL, AND SAID SO ===

`P1_EPISODES = 300`, one REINFORCE update per P1 episode, from
`failure_autopsy_V3-EXQ-1028_2026-09-15`:

  * Route A (criterion power, PRIMARY): the 1028 load-bearing criterion scores on 5/5
    seeds only above T = 200; at T = 300 the worst half-count is 70 against a floor of 50
    (1.40x margin). At T = 150 two seeds starve (45, 48).
  * Route B (the persistence reading the cluster asked for): `n_persistent = 0` of 5 in
    W4 = [300, end] -- manifest field `criteria[1].n_persistent`, mirrored
    `readout.n_c2_persistent`. A longer budget is not the lever; there is no refusal
    above 300.

**THAT AUTOPSY IS STAGING, NOT GOVERNANCE-CONFIRMED** -- `status:
awaiting_human_confirmation` at the time this was queued. The budget is therefore cited
as PROVISIONAL. Following its own instruction ("the successor must record its OWN
windowed-persistence profile and per-half init-flip-tick counts so its budget can be
re-read in its own regime rather than inherited"), this driver records its own per-half
update counts and flip-tick counts -- measured at 1020's measurement point, i.e. against
the PRE-update `rule_state` of the tick (see `enable_capture`), so the halves are
comparable to 1028's.

**H-learning-signal-sign is ALIVE and this design does not assume it resolved.** 1028's
C3 scored 0 negative and 0 positive (r_local +0.045/+0.020/+0.017/+0.005/-0.002 inside
+/-2 SE bands of ~0.091) while 1020's own recorded statistic points the other way
(`adv_flip_minus_nonflip` negative on 4/5 seeds). So this driver RECORDS the same sign
statistic during P1 rather than designing against a resolved null -- but no criterion
gates on it, because a live leg is not this run's question.

=== THE DESIGN ===

Per the cluster autopsy's `successor_experiment_recommendation.sketch`.

**Five yoked runners per seed**, each an independently-constructed agent on a PRIVATE RNG
stream (torch + numpy + python `random`), all stepped on the REFERENCE's observation so
their substrates cannot diverge through the environment:

  TRAINED_INTACT    reference; drives the env; its head is the one REINFORCE updates
  TRAINED_ABLATED   same head weights, `rule_state` zeroed for the `compute_bias` call
  INIT_INTACT       head held at its P0-end (init) weights for the whole run
  INIT_ABLATED      the same init head, `rule_state` zeroed for the `compute_bias` call
  REF_SELF          bit-identical twin of TRAINED_INTACT -- the R1 self-yoke

There is no `deepcopy` of an agent anywhere: a stepped `REEAgent` holds non-leaf tensors
and `copy.deepcopy` raises (measured at authoring time). Instead every runner is built
FRESH from the same seed, so their substrates are identical by construction, and after
each REINFORCE update the trained head's `state_dict` is COPIED into TRAINED_ABLATED and
REF_SELF -- leaf tensors, which copy cleanly. INIT_* never receive a copy, so the ONLY
thing that differs between the trained pair and the init pair is the head's weights, on a
substrate that experienced the identical P0 and P1.

**PAIRS AND THE DV.** On every SCORE-phase tick where both members of a pair produced a
FRESH E3 selection AND their RAW (pre-modulatory) score vectors are bit-identical, the
pair's divergence is `argmin(post-bias scores) differs`.

    trained_pair   TRAINED_INTACT vs TRAINED_ABLATED
    init_pair      INIT_INTACT    vs INIT_ABLATED
    head_effect    TRAINED_INTACT vs INIT_INTACT      (recorded; not a criterion)
    self_pair      TRAINED_INTACT vs REF_SELF         (R1)

Because all four pair runners are yoked to one observation stream and one E3 cadence,
the trained pair and the init pair are evaluated on the SAME ticks. The DV is therefore a
PAIRED, tick-level comparison, and the joint 2x2 table is what the criteria read:

    b = ticks where the trained pair diverges and the init pair does not
    c = ticks where the init pair diverges and the trained pair does not
    n = all four runners fresh AND all four raw score vectors identical -- the two pairs
        are compared on the SAME raw-score landscape (a tick where the trained and init
        substrates' raw scores differ is counted under `cross_pair_raw_mismatch` and
        excluded; 1029 measured the analogous filter retaining 100%). `both` / `neither`,
        the run counts (`b_runs`, `c_runs`) and the number of distinct EPISODES carrying a
        discordant tick (`b_episodes`, `c_episodes`) are recorded so the tick-independence
        assumption behind the exact test can be re-read by an autopsy.

    DV (per seed) = (b - c) / n  ==  trained_pair.argmin_divergence - init_pair.argmin_divergence

**THE NULL IS THE PAIRED EXACT TEST, PER SEED, IN THIS RUN'S OWN REGIME.** Under the
null that the trained head is no more argmin-consequential than the init head, the
discordant ticks split evenly between b and c, so each seed carries an exact one-sided
binomial (McNemar) test on (b, c) at ALPHA_ONE_SIDED = 0.05 in each direction, plus an
absolute effect floor DELTA_ABS_FLOOR = 0.02 (1029's own `DIV_FLOOR`, carried unchanged)
so a significant-but-negligible difference does not clear. The scientific bar is then a
SEED MAJORITY (4 of 7) of individually significant seeds -- replication across seeds is
what protects the tick-level test against within-seed autocorrelation (E3 fires every
`e3_steps_per_tick` env steps and `rule_state` evolves slowly), and the per-seed p-values,
b, c, n and one-sided 95% bounds are all recorded so the assumption can be re-read.

The first draft of this driver used `max(0.02, 2 * SD(init pair across seeds))` as the
margin, reading the autopsy's "the margin must come from an init-only pilot". The pass-1
red-team showed on paper why that was wrong here: (i) the across-seed SD of the init
pair measures REGIME HETEROGENEITY (1029 recorded the raw E3 score range varying 23x
across seeds), not the noise of the per-seed comparison; (ii) the DV is asymmetrically
bounded (`trained >= 0`, so `DV >= -init`), and a symmetric margin built from that SD
made the sharpest negative the instrument can produce -- trained pair collapsing to zero
while the init pair diverges -- unreachable at the only banked prior (1029's on_pair, CV
0.99); (iii) in the low-variance limit it degenerated to exactly the absolute floor the
docstring promised never to use. The across-seed SD and the margin it would have implied
are still RECORDED (`diagnostics.legacy_across_seed_sd_margin`) for comparability with
1029, and gate nothing.

**MINIMAL EFFECT OF INTEREST = 0.05 (pre-registered, provisional) -- ONE number for all
three criteria.** A result is informative only against a stated effect size.
MIN_EFFECT_OF_INTEREST = 0.05 -- 2.5x the reach floor, and one quarter of 1029's banked
mean init-pair divergence (0.10) -- is the smallest trained-vs-init difference this run
treats as scientifically meaningful, in EITHER direction. It is the per-seed and mean
effect floor of C1 and C1neg and the two-sided equivalence bound of C4, so the three
criteria partition each seed's outcome (a seed cannot be both "advantage >= 0.05" and
"|difference| < 0.05"). A significant advantage SMALLER than it is recorded per seed
(`c1_sig_below_min_effect`) and, when universal, routes to its own weakens label rather
than being read as "no effect". It is provisional in the same sense as the P1 budget:
stated, not derived, and recorded so a later autopsy can re-adjudicate the branches.

**gated_policy IS OFF FOR THE WHOLE GROUP, AND "SOLE MODULATORY CHANNEL" IS MEASURED,
NOT ASSERTED.** The autopsy's second named reading is `authority_channel_confound`:
modulatory selection authority rescales the COMBINED modulatory vector by a single
per-tick scalar (`scale_factor = target_range / modulatory_spread`,
`e3_selector.py`), so it is channel-blind and amplifies every co-summed channel together
with the readout's. This driver disables `gated_policy` (the channel the autopsy named)
and the config it builds has dACC, OFC, tonic vigor, MECH-295, structured curiosity and
channel routing all OFF -- but a config argument is not a measurement, so readiness gate
R7 compares, on every fresh reference tick, E3's recorded composite modulatory range
(`modulatory_authority_range`) against the lateral_pfc head's own cross-candidate bias
range captured at the `compute_bias` call. When the readout is the only contributor the
two are equal; R7 requires their median relative difference below
SOLE_CHANNEL_REL_TOL = 1e-3 and records the maximum. That is the per-channel record the
sketch asks for, reduced to the one number that decides attribution here.

**WHAT AUTHORITY DOES TO MAGNITUDE, STATED PLAINLY.** Under authority the applied
modulatory vector is renormalised, per runner and per tick, to a FIXED cross-candidate
range (`gain * raw_score_range`, gain 0.5), independently of the head's own output
magnitude. So this run cannot and does not test whether training makes the readout
LARGER: magnitude is divided out before selection, and the DV compares the per-candidate
SHAPE of the trained head's bias against the init head's. That is the operationalisation
of H1 in the only regime where the readout reaches selection at all, and it is why the
cluster autopsy's closure of the magnitude sub-question (native |bias| bounded by the 0.1
tanh; 1029's MAG x50 already beyond any realisable head) is doubly moot here: no
magnitude arm is run, and none could inform under authority. `modulatory_authority_gain`
stays at 1029's 0.5 so the init pair is directly comparable to 1029's banked `on_pair`;
the ~50 ARC-062/GAP-B drivers' gain 2.0 targets a modulatory range twice the raw range,
a takeover rather than a graded bias, and is not adopted.

**SEEDS ARE A PRE-REGISTERED FACTOR.** `[611, 622, 633, 644, 655]` (the five lineage
seeds) plus `[666, 677]` (fresh). The autopsy asks for this specifically because 1020 and
1027 both land persistence-low on 622/633/644 only, so a lineage-seeds-only result cannot
separate "the effect" from "those three seeds". C3 asks, on a PASS, whether at least one
FRESH seed is among the clearing seeds.

**REINFORCE SURROGATE -- SCOPE, NOT A FIX.** P1 uses 1020's `_reinforce_step` unchanged:
`log_softmax(-bias / T)` over the head's OWN output, credited at the candidate E3
actually selected, advantage = episode return minus an EMA baseline. Authority changes
whether the head's output INFLUENCED that selection (it now does; R5 measures it); it
does not change the surrogate's form. A negative result here therefore weakens H1 AS
OPERATIONALISED BY THIS LINEAGE'S TRAINING PROCEDURE, and "the surrogate form itself"
remains a named residual alternative for the autopsy -- this driver does not claim to
have excluded it. Recorded in `config.reinforce.surrogate`.

=== GOV-REUSE-1 (Step 2.4) ===

Decisive readout: the paired trained-minus-init argmin-divergence difference between a
CONSEQUENCE-TRAINED head and its frozen-init counterpart, with
`lateral_pfc_rule_readout_consumer=True` and authority ON. The autopsy did this search
and recorded the result: the ~50 ARC-062/GAP-B drivers train the lpfc head under
authority but **NONE with `lateral_pfc_rule_readout_consumer=True`** (they use the
pre-SD-082 hard-clamp head and `e2_world_forward` summaries), and the whole SD-082
lineage (822*/1020/1027/1028/1029) trains with authority OFF. So no recorded manifest
carries the readout or its inputs on a compatible substrate. Not recoverable -> run.

What IS reused rather than re-run: 1029's recorded `on_pair` per-seed divergences serve as
the design-time feasibility prior, and 1020's `_build_env` / `_candidate_summaries` /
`_raw_ratio_and_flip` / `_reinforce_step` / `_StepAccumulator` and 1029's `_Runner` are
IMPORTED, never reimplemented, so this run's instrument is theirs.

=== PRE-REGISTERED CRITERIA (adjudicated over the READY seeds; see readiness) ===

Per seed k, from the joint table (b_k, c_k, n_k), with M = MIN_EFFECT_OF_INTEREST = 0.05:
    DV_k = (b_k - c_k) / n_k
    p_pos_k = P[Bin(b_k + c_k, 1/2) >= b_k]      p_neg_k = P[Bin(b_k + c_k, 1/2) >= c_k]
    SE_k = sqrt((b_k + c_k) - (b_k - c_k)^2 / n_k) / n_k
    UB_k = DV_k + 1.645 * SE_k,   LB_k = DV_k - 1.645 * SE_k

C1 (LOAD-BEARING) -- `p_pos_k < 0.05 AND DV_k >= M` on at least SEED_MAJORITY = 4 of the
seeds, AND mean(DV) over ready seeds >= M.

C1neg (ROUTING, not load-bearing) -- the mirror: `p_neg_k < 0.05 AND DV_k <= -M` on at
least 4 seeds, AND mean(DV) <= -M. Training made the readout LESS consequential than init.
It can only route a FAIL to `weakens`; a PASS run necessarily fails it. Because
`DV_k >= -init_k`, a deficit of size M is REACHABLE on a seed only when the init readout
itself reaches >= M inside the joint table; `c1neg_reachable` is recorded per seed and
the counts appear in every negative summary, so "C1neg failed on the evidence" and
"C1neg could not fire on this seed" are distinguishable.

C4 (ROUTING, not load-bearing) -- two-sided equivalence, a UNIVERSAL claim: `-M < LB_k AND
UB_k < M` on EVERY ready seed, AND |mean(DV)| < M, evaluated only when neither C1 nor
C1neg fired. A strongly negative or strongly positive seed is not "equivalent", so a
heterogeneous run cannot satisfy it (it falls to `unknown`). When C4 holds and a
significant sub-M advantage was found on a seed majority, the label says so
(`h1_weakened_advantage_below_minimal_effect`) instead of "no detectable advantage".

C2 (RECORDED, not load-bearing) -- the trained pair is itself divergent:
`trained_pair.argmin_divergence >= 0.02` on a seed majority. The pass-1 red-team proved on
paper that C2 is IMPLIED by C1 given R5 (every C1-clearing seed has trained >= init +
0.02 > 0.02), so it cannot flip a verdict and is not declared load-bearing; it annotates
the negative labels (did the trained pair collapse, or merely fall behind?).

C3 (REFINES a PASS only) -- at least one FRESH seed (666/677) is among the seeds clearing
C1. Applies only when C1 passes; recorded `applies: false` otherwise, never a vacuous
pass. (The first draft's "not exclusively 622/633/644" could never be false on a PASS,
since three seeds cannot form a four-seed majority.)

combination_rule: after readiness, PASS iff C1. Among FAILs: C1neg -> weakens; else C4 ->
weakens (equivalence); else `unknown` (inconclusive at this power or heterogeneous across
seeds, with sign, p-values, reachability and independence counts recorded). C3 refines a
PASS label; C2 annotates a FAIL label. SEED_MAJORITY is ABSOLUTE (4 of the 7 intended
seeds), never a fraction of the realised ready count; a run on fewer than 4 seeds (a
`--seeds` probe) is `insufficient_seeds_run` and makes no substrate statement. The
`_self_test` executes 13 synthetic grid branches plus an exhaustive per-seed
mutual-exclusion sweep before any compute.

=== READINESS GATES (per seed; a red seed is UNSCORED, it does not vacate the others) ===

Each seed carries its own gate, built with `experiments/_lib/precondition_gate.py`
(the V3-EXQ-785 machinery): a red seed is excluded from adjudication and reported under
`per_arm_gate.red` with the precondition it failed, while `interpretation.preconditions`
carries the GREEN seeds only, so the indexer's arm-blind recompute cannot re-vacate a
partial run. Adjudication requires MIN_READY_SEEDS = 4 green seeds (the same absolute
bar as the criteria); fewer is a REFUSAL WITH A RECORD, whose label names the dominant
red gate and whose summary states the exact counts -- never a substrate conclusion the
green seeds contradict.

  R1 self-yoke        TRAINED_INTACT vs REF_SELF: ZERO divergent ticks, whole run.
  R2 head flip        the INIT head's own argmax flip fraction (`rule_state` vs zeroed)
                      >= FLIP_FRACTION_FLOOR on INIT_INTACT, score phase -- the
                      instrument is rule-sensitive. The TRAINED head's flip fraction is
                      RECORDED, never gated on: a trained head that has lost rule
                      sensitivity is a RESULT this run exists to detect (C1neg), not an
                      instrument fault.
  R3 authority active on at least AUTHORITY_ACTIVE_FLOOR of fresh selects on the two
                      INIT runners. The trained runners' fractions are RECORDED, with an
                      equality flag across all four: authority keys on the modulatory
                      spread, i.e. on the trained head's own output, so gating on the
                      trained runners would gate on the manipulation (the R2 rule).
                      Threshold inherited from 1029, which measured it with gated_policy
                      ON -- a different composite; the dry run and probe show 1.0 on
                      every runner with the channel OFF, so the floor is slack here.
  R4 sample size      joint paired ticks n_k >= MIN_RAW_IDENTICAL_CO_FRESH (50). Note
                      1029 found the raw-identical filter retained 100% of co-fresh
                      ticks -- observation-yoking prevents the state echo it guards
                      against -- so R4 is in practice a co-fresh count; the filter is
                      kept as a zero-cost safeguard, not as a live selector.
  R5 readout reaches  the init pair's argmin divergence INSIDE the joint table
                      (`(c + both) / n`) >= INIT_REACH_FLOOR = 0.02, 1029's own
                      pre-registered readiness floor for this statistic -- the
                      consequence loop is OPEN for this seed, on the criteria's own
                      sample. Below-floor on enough seeds to deny adjudication
                      self-routes `substrate_not_ready_requeue`, NEVER a substrate
                      verdict; the summary states how many seeds were red.
  R6 the head MOVED   P1 performed >= MIN_UPDATES updates AND the trained head's
                      parameter vector differs from the init head's by more than
                      HEAD_MOVE_FLOOR (two records). Without it "TRAINED" is a misnomer.
  R7 sole channel     median relative difference between E3's composite modulatory
                      range and the lateral_pfc bias range on fresh reference ticks
                      <= SOLE_CHANNEL_REL_TOL. The attribution premise, measured. E3
                      adds two exploration channels (noisy selection head,
                      model-disagreement curiosity) AFTER that range is recorded; both
                      are off in this config, and a tick on which E3 reports either
                      active counts as a full miss (1.0) so R7 cannot be blind to them.

Every precondition record carries numeric `measured`, `threshold`, `direction` and an
explicit `comparator` (the indexer's floor default is inclusive; R5's floor is 0.0 with
`>`, so without the comparator an init pair at exactly 0.0 would be recomputed as met).

=== NULL TABLE -- every branch attributable ===

  C1 pass                  -> the consequence-trained readout is MORE argmin-consequential
      than its frozen-init counterpart on the same substrate and the same ticks:
      registry leg H1 SUPPORTED in this regime (SHAPE, not magnitude -- see above).
      C3 false -> label `..._lineage_seeds_only`: no fresh seed reproduced it.
      FORBIDDEN: any claim that the readout discriminates the RIGHT rule; any
      behavioural claim (E3 does not commit here; the DV is its ranking).
  C1neg pass               -> training made the readout LESS consequential than init,
      beyond chance, on a seed majority: H1 WEAKENED in the direction opposite to its
      prediction. C2 annotates whether the trained pair collapsed outright. WHY (rule
      insensitivity vs candidate-uniform output) is for the autopsy; the trained head's
      flip fraction and |bias| are recorded as its inputs.
  C4 pass (neither above)  -> the loop was open and |DV| is bounded inside +/-M on EVERY
      ready seed: H1 WEAKENED (equivalence). If a significant sub-M advantage was found
      on a seed majority the label says so (`..._advantage_below_minimal_effect`);
      otherwise `..._no_detectable_trained_advantage`. Sign and C1neg reachability
      counts are reported.
  none of C1/C1neg/C4      -> `unknown`: inconclusive at this power, or heterogeneous
      across seeds (e.g. a small advantage on some seeds and a large deficit on others);
      sign, per-seed p-values, bounds, reachability and independence counts recorded.
      Not a null, not support.
  fewer than 4 ready seeds -> refusal with a record. Fewer than 4 seeds RUN ->
      `insufficient_seeds_run` (no substrate statement). R5 the dominant red ->
      `substrate_not_ready_requeue` (the successor's lever is the authority
      configuration, not the training budget); R6 -> the head did not move; otherwise
      `instrument_refused_gate_red` naming the red gates. Green seeds' readings are
      carried in `arm_results` and `per_arm_gate` and are never described as
      contradicted; `interpretation.preconditions_scope_note` and
      `partial_readiness_note` carry the library's disclosure at the keys the indexer
      reads.

=== DV-SYMMETRY INVARIANCE (mandatory per-arm declaration) ===

DV = the paired difference of argmin-divergence fractions. Symmetry group of an argmin
over `raw + s * bias`: a CONSTANT BROADCAST added to every candidate's bias (cancels), and
any strictly monotone rescaling of the WHOLE post-bias score vector.

  trained_pair  manipulation = zeroing `rule_state` for the `compute_bias` call. That
                changes the head's INPUT, and the head is a nonlinear MLP whose output is
                per-candidate, so the resulting bias differs BY CANDIDATE, not by a shared
                offset. Not invariant. R2 measures it on the init head (a flip is by
                definition a non-broadcast difference); the trained head's flip fraction
                is recorded, and a trained head whose bias HAS become a broadcast is
                exactly what C1neg / C2 read out.
  init_pair     the identical manipulation on the identical substrate with the head's
                weights held at P0-end. Same argument, same R2 gate.
  head_effect   manipulation = the head's WEIGHTS (trained vs init). Distinct weight
                vectors produce distinct per-candidate biases; R6 asserts the weights
                actually differ, so this is not a comparison with itself.
  self_pair     NO manipulation, by construction. Its expected divergence is exactly 0 and
                R1 requires it -- a self-yoke that diverges means the private-RNG
                isolation leaked, which invalidates every other pair.

Authority is NOT a symmetry of this DV and the design does not claim it is: it rescales
the modulatory vector alone, not the whole score vector, so the argmin of
`raw + s * bias` does depend on `s`. What authority does is fix the applied modulatory
RANGE per runner (see above), which is why the comparison is one of bias SHAPE; the two
members of a pair each receive their own `s`, both recorded.

=== SAMPLE-SIZE INTEGRITY ===

E3 runs on a cadence (`heartbeat.e3_steps_per_tick`) and `generate_trajectories` returns
CACHED candidates when it does not fire, so a per-env-step read of `e3.last_*` is
pseudo-replicated by hold duration. This driver inherits 1029's `_Runner._choose`, which
CLEARS `last_raw_scores` / `last_scores` / `last_score_diagnostics` /
`_last_e3_selection_result` immediately before `select_action` and treats a still-`None`
read as "not fresh" -- so every recorded tick is a genuine fresh selection. `n_fresh`,
the per-pair raw-identical counts and the joint n are recorded, so the true denominator
is auditable. Runner summaries (R2, R3, |bias|) are SCORE-phase statistics: `choose` is
called with `scoring=True` only in the score phase.

=== KNOWN OPEN SUBSTRATE DEFECTS OVERLAPPING THIS DRIVER (skill Step 2.5c) ===

The SD-082 substrate_queue entry is `severity: corrupting` with `substrate_paths: []` --
DELIBERATELY EMPTIED by governance on 2026-09-09 and recorded by both autopsies as
UNCHANGED -- so the overlap gate cannot fire on it. That is a known, adjudicated
non-block, not an oversight. Module footprint is 1020's plus the e3_selector authority
block; no OPEN `corrupting` substrate_queue entry with a populated `substrate_paths` names
`e3_selector.py` or `lateral_pfc_analog.py`. Open `degrading` entries touching this
footprint, recorded rather than blocking: `SD-MECH303-THRESHOLD-SOURCING` and
`mech357-freeze-incompatible-pressure-mechanism` (both name
`ree_core/environment/causal_grid_world.py`). The authority flag itself is
`implemented_pending_validation`; it is the MANIPULATED VARIABLE here and is disclosed as
such, not a hidden confound.

=== ETHICS PREFLIGHT (Step 2.6) ===

involves_negative_valence false; involves_suffering_like_state false; involves_self_model
false; involves_inescapability_or_helplessness false; involves_offline_replay_over_harm
false; involves_social_mind_or_language false; involves_human_data_or_clinical_context
false. decision: allow.

=== SCALE PROBE AT AUTHORING (seed 611; p0=12, p1=25, score=20, 24 steps) ===

Run before queueing because the dry run cannot reach an update or a divergent tick.
Readings (2026-09-16, darwin-arm64), on the pre-red-team criteria:
  R1 self-yoke divergent ticks 0; R2 INIT head flip fraction 0.115 (floor 0.01); R3
  authority active 1.0 on every runner; R4 60/60 of co-fresh ticks raw-identical; R5 init
  pair argmin divergence 12/60 = 0.20 (the readout DOES reach E3's argmin with authority
  ON and gated_policy OFF); R6 25 updates, one per P1 episode (300 P1 episodes reach the
  200 floor), head_move_l2 0.242.
  Trained pair 0/60 = 0.00; DV = -0.20; head_effect (trained vs init intact) 0.85;
  trained head flip fraction 0.00 (P1: 0 flips on 53 measured ticks); |bias| trained
  0.015 vs init 0.023; authority scale factor median 131 (raw range 5.5 -> post range
  7.9); committed_fraction 0.0.
  Wall 552 s for 57 episodes x 24 steps x 5 runners -> ~19 s per 48-step episode ->
  ~950 min for 7 seeds x 420 episodes on this machine.
The same probe caught a crash the dry run cannot see (1020's `sample_flip_adv` entries
are 3-tuples; the sign statistic unpacked pairs) -- fixed.

Post-redesign probe, same configuration (452 s): joint table b=1, c=11, both=0, n=60
(paired DV -0.167, one-sided p_neg 0.0032, 95% upper bound -0.079 -> c1neg_seed True,
equiv_seed True); init pair 0.183, trained pair 0.017, head_effect 0.867; INIT head flip
0.125 (score phase), trained head flip 0.00; R7 composite-vs-readout range relative
difference 0.0 median and max on 60 fresh reference ticks -- the readout IS the sole
modulatory contributor, measured; authority active 1.0; 24 updates (R6a red at probe
scale only), head_move_l2 0.266; P1 flip ticks 0/31 measured at the pre-update rule
state; committed_fraction 0.0. Every gate but R6a green; the negative direction is now
reachable on the pattern the first probe produced.

=== RED-TEAM RECORD (Step 4.5) ===

Pass 1 (opus, 2026-09-16): BLOCKING. Findings and dispositions:
  F2-A C2 implied by C1 under R5, (C1 AND C2) == C1     -> FIXED: C2 demoted to a recorded
       annotation; unreachable NULL TABLE row removed.
  F2-B C1neg unreachable at the banked prior (CV 0.99)   -> FIXED: across-seed-SD margin
       replaced by the per-seed paired exact test; legacy margin recorded only.
  F2-C margin degenerates to the absolute floor           -> FIXED by the same change.
  F3-A `weakens` stamped on a positive near-miss          -> FIXED: sign-aware grid; a
       non-significant positive routes `unknown`; `weakens` requires C1neg or C4.
  F4-B all-seeds readiness vs majority criteria           -> FIXED: per-seed gates via
       precondition_gate.py, MIN_READY_SEEDS = 4, green-only adjudication list.
  F4-A init pair used three times                         -> RESOLVED by F2-B's fix (the
       margin no longer derives from it); gate + subtrahend are the design.
  F1-B authority divides out magnitude; symmetry text     -> FIXED (docstring): the DV is
       a shape comparison; the false "scalar rescale" argument withdrawn.
  F1-C sole channel asserted, not measured                -> FIXED: R7 measures it.
  F1-D surrogate is still 1020's imitation form           -> SCOPED (docstring + config):
       a negative weakens H1 as operationalised by this lineage's procedure.
  F1-E "committed preference" false (E3 never commits)   -> FIXED: "top preference".
  F1-F raw-identical filter vacuous under yoking          -> FIXED (docstring, R4 text).
  F1-G P1 flip measured post-update vs 1020 pre-update    -> FIXED: pre-update hook.
  F3-B refusal prose asserts a substrate conclusion       -> FIXED: counts, per seed.
  F3-C C3 vacuous; blanket criteria_non_degenerate        -> FIXED: C3 applies on PASS
       only and is now "a fresh seed reproduces"; per-criterion non-degeneracy.
  F3-D console verdict keyed on readiness                 -> FIXED: per-seed line reports
       readiness and the seed's own C1; `verdict:` is that seed's C1.
  F4-D R3 threshold provenance (1029, gated_policy ON)    -> NOTED in the R3 text.
  F1-A, F4-C                                              -> CLEAN, recorded as such.
Pass 2 (opus, 2026-09-16): CONTESTED, 14 findings, no BLOCKING. Dispositions:
  F-1 C1 and C4 not mutually exclusive                    -> FIXED: one effect bound M for
       C1/C1neg (per seed and mean) and two-sided C4; exclusive per seed by construction,
       swept exhaustively in the self-test.
  F-2 C4 seizable by strongly negative seeds              -> FIXED: C4 universal over ready
       seeds, |mean| < M, evaluated only after C1/C1neg; the F-2 shape routes `unknown`.
  F-3 C1 floor 0.02 below the minimal effect 0.05         -> FIXED: C1 floor = M; a
       significant sub-M advantage is recorded and gets its own weakens label.
  F-4 R5's ">0" floor at n~300 admits C1neg-blind seeds   -> FIXED: R5 floor = 1029's
       DIV_FLOOR 0.02 (>=), `c1neg_reachable` recorded per seed and reported.
  F-5 R5 measured on the init pair's own denominator      -> FIXED: R5 measured inside the
       joint table.
  F-6 run-length / independence statistics dropped        -> FIXED: 1029's per-pair run
       stats restored; joint b/c runs and episode counts added; min discordant count on
       clearing seeds recorded.
  F-7 C4 non-degeneracy condition                         -> DISMISSED with note: R5 on
       the joint table guarantees init reach on every ready seed, so `adjudicated` is the
       condition (recorded in the manifest's criteria_non_degenerate comment).
  F-8 scope note not written at the indexer's key          -> FIXED.
  F-9 refusal denominated on realised n_rows; gate_red    -> FIXED: `insufficient_seeds_run`
       branch; grid cases for it and for `instrument_refused_gate_red`.
  F-10 joint table did not require a shared raw landscape -> FIXED: cross-pair raw
       identity required; mismatches counted and surfaced.
  F-11 R7 blind to post-authority channels                -> FIXED: active-flag check.
  F-12 compute_bias hook fires in the REINFORCE replay    -> FIXED: captures cleared at
       the top of every step.
  F-13 R3 gated on the trained runners                    -> FIXED: INIT runners only;
       all four recorded with an equality flag.
  F-14 queue title carried the withdrawn margin           -> FIXED.
  Clean checks 1-9 (copy_head coverage, post-authority argmin, R7 like-for-like,
  comparator honoured downstream, exact test well-posed, self-yoke unaffected by hooks,
  grid routing, C4 reachability) recorded as such. No third pass: the skill permits one
  re-spawn, and it was spent on the causal-chain change after pass 1.
"""

from __future__ import annotations

import argparse
import copy
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _p in (str(_ROOT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from experiment_protocol import emit_outcome  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell, reset_all_rng  # noqa: E402
from experiments._lib.precondition_gate import (  # noqa: E402
    PreconditionSpec, evaluate_arm_gate, aggregate_arm_gates,
    assert_no_structurally_unsatisfiable_gate)
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402
# IMPORTED, NEVER REIMPLEMENTED: the env, the dataset recipe, the REINFORCE step and its
# telemetry are 1020's; the yoked runner and its RNG isolation are 1029's.
import experiments.v3_exq_1020_sd082_learning_signal_probe as x1020  # noqa: E402
import experiments.v3_exq_1029_sd082_selection_authority_readout_consequence as x1029  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_1046_sd082_consequence_trained_readout"
QUEUE_ID = "V3-EXQ-1046"
EXPERIMENT_PURPOSE = "diagnostic"
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"
SLEEP_DRIVER_PATTERN = "none"
CLAIM_IDS = ["SD-082"]
FANOUT_QID = "sd082_candidate_discriminating_readout_locus"
FANOUT_HYPOTHESES = ["H1-trained-discriminating-readout"]
FANOUT_AXIS = "readout"
FANOUT_SOURCE_AUTOPSY = "failure_autopsy_V3-EXQ-1027-1029-cluster_2026-09-14"
P1_BUDGET_SOURCE_AUTOPSY = "failure_autopsy_V3-EXQ-1028_2026-09-15"
P1_BUDGET_SOURCE_STATUS = "awaiting_human_confirmation -- PROVISIONAL, cited as such"
SOURCE_CHIP_REF = "chip-20260915-sd082-readout-consequence-successor"

# --------------------------------------------------------------------------------------
# SCHEDULE
# --------------------------------------------------------------------------------------
P0_EPISODES = x1020.P0_WARMUP_EPISODES            # 60 -- unchanged from 1020/1028/1029
P1_EPISODES = 300                                 # PROVISIONAL, see the docstring
SCORE_EPISODES = x1029.SCORE_EPISODES             # 60 -- 1029's post-red-team value
TOTAL_EPISODES = P0_EPISODES + P1_EPISODES + SCORE_EPISODES   # the [train] denominator
STEPS_PER_EPISODE = x1020.STEPS_PER_EPISODE       # 48

# Five lineage seeds + two FRESH. Seed is a pre-registered factor: 1020 and 1027 both land
# persistence-low on 622/633/644 only, so a lineage-only result cannot separate the effect
# from those three seeds.
LINEAGE_SEEDS = list(x1020.SEEDS)                 # [611, 622, 633, 644, 655]
FRESH_SEEDS = [666, 677]
SEEDS = LINEAGE_SEEDS + FRESH_SEEDS

CENTERING = x1029.CENTERING                       # True
MODULATORY_AUTHORITY_GAIN = x1029.MODULATORY_AUTHORITY_GAIN   # 0.5 -- see the docstring
USE_GATED_POLICY = False                          # the authority_channel_confound control
LR_LPFC_BIAS = x1020.LR_LPFC_BIAS                 # 5e-4, the lineage's own

# --------------------------------------------------------------------------------------
# RUNNERS AND PAIRS
# --------------------------------------------------------------------------------------
R_TRAINED_INTACT = "TRAINED_INTACT"
R_TRAINED_ABLATED = "TRAINED_ABLATED"
R_INIT_INTACT = "INIT_INTACT"
R_INIT_ABLATED = "INIT_ABLATED"
R_REF_SELF = "REF_SELF"
# name -> (rule_readout_ablated, receives_trained_head)
RUNNERS: Dict[str, Tuple[bool, bool]] = {
    R_TRAINED_INTACT: (False, True),
    R_TRAINED_ABLATED: (True, True),
    R_INIT_INTACT: (False, False),
    R_INIT_ABLATED: (True, False),
    R_REF_SELF: (False, True),
}
REFERENCE = R_TRAINED_INTACT
TRAINED_MIRRORS = [R_TRAINED_ABLATED, R_REF_SELF]   # receive the trained head's state_dict
PAIRS: Dict[str, Tuple[str, str]] = {
    "trained_pair": (R_TRAINED_INTACT, R_TRAINED_ABLATED),
    "init_pair": (R_INIT_INTACT, R_INIT_ABLATED),
    "head_effect": (R_TRAINED_INTACT, R_INIT_INTACT),
    "self_pair": (R_TRAINED_INTACT, R_REF_SELF),
}

# --------------------------------------------------------------------------------------
# PRE-REGISTERED THRESHOLDS
# --------------------------------------------------------------------------------------
# 1029's own DIV_FLOOR, carried unchanged. A FLOOR UNDER the measured margin, never a
# replacement for it -- the 822e lesson is that an absolute bar is not the null.
DELTA_ABS_FLOOR = x1029.DIV_FLOOR                 # 0.02
DELTA_SD_MULTIPLE = 2.0
SEED_MAJORITY = 4                                 # of 7
FLIP_FRACTION_FLOOR = x1029.FLIP_FRACTION_FLOOR   # 0.01
AUTHORITY_ACTIVE_FLOOR = x1029.AUTHORITY_ACTIVE_FLOOR    # 0.5
MIN_RAW_IDENTICAL_CO_FRESH = x1029.MIN_RAW_IDENTICAL_CO_FRESH   # 50
# R5: the init readout must reach selection on this seed, measured INSIDE the joint paired
# table (the criteria's own sample), at 1029's own pre-registered readiness floor for the
# same statistic (DIV_FLOOR 0.02, comparator >=). Pass-2 red-team F-4/F-5: a bare "> 0" at
# the realised n~300 admitted seeds with reach in (0, 0.02), and the init pair's own
# denominator is a superset of the joint table.
INIT_REACH_FLOOR = x1029.DIV_FLOOR                # 0.02
MIN_UPDATES = 200                                 # R6: P1 must actually have updated
HEAD_MOVE_FLOOR = 1e-6                            # R6: and the head must have MOVED
# Paired exact test (per seed) -- the null measured in this run's own regime.
ALPHA_ONE_SIDED = 0.05
Z_ONE_SIDED_95 = 1.6448536269514722
# The smallest trained-vs-init difference treated as meaningful. It is the per-seed AND mean
# effect floor for C1 / C1neg and the two-sided equivalence bound for C4 -- ONE number, so
# the three criteria partition the outcome space (pass-2 red-team F-1/F-3). 2.5x the reach
# floor, a quarter of 1029's banked mean init-pair divergence. PROVISIONAL, stated as such.
MIN_EFFECT_OF_INTEREST = 0.05
MIN_READY_SEEDS = SEED_MAJORITY                   # absolute, same bar as the criteria
# R7: composite modulatory range vs the readout's own bias range, median rel. diff.
SOLE_CHANNEL_REL_TOL = 1e-3

# Design-time feasibility reference -- V3-EXQ-1029's recorded on_pair (authority ON, init
# head, gated_policy ON) per seed. NOT a threshold: the margin is measured in-run.
X1029_ON_PAIR_BY_SEED = {611: 0.2615, 622: 0.0271, 633: 0.0144, 644: 0.0821, 655: 0.1155}

DRY_RUN_SEEDS = [611]
DRY_RUN_EPISODES = {"p0": 3, "p1": 4, "score": 3, "steps": 12}

_EXTRA_SUBSTRATE_PATHS = [Path(x1020.__file__), Path(x1029.__file__)]


# --------------------------------------------------------------------------------------
# AGENT
# --------------------------------------------------------------------------------------
def _make_agent(env) -> REEAgent:
    """1029's `_make_agent` kwargs with authority ALWAYS on and gated_policy OFF.

    Both differences are properties of the REGIME and are applied identically to every
    runner, so neither is a difference between arms.
    """
    return REEAgent(REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=4,
        alpha_world=0.9,
        use_lateral_pfc_analog=True,
        lateral_pfc_train_rule_bias_head=True,
        lateral_pfc_rule_readout_consumer=True,
        lateral_pfc_capture_head_diagnostics=True,
        candidate_summary_source="proposer_post_action",
        use_gated_policy=USE_GATED_POLICY,
        use_candidate_rule_field=True,
        crf_persist_rules_across_episode_reset=True,
        crf_mature_pool_dynamics=True,
        crf_availability_maintenance=True,
        crf_maintenance_floor=0.45,
        crf_maintenance_couple_to_theta=True,
        crf_tolerance_conflict_cap=3,
        crf_cue_centering=CENTERING,
        crf_cue_baseline_alpha=0.02,
        use_modulatory_selection_authority=True,
        modulatory_authority_gain=MODULATORY_AUTHORITY_GAIN,
    ))


class _CellRunner(x1029._Runner):
    """1029's yoked runner -- private RNG stream, ablation wrapper, `_choose` -- with the
    agent built by THIS driver's `_make_agent`.

    Subclassed rather than copied so `_swap`, `_choose` and `summary` stay literally
    1029's code: the RNG isolation and the fresh/raw-identical bookkeeping are the parts
    the whole design rests on, and a transcription of them would be a second
    implementation to keep in sync.
    """

    def __init__(self, env, ablated: bool) -> None:
        self.agent = _make_agent(env)
        self.authority = True
        self.ablated = bool(ablated)
        self.bias_gain = 1.0
        self._rng = (torch.get_rng_state(), np.random.get_state(), random.getstate())
        self.counters = {"dispatch": 0, "fallback": 0}
        lpfc = self.agent.lateral_pfc
        if ablated:
            orig = lpfc.compute_bias

            def _wrapped(summaries, _orig=orig, _lpfc=lpfc):
                saved = _lpfc.rule_state.detach().clone()
                with torch.no_grad():
                    _lpfc.rule_state.zero_()
                try:
                    return _orig(summaries)
                finally:
                    with torch.no_grad():
                        _lpfc.rule_state.copy_(saved)

            lpfc.compute_bias = _wrapped
        self.n_fresh = 0
        self.n_committed = 0
        self.n_head_flip_measured = 0
        self.n_head_flip = 0
        self.n_auth_active = 0
        self.raw_ranges: List[float] = []
        self.post_ranges: List[float] = []
        self.bias_abs: List[float] = []
        self.auth_scale: List[float] = []
        # REINFORCE capture (reference runner only -- see `enable_capture`).
        self.cap_summaries: Optional[torch.Tensor] = None
        self.cap_candidates: Optional[Any] = None
        self.cap_pre_rule_state: Optional[torch.Tensor] = None
        self.cap_bias_range: Optional[float] = None

    def enable_capture(self) -> None:
        """Capture the REINFORCE inputs WITHOUT reimplementing 1029's `_choose`.

        `_choose` is the part of the design the RNG isolation and the fresh /
        raw-identical bookkeeping rest on, so it is inherited verbatim rather than
        transcribed with extra lines spliced in -- a second copy would be a second thing
        to keep in sync with 1029. Instead two non-invasive hooks record exactly the two
        objects 1020's buffer needs, at the moment the agent itself produces them:

          * `lateral_pfc.compute_bias(summaries)` receives the candidate summaries;
          * `agent.select_action(candidates, ticks)` receives the candidate list.

        Only the reference runner is hooked, and neither hook changes a returned value.
        """
        agent = self.agent
        lpfc = agent.lateral_pfc
        _cb = lpfc.compute_bias

        def _cap_bias(summaries, _orig=_cb):
            try:
                if summaries is not None and torch.is_tensor(summaries):
                    self.cap_summaries = summaries.detach().clone()
            except Exception:
                self.cap_summaries = None
            out = _orig(summaries)
            # R7 input: the readout's own cross-candidate bias range this call.
            try:
                if torch.is_tensor(out) and out.numel() > 1:
                    self.cap_bias_range = float((out.max() - out.min()).item())
            except Exception:
                self.cap_bias_range = None
            return out

        lpfc.compute_bias = _cap_bias
        # 1020 measures its P1 flip label against the rule_state BEFORE the tick's
        # `lateral_pfc.update()` (which `select_action` runs before `compute_bias`);
        # snapshot it here so the per-half flip profile is comparable to 1028's.
        _up = lpfc.update

        def _cap_update(*a, _orig=_up, **k):
            try:
                self.cap_pre_rule_state = lpfc.rule_state.detach().clone()
            except Exception:
                self.cap_pre_rule_state = None
            return _orig(*a, **k)

        lpfc.update = _cap_update
        _sa = agent.select_action

        def _cap_select(candidates, ticks, _orig=_sa):
            self.cap_candidates = candidates
            return _orig(candidates, ticks)

        agent.select_action = _cap_select

    def take_capture(self) -> Tuple[Optional[torch.Tensor], Optional[Any],
                                    Optional[torch.Tensor]]:
        out = (self.cap_summaries, self.cap_candidates, self.cap_pre_rule_state)
        self.cap_summaries = None
        self.cap_candidates = None
        self.cap_pre_rule_state = None
        return out


def _resolve_selected_index(candidates, committed_class: int, n_rows: int) -> int:
    """1020's committed-candidate resolution, reproduced deliberately.

    A constant index here would credit the same candidate on every sample regardless of
    what the agent chose, which silently destroys the learning signal this run measures.
    """
    if not candidates:
        return 0
    for ci, c in enumerate(candidates):
        acts = getattr(c, "actions", None)
        if acts is not None and acts.shape[1] >= 1:
            if int(acts[:, 0, :].argmax(-1).reshape(-1)[0].item()) == committed_class:
                return min(ci, max(0, n_rows - 1))
    return 0


def _head_vector(lpfc) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for p in lpfc.bias_head_parameters()])


def _copy_head(src_lpfc, dst_lpfc) -> None:
    """Mirror the trained head into a sibling runner. Leaf tensors only -- a stepped
    REEAgent cannot be `copy.deepcopy`d (non-leaf tensors in the graph; measured at
    authoring time), which is why the whole design copies HEADS and never AGENTS."""
    dst_lpfc.rule_bias_head.load_state_dict(src_lpfc.rule_bias_head.state_dict())


def _config_slice() -> Dict[str, Any]:
    s = x1020._config_slice(CENTERING)
    s["schedule"] = {"p0": P0_EPISODES, "p1": P1_EPISODES, "score": SCORE_EPISODES,
                     "steps": STEPS_PER_EPISODE}
    s["runners"] = {k: {"ablated": v[0], "trained_head": v[1]} for k, v in RUNNERS.items()}
    s["modulatory_authority_gain"] = MODULATORY_AUTHORITY_GAIN
    s["use_modulatory_selection_authority"] = True
    s["use_gated_policy"] = USE_GATED_POLICY
    s["lr_lpfc_bias"] = LR_LPFC_BIAS
    s["delta_abs_floor"] = DELTA_ABS_FLOOR
    s["delta_sd_multiple"] = DELTA_SD_MULTIPLE
    s["alpha_one_sided"] = ALPHA_ONE_SIDED
    s["min_effect_of_interest"] = MIN_EFFECT_OF_INTEREST
    s["sole_channel_rel_tol"] = SOLE_CHANNEL_REL_TOL
    s["criteria_version"] = "paired-exact-v2"
    s["head_trained"] = True
    return s


# --------------------------------------------------------------------------------------
# PER-SEED CELL
# --------------------------------------------------------------------------------------
def _empty_pair() -> Dict[str, Any]:
    # `first_argmin_div_tick` / `argmin_divergence_runs` are 1029's run-length statistics,
    # restored (pass-2 red-team F-6): they are what lets an autopsy tell N independent
    # divergences from N/k clustered ones.
    return {"n_tick": 0, "d_action_tick": 0, "n_cofresh": 0, "n_raw_identical": 0,
            "d_argmin": 0, "d_action_raw_identical": 0,
            "first_argmin_div_tick": None, "argmin_divergence_runs": 0, "_in_run": False}


def _pair_out(st: Dict[str, Any]) -> Dict[str, Any]:
    ni = st["n_raw_identical"]
    st = {k: v for k, v in st.items() if k != "_in_run"}
    return {
        **st,
        "argmin_divergence": (st["d_argmin"] / ni) if ni else None,
        "raw_identical_fraction_of_cofresh": (ni / st["n_cofresh"]) if st["n_cofresh"] else None,
        "action_divergence_cofresh_raw_identical": (st["d_action_raw_identical"] / ni)
        if ni else None,
    }


def _empty_joint() -> Dict[str, int]:
    # b_runs / c_runs: runs of consecutive discordant scored ticks of each type;
    # b_episodes / c_episodes: distinct episodes containing one (independence audit, F-6).
    # cross_pair_raw_mismatch: ticks excluded because the TRAINED and INIT pairs did not
    # share the same raw-score landscape (F-10); the joint table requires they do.
    return {"n": 0, "b": 0, "c": 0, "both": 0, "neither": 0, "b_runs": 0, "c_runs": 0,
            "b_episodes": 0, "c_episodes": 0, "cross_pair_raw_mismatch": 0,
            "post_authority_channel_ticks": 0}


def _binom_tail_ge(k: int, n: int) -> float:
    """P[Bin(n, 1/2) >= k], exact (stdlib only)."""
    if n <= 0 or k <= 0:
        return 1.0
    if k > n:
        return 0.0
    return float(sum(math.comb(n, j) for j in range(k, n + 1))) / float(2 ** n)


def _paired_stats(joint: Dict[str, int]) -> Dict[str, Any]:
    """Per-seed paired (McNemar-form) statistics from the joint 2x2 tick table."""
    n, b, c = int(joint["n"]), int(joint["b"]), int(joint["c"])
    disc = b + c
    both = int(joint["both"])
    out: Dict[str, Any] = {"n": n, "b": b, "c": c, "both": both,
                           "neither": int(joint["neither"]), "n_discordant": disc,
                           "b_runs": int(joint.get("b_runs", 0)), "c_runs": int(joint.get("c_runs", 0)),
                           "b_episodes": int(joint.get("b_episodes", 0)),
                           "c_episodes": int(joint.get("c_episodes", 0)),
                           "cross_pair_raw_mismatch": int(joint.get("cross_pair_raw_mismatch", 0)),
                           "post_authority_channel_ticks": int(joint.get("post_authority_channel_ticks", 0)),
                           "init_div_in_joint": ((c + both) / n) if n else None,
                           "trained_div_in_joint": ((b + both) / n) if n else None,
                           "dv": None, "se": None, "p_pos": None, "p_neg": None,
                           "ub95": None, "lb95": None}
    if n <= 0:
        return out
    dv = (b - c) / n
    var = max(disc - (b - c) ** 2 / n, 0.0) / (n * n)
    se = math.sqrt(var)
    out.update({
        "dv": float(dv), "se": float(se),
        "p_pos": (_binom_tail_ge(b, disc) if disc else 1.0),
        "p_neg": (_binom_tail_ge(c, disc) if disc else 1.0),
        "ub95": float(dv + Z_ONE_SIDED_95 * se),
        "lb95": float(dv - Z_ONE_SIDED_95 * se),
    })
    return out


def _seed_flags(ps: Dict[str, Any]) -> Dict[str, bool]:
    """Per-seed criterion flags. C1 / C1neg / C4 are mutually exclusive per seed by
    construction: UB >= DV >= +M excludes UB < M; LB <= DV <= -M excludes LB > -M."""
    dv = ps["dv"]
    keys = ("c1_seed", "c1neg_seed", "c4_equiv_seed", "c1_sig_below_min_effect",
            "c1neg_reachable")
    if dv is None:
        return {k: False for k in keys}
    disc = int(ps["n_discordant"])
    m = MIN_EFFECT_OF_INTEREST
    sig_pos = bool(disc > 0 and ps["p_pos"] < ALPHA_ONE_SIDED)
    sig_neg = bool(disc > 0 and ps["p_neg"] < ALPHA_ONE_SIDED)
    return {
        "c1_seed": bool(sig_pos and dv >= m),
        "c1neg_seed": bool(sig_neg and dv <= -m),
        # Two-sided equivalence: |DV| bounded inside (-m, m) at one-sided 95% each side.
        "c4_equiv_seed": bool(ps["n"] >= MIN_RAW_IDENTICAL_CO_FRESH
                              and ps["ub95"] < m and ps["lb95"] > -m),
        # Recorded: a real but sub-meaningful advantage (significant, floor <= DV < m).
        "c1_sig_below_min_effect": bool(sig_pos and DELTA_ABS_FLOOR <= dv < m),
        # DV >= -init, so a deficit of size m is only REACHABLE when the init readout
        # itself reaches at least m inside the joint table (pass-2 red-team F-4).
        "c1neg_reachable": bool(ps["init_div_in_joint"] is not None
                                and ps["init_div_in_joint"] >= m),
    }


# Readiness preconditions -- one gate PER SEED (precondition_gate.py, the V3-EXQ-785
# machinery), so a red seed is unscored and never vacates a green one. Comparators are
# explicit: the indexer's floor default is inclusive, and R5's floor is 0.0 with ">".
_PRECONDITION_COMPARATORS: Dict[str, str] = {
    "self_yoke_zero_divergent_ticks": "<",
    "init_head_flip_fraction_supra_floor": ">=",
    "authority_active_fraction": ">=",
    "paired_sample_size": ">=",
    "init_readout_reaches_selection": ">=",
    "p1_updates_supra_floor": ">=",
    "head_moved_during_p1": ">",
    "sole_modulatory_channel": "<=",
}
_CMP = {"<": lambda a, b: a < b, "<=": lambda a, b: a <= b,
        ">": lambda a, b: a > b, ">=": lambda a, b: a >= b}


def _readiness_specs() -> List[PreconditionSpec]:
    return [
        PreconditionSpec("self_yoke_zero_divergent_ticks",
                         "R1: TRAINED_INTACT vs REF_SELF divergent ticks (action or argmin)",
                         "the reference's bit-identical twin -- a known-zero positive control",
                         1.0, "upper"),
        PreconditionSpec("init_head_flip_fraction_supra_floor",
                         "R2: the INIT head's own argmax flip fraction, rule_state vs zeroed, "
                         "score phase",
                         "the instrument's rule sensitivity; the trained head's flip fraction "
                         "is a result, recorded as trained_head_flip_fraction",
                         FLIP_FRACTION_FLOOR, "lower"),
        PreconditionSpec("authority_active_fraction",
                         "R3: min over the two INIT runners of the authority-active fraction "
                         "on fresh selects (the trained runners' fractions are recorded: the "
                         "trained head's own spread is what authority keys on, so gating on "
                         "them would gate on the manipulation -- pass-2 F-13)",
                         "authority engaged on fresh E3 selects, instrument side",
                         AUTHORITY_ACTIVE_FLOOR, "lower"),
        PreconditionSpec("paired_sample_size",
                         "R4: joint ticks with all four pair runners fresh and both pairs "
                         "raw-identical -- the criteria's own denominator",
                         "the paired sample the McNemar test runs on",
                         float(MIN_RAW_IDENTICAL_CO_FRESH), "lower"),
        PreconditionSpec("init_readout_reaches_selection",
                         "R5: the init pair's argmin divergence INSIDE the joint paired table "
                         "-- the consequence loop is open for this seed, on the criteria's own "
                         "sample (1029's DIV_FLOOR)",
                         "the SAME statistic the load-bearing criterion subtracts, measured "
                         "on the arm and the sample it subtracts", INIT_REACH_FLOOR, "lower"),
        PreconditionSpec("p1_updates_supra_floor", "R6a: REINFORCE updates performed in P1",
                         "one update per P1 episode once the buffer holds two samples",
                         float(MIN_UPDATES), "lower"),
        PreconditionSpec("head_moved_during_p1",
                         "R6b: L2 distance between the trained head and its P0-end weights",
                         "the trained head is not the init head", HEAD_MOVE_FLOOR, "lower"),
        PreconditionSpec("sole_modulatory_channel",
                         "R7: median relative difference between E3's composite modulatory "
                         "range and the lateral_pfc bias range on fresh reference ticks; a "
                         "tick on which a POST-authority channel (noisy selection head / "
                         "model-disagreement curiosity, added after the range is recorded) "
                         "was active counts as 1.0 (pass-2 F-11)",
                         "equal when the readout is the only modulatory contributor",
                         SOLE_CHANNEL_REL_TOL, "upper"),
    ]


def _flip_at_rule_state(lpfc, snap: torch.Tensor, rule_state: torch.Tensor):
    """1020's flip statistic evaluated against a GIVEN rule_state (the pre-update one)."""
    saved = lpfc.rule_state.detach().clone()
    with torch.no_grad():
        lpfc.rule_state.copy_(rule_state)
    try:
        return x1020._raw_ratio_and_flip(lpfc, snap)
    finally:
        with torch.no_grad():
            lpfc.rule_state.copy_(saved)


def _run_seed(seed: int, episodes: Dict[str, int], zg: ZGoalStreamAccumulator
              ) -> Dict[str, Any]:
    p0, p1, score, steps = (episodes["p0"], episodes["p1"], episodes["score"],
                            episodes["steps"])
    total = p0 + p1 + score
    print("Seed %d Condition consequence_trained_yoked" % seed, flush=True)
    with arm_cell(seed, config_slice=_config_slice(), script_path=Path(__file__),
                  extra_substrate_paths=_EXTRA_SUBSTRATE_PATHS) as cell:
        runners: Dict[str, _CellRunner] = {}
        for name, (abl, _trained) in RUNNERS.items():
            reset_all_rng(seed)
            env_build = x1020._build_env(seed)
            runners[name] = _CellRunner(env_build, abl)
        reset_all_rng(seed)
        env = x1020._build_env(seed)

        ref = runners[REFERENCE]
        ref.enable_capture()
        lpfc = ref.agent.lateral_pfc
        device = ref.agent.device
        bias_opt = torch.optim.Adam(list(lpfc.bias_head_parameters()), lr=LR_LPFC_BIAS)
        names = [n for n, _ in lpfc.rule_bias_head.named_parameters()]
        acc = x1020._StepAccumulator(names)
        tel: Dict[str, Any] = {"grad_norms": [], "sample_flip_adv": [], "n_updates": 0,
                               "n_adv_filtered": 0, "n_grad_nonfinite": 0,
                               "sample_flip_ret": [], "rule_state_norm_at_update": []}

        init_head_vec: Optional[torch.Tensor] = None
        outcome_buf: List[Tuple[Any, ...]] = []
        baseline = 0.0
        stats = {p: _empty_pair() for p in PAIRS}
        joint = _empty_joint()
        joint_eps = {"b": set(), "c": set()}
        joint_prev = {"b": False, "c": False}
        scored_tick = 0
        sole_rel: List[float] = []
        # Per-half P1 bookkeeping so THIS run's budget is re-readable in its own regime
        # rather than inherited from 1028 (that autopsy's own instruction).
        half = {"first": {"updates": 0, "flip_ticks": 0, "measured": 0},
                "second": {"updates": 0, "flip_ticks": 0, "measured": 0}}
        p1_mid = p0 + p1 // 2

        for ep in range(total):
            phase = "p0" if ep < p0 else ("p1" if ep < p0 + p1 else "score")
            if phase == "p1" and init_head_vec is None:
                # END OF P0 == the init head. Every INIT_* runner already holds these
                # weights (they are never updated); this snapshot is what R6 compares to.
                init_head_vec = _head_vector(lpfc).clone()
            scoring = (phase == "score")
            _, obs = env.reset()
            for r in runners.values():
                r.reset_episode()
            ep_reward = 0.0
            ep_buf: List[Tuple[Any, ...]] = []
            hk = "first" if ep < p1_mid else "second"

            for _s in range(steps):
                # Clear every capture at the top of the step (pass-2 F-12: the REINFORCE
                # replay at episode end also calls the hooked compute_bias).
                ref.cap_bias_range = None
                ref.cap_summaries = None
                ref.cap_candidates = None
                ref.cap_pre_rule_state = None
                # Clear the E3 latch before this tick's select (1029's `_choose` does the
                # same for every runner and gates `fresh` on it; this is the visible copy
                # for the R7 read below, which is additionally gated on that flag).
                ref.agent.e3.last_score_diagnostics = None
                # `scoring` is True in the SCORE phase only, so the runner summaries R2/R3
                # and |bias| read are score-phase statistics.
                outs = {k: r.choose(obs, scoring) for k, r in runners.items()}

                if phase == "p1":
                    # Build the REINFORCE buffer from what the REFERENCE agent itself
                    # produced this tick, captured by the hooks -- the same quantities
                    # 1020 buffers, resolved the same way. The flip label is evaluated at
                    # 1020's measurement point: against the PRE-update rule_state.
                    snap, cands, pre_rs = ref.take_capture()
                    if snap is not None and torch.isfinite(snap).all():
                        rs_flip = pre_rs if pre_rs is not None else lpfc.rule_state.detach().clone()
                        if float(rs_flip.norm()) > x1020.RULE_STATE_LIVE_FLOOR:
                            rf = _flip_at_rule_state(lpfc, snap, rs_flip)
                            was_flip = bool(rf[1] > 0.5) if rf is not None else False
                            sel = _resolve_selected_index(cands, int(outs[REFERENCE]["action"]),
                                                          int(snap.shape[0]))
                            ep_buf.append((snap, sel, was_flip, True))
                            half[hk]["measured"] += 1
                            if was_flip:
                                half[hk]["flip_ticks"] += 1

                if scoring:
                    scored_tick += 1
                    for pname, (a, b) in PAIRS.items():
                        st = stats[pname]
                        oa, ob = outs[a], outs[b]
                        st["n_tick"] += 1
                        st["d_action_tick"] += int(oa["action"] != ob["action"])
                        if oa["fresh"] and ob["fresh"]:
                            st["n_cofresh"] += 1
                            if torch.equal(oa["raw"], ob["raw"]):
                                st["n_raw_identical"] += 1
                                div = oa["argmin"] != ob["argmin"]
                                st["d_argmin"] += int(div)
                                st["d_action_raw_identical"] += int(oa["action"] != ob["action"])
                                if div and st["first_argmin_div_tick"] is None:
                                    st["first_argmin_div_tick"] = scored_tick
                                if div and not st["_in_run"]:
                                    st["argmin_divergence_runs"] += 1
                                st["_in_run"] = bool(div)
                    # The PAIRED table: both pairs on the same tick, on the SAME raw-score
                    # landscape (all four runners fresh, all four raw vectors identical).
                    o4 = [outs[k] for k in (R_TRAINED_INTACT, R_TRAINED_ABLATED,
                                            R_INIT_INTACT, R_INIT_ABLATED)]
                    if (all(o["fresh"] for o in o4)
                            and torch.equal(o4[0]["raw"], o4[1]["raw"])
                            and torch.equal(o4[2]["raw"], o4[3]["raw"])):
                        if not torch.equal(o4[0]["raw"], o4[2]["raw"]):
                            joint["cross_pair_raw_mismatch"] += 1
                        else:
                            td = o4[0]["argmin"] != o4[1]["argmin"]
                            idv = o4[2]["argmin"] != o4[3]["argmin"]
                            joint["n"] += 1
                            is_b = bool(td and not idv)
                            is_c = bool(idv and not td)
                            if is_b:
                                joint["b"] += 1
                                joint_eps["b"].add(ep)
                                if not joint_prev["b"]:
                                    joint["b_runs"] += 1
                            elif is_c:
                                joint["c"] += 1
                                joint_eps["c"].add(ep)
                                if not joint_prev["c"]:
                                    joint["c_runs"] += 1
                            elif td and idv:
                                joint["both"] += 1
                            else:
                                joint["neither"] += 1
                            joint_prev["b"], joint_prev["c"] = is_b, is_c
                    # R7: E3's composite modulatory range vs the readout's own bias range.
                    # A post-authority channel active on the tick fails the tick outright.
                    if outs[REFERENCE]["fresh"]:
                        diag = ref.agent.e3.last_score_diagnostics
                        comp = diag.get("modulatory_authority_range") if isinstance(diag, dict) else None
                        post_auth = bool(isinstance(diag, dict) and (
                            diag.get("noisy_selection_active") or diag.get("model_disagreement_active")))
                        br = ref.cap_bias_range
                        if post_auth:
                            joint["post_authority_channel_ticks"] += 1
                            sole_rel.append(1.0)
                        elif (comp is not None and br is not None and math.isfinite(float(comp))
                                and float(comp) > 0.0):
                            sole_rel.append(abs(float(comp) - float(br)) / float(comp))

                _, _h, done, _info, obs = env.step(outs[REFERENCE]["action"])
                if phase == "p1":
                    ep_reward += float(_h)
                if done:
                    break

            if phase == "p1":
                baseline = x1020.EMA_DECAY * baseline + (1.0 - x1020.EMA_DECAY) * ep_reward
                for cand_features, sel, flip_lbl, fresh_lbl in ep_buf:
                    outcome_buf.append((cand_features, sel, ep_reward, flip_lbl, None,
                                        fresh_lbl))
                if len(outcome_buf) > x1020.OUTCOME_BUF_MAX:
                    outcome_buf = outcome_buf[-x1020.OUTCOME_BUF_MAX:]
                did = x1020._reinforce_step(lpfc, bias_opt, outcome_buf, baseline, device,
                                            acc, tel)
                if did:
                    half[hk]["updates"] += 1
                # Mirror the trained head into its siblings so the pair differs ONLY in
                # the ablation, never in the weights.
                for m in TRAINED_MIRRORS:
                    _copy_head(lpfc, runners[m].agent.lateral_pfc)

            if (ep + 1) % 25 == 0 or (ep + 1) == total:
                print("  [train] %s seed=%d ep %d/%d updates=%d b=%d c=%d n=%d trained=%d/%d init=%d/%d"
                      % (phase, seed, ep + 1, total, tel["n_updates"],
                         joint["b"], joint["c"], joint["n"],
                         stats["trained_pair"]["d_argmin"],
                         stats["trained_pair"]["n_raw_identical"],
                         stats["init_pair"]["d_argmin"],
                         stats["init_pair"]["n_raw_identical"]), flush=True)

        for r in runners.values():
            zg.observe(r.agent)
        summaries = {k: r.summary() for k, r in runners.items()}
        pairs_out = {p: _pair_out(st) for p, st in stats.items()}
        joint["b_episodes"] = len(joint_eps["b"])
        joint["c_episodes"] = len(joint_eps["c"])
        paired = _paired_stats(joint)
        flags = _seed_flags(paired)

        trained_vec = _head_vector(lpfc)
        head_move = (float((trained_vec - init_head_vec).norm())
                     if init_head_vec is not None else 0.0)
        t_div = pairs_out["trained_pair"]["argmin_divergence"]
        i_div = pairs_out["init_pair"]["argmin_divergence"]
        init_flip = summaries[R_INIT_INTACT]["head_flip_fraction"]
        trained_flip = summaries[R_TRAINED_INTACT]["head_flip_fraction"]
        auth_by_runner = {k: summaries[k]["authority_active_fraction"] for k in
                          (R_TRAINED_INTACT, R_TRAINED_ABLATED, R_INIT_INTACT, R_INIT_ABLATED)}
        auths = [auth_by_runner[R_INIT_INTACT], auth_by_runner[R_INIT_ABLATED]]
        self_div = float(max(stats["self_pair"]["d_action_tick"], stats["self_pair"]["d_argmin"]))
        sole_median = float(np.median(sole_rel)) if sole_rel else None
        sole_max = float(np.max(sole_rel)) if sole_rel else None

        # ---- readiness: one gate for THIS seed ----
        # A None measurement is recorded as a FAILING sentinel, never silently skipped.
        measured: Dict[str, float] = {
            "self_yoke_zero_divergent_ticks": self_div,
            "init_head_flip_fraction_supra_floor": (float(init_flip) if init_flip is not None else 0.0),
            "authority_active_fraction": (float(min(auths)) if all(a is not None for a in auths) else 0.0),
            "paired_sample_size": float(paired["n"]),
            "init_readout_reaches_selection": (float(paired["init_div_in_joint"])
                                               if paired["init_div_in_joint"] is not None else 0.0),
            "p1_updates_supra_floor": float(tel["n_updates"]),
            "head_moved_during_p1": float(head_move),
            "sole_modulatory_channel": (sole_median if sole_median is not None else 1.0),
        }
        missing = [k for k, v in (("init_head_flip_fraction_supra_floor", init_flip),
                                  ("init_readout_reaches_selection", paired["init_div_in_joint"]),
                                  ("sole_modulatory_channel", sole_median)) if v is None]
        specs = _readiness_specs()
        overrides = {s.name: bool(_CMP[_PRECONDITION_COMPARATORS[s.name]](measured[s.name], s.threshold))
                     for s in specs}
        gate = evaluate_arm_gate("seed%d" % seed, {"seed": int(seed)}, specs, measured,
                                 met_overrides=overrides, auto_detect_vacuity=False)
        for p in gate["preconditions"]:
            p["comparator"] = _PRECONDITION_COMPARATORS[p["precondition"]]
            if p["precondition"] in missing:
                p["measured_missing"] = True

        row: Dict[str, Any] = {
            "arm_id": "yoked_group", "seed": int(seed),
            "seed_family": ("lineage" if seed in LINEAGE_SEEDS else "fresh"),
            "runners": summaries, "pairs": pairs_out,
            "paired": paired, **flags,
            "trained_pair_argmin_divergence": t_div,
            "init_pair_argmin_divergence": i_div,
            "dv_trained_minus_init": (None if (t_div is None or i_div is None)
                                      else float(t_div - i_div)),
            "n_updates": int(tel["n_updates"]),
            "head_move_l2": head_move,
            "init_head_flip_fraction": init_flip,
            "trained_head_flip_fraction": trained_flip,
            "authority_active_fraction_by_runner": auth_by_runner,
            "authority_active_fraction_equal_across_pair_runners": bool(
                all(v is not None for v in auth_by_runner.values())
                and max(auth_by_runner.values()) - min(auth_by_runner.values()) < 1e-9),
            "sole_channel_rel_median": sole_median,
            "sole_channel_rel_max": sole_max,
            "sole_channel_n_ticks": len(sole_rel),
            "p1_half_profile": half,
            "grad_norm_mean": (float(np.mean(tel["grad_norms"])) if tel["grad_norms"] else None),
            "n_adv_filtered": int(tel["n_adv_filtered"]),
            "n_grad_nonfinite": int(tel["n_grad_nonfinite"]),
            # H-learning-signal-sign is ALIVE -- recorded, never gated on.
            "adv_flip_minus_nonflip": _adv_flip_minus_nonflip(tel),
            "x1029_on_pair_reference": X1029_ON_PAIR_BY_SEED.get(seed),
            "summary_fallback_calls_max": max(s["summary_fallback_calls"]
                                              for s in summaries.values()),
            "readiness_gate": gate,
            "ready": bool(gate["gate_green"]),
            "c2_trained_pair_divergent": bool(t_div is not None and t_div >= DELTA_ABS_FLOOR),
        }
        cell.stamp(row)
    print("  [seed %d] ready=%s failed=%s b=%d c=%d n=%d dv=%s p_pos=%s p_neg=%s ub95=%s "
          "c1_seed=%s c1neg_seed=%s equiv_seed=%s c1neg_reachable=%s"
          % (seed, row["ready"], ",".join(gate["failed_preconditions"]) or "-",
             paired["b"], paired["c"], paired["n"],
             ("%.4f" % paired["dv"]) if paired["dv"] is not None else "None",
             ("%.3g" % paired["p_pos"]) if paired["p_pos"] is not None else "None",
             ("%.3g" % paired["p_neg"]) if paired["p_neg"] is not None else "None",
             ("%.4f" % paired["ub95"]) if paired["ub95"] is not None else "None",
             flags["c1_seed"], flags["c1neg_seed"], flags["c4_equiv_seed"],
             flags["c1neg_reachable"]), flush=True)
    # The per-seed verdict is that seed's own C1 (ready AND significant trained-over-init).
    print("verdict: %s" % ("PASS" if (row["ready"] and flags["c1_seed"]) else "FAIL"), flush=True)
    return row


def _adv_flip_minus_nonflip(tel: Dict[str, Any]) -> Optional[float]:
    """1020's own sign statistic, recorded so this run feeds the ALIVE
    H-learning-signal-sign leg instead of presupposing it resolved."""
    # 1020's `_reinforce_step` appends (was_flip, adv, fresh) 3-tuples, so index rather
    # than unpack pairs -- a bare 2-tuple unpack crashed at the first scale probe (the
    # dry run never reaches an update, so it cannot see this).
    pairs = tel.get("sample_flip_adv") or []
    fl = [t[1] for t in pairs if t[0]]
    nf = [t[1] for t in pairs if not t[0]]
    if not fl or not nf:
        return None
    return float(np.mean(fl) - np.mean(nf))


# --------------------------------------------------------------------------------------
# ADJUDICATION
# --------------------------------------------------------------------------------------
def _adjudicate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    gates = [r["readiness_gate"] for r in rows]
    agg = aggregate_arm_gates(gates)
    ready = [r for r in rows if r["readiness_gate"]["gate_green"]]
    n_ready = len(ready)
    adjudicated = bool(n_ready >= MIN_READY_SEEDS)
    red_by_seed = {r["seed"]: list(r["readiness_gate"]["failed_preconditions"]) for r in rows
                   if not r["readiness_gate"]["gate_green"]}

    def _mean(xs: List[float]) -> Optional[float]:
        return float(np.mean(xs)) if xs else None

    dvs = [(r["seed"], r["paired"]["dv"]) for r in ready if r["paired"]["dv"] is not None]
    mean_dv = _mean([v for _s, v in dvs])
    n_c1 = sum(1 for r in ready if r["c1_seed"])
    n_c1neg = sum(1 for r in ready if r["c1neg_seed"])
    n_c4 = sum(1 for r in ready if r["c4_equiv_seed"])
    n_c2 = sum(1 for r in ready if r["trained_pair_argmin_divergence"] is not None
               and r["trained_pair_argmin_divergence"] >= DELTA_ABS_FLOOR)
    n_init_div = sum(1 for r in ready if r["init_pair_argmin_divergence"] is not None
                     and r["init_pair_argmin_divergence"] >= DELTA_ABS_FLOOR)
    pooled = {"b": sum(r["paired"]["b"] for r in ready), "c": sum(r["paired"]["c"] for r in ready),
              "n": sum(r["paired"]["n"] for r in ready)}

    n_sig_small = sum(1 for r in ready if r["c1_sig_below_min_effect"])
    n_c1neg_unreachable = sum(1 for r in ready if not r["c1neg_reachable"])
    m = MIN_EFFECT_OF_INTEREST
    c1 = bool(adjudicated and n_c1 >= SEED_MAJORITY and mean_dv is not None and mean_dv >= m)
    c1_neg = bool(adjudicated and n_c1neg >= SEED_MAJORITY and mean_dv is not None
                  and mean_dv <= -m)
    # C4 is a UNIVERSAL claim: every ready seed two-sided equivalent AND |mean| < m, and
    # only when neither directional criterion fired (pass-2 red-team F-1/F-2).
    c4 = bool(adjudicated and not c1 and not c1_neg and n_ready > 0 and n_c4 == n_ready
              and mean_dv is not None and abs(mean_dv) < m)
    c2 = bool(adjudicated and n_c2 >= SEED_MAJORITY)
    clearing = [r["seed"] for r in ready if r["c1_seed"]]
    c3_applies = c1
    c3 = bool(c1 and any(s in FRESH_SEEDS for s in clearing))

    # Legacy (first-draft) margin, RECORDED for comparability with 1029, gating nothing.
    init_divs = [r["init_pair_argmin_divergence"] for r in ready
                 if r["init_pair_argmin_divergence"] is not None]
    sd_init = float(np.std(np.asarray(init_divs, dtype=np.float64), ddof=1)) \
        if len(init_divs) > 1 else 0.0
    legacy_margin = float(max(DELTA_ABS_FLOOR, DELTA_SD_MULTIPLE * sd_init))

    per_seed = {str(r["seed"]): {"dv": r["paired"]["dv"], "b": r["paired"]["b"],
                                 "c": r["paired"]["c"], "n": r["paired"]["n"],
                                 "p_pos": r["paired"]["p_pos"], "p_neg": r["paired"]["p_neg"],
                                 "ub95": r["paired"]["ub95"], "lb95": r["paired"]["lb95"],
                                 "ready": bool(r["readiness_gate"]["gate_green"])}
                for r in rows}
    criteria = [
        {"name": "C1_trained_over_init_paired_exact", "load_bearing": True, "passed": c1,
         "measured": float(n_c1), "threshold": float(SEED_MAJORITY),
         "mean_dv_ready_seeds": mean_dv, "effect_floor": m,
         "alpha_one_sided": ALPHA_ONE_SIDED, "n_ready_seeds": n_ready,
         "clearing_seeds": clearing, "per_seed": per_seed,
         "n_seeds_significant_below_min_effect": n_sig_small,
         "min_n_discordant_on_clearing_seeds": (min(r["paired"]["n_discordant"] for r in ready
                                                    if r["c1_seed"]) if clearing else None),
         "definition": "per seed: P[Bin(b+c,1/2) >= b] < alpha AND (b-c)/n >= effect_floor; "
                       "passes on >= threshold seeds AND mean DV over ready seeds >= floor"},
        {"name": "C1neg_init_over_trained_paired_exact", "load_bearing": False,
         "passed": c1_neg, "measured": float(n_c1neg), "threshold": float(SEED_MAJORITY),
         "mean_dv_ready_seeds": mean_dv, "effect_floor": -m,
         "n_ready_seeds_c1neg_unreachable": n_c1neg_unreachable,
         "routes": "FAIL -> weakens (training reduced the readout's argmin consequence)"},
        {"name": "C4_two_sided_equivalence_all_ready_seeds", "load_bearing": False,
         "passed": c4, "measured": float(n_c4), "threshold": float(n_ready),
         "min_effect_of_interest": m, "abs_mean_dv": (abs(mean_dv) if mean_dv is not None else None),
         "definition": "per seed: -m < LB95 and UB95 < m (m=%.3f); passes only on EVERY ready "
                       "seed with |mean DV| < m, and only when neither C1 nor C1neg fired" % m,
         "routes": "FAIL -> weakens (equivalence)"},
        {"name": "C2_trained_pair_itself_divergent", "load_bearing": False,
         "passed": c2, "measured": float(n_c2), "threshold": float(SEED_MAJORITY),
         "n_seeds_init_pair_divergent": n_init_div,
         "note": "implied by C1 given R5 (pass-1 red-team F2-A); annotates FAIL labels only"},
        {"name": "C3_fresh_seed_reproduces", "load_bearing": False, "passed": c3,
         "threshold_not_applicable": "applies only on a C1 PASS; records which clearing seeds are fresh",
         "applies": c3_applies, "measured": float(sum(1 for s in clearing if s in FRESH_SEEDS)),
         "threshold": 1.0, "clearing_seeds": clearing, "fresh_seeds": list(FRESH_SEEDS),
         "note": "refines a PASS label only; recorded applies:false on any FAIL"},
    ]
    combination_rule = (
        "Adjudicated over the READY seeds only (per-seed readiness gates; MIN_READY_SEEDS=%d, "
        "absolute). PASS iff C1 (significant AND >= MIN_EFFECT_OF_INTEREST=%.2f per seed on a "
        "seed majority, mean >= it). Among FAILs: C1neg (the mirror) -> weakens; else C4 "
        "(two-sided equivalence within +/-%.2f on EVERY ready seed, |mean| < it) -> weakens; "
        "else unknown. C1/C1neg/C4 are mutually exclusive per seed by construction. C3 "
        "refines a PASS; C2 annotates a FAIL. SEED_MAJORITY=%d is absolute, never a fraction "
        "of the realised ready count." % (MIN_READY_SEEDS, m, m, SEED_MAJORITY))

    n_rows = len(rows)
    if not adjudicated:
        outcome, direction = "FAIL", "non_contributory"
        r5_red = [s for s, f in red_by_seed.items() if "init_readout_reaches_selection" in f]
        r6_red = [s for s, f in red_by_seed.items()
                  if "p1_updates_supra_floor" in f or "head_moved_during_p1" in f]
        green = [r["seed"] for r in ready]
        counts = "; ".join("seed %s failed %s" % (s, ", ".join(f)) for s, f in red_by_seed.items())
        if n_rows < MIN_READY_SEEDS:
            label = "insufficient_seeds_run"
            summary = ("Run on %d seed(s); adjudication requires %d ready seeds of the %d "
                       "intended. No substrate statement is made (pass-2 red-team F-9). Green "
                       "seeds: %s. Red gates by seed: %s."
                       % (n_rows, MIN_READY_SEEDS, len(SEEDS), green, counts or "-"))
        elif n_rows - len(r5_red) < MIN_READY_SEEDS:
            label = "substrate_not_ready_requeue"
            summary = ("R5 red on %d of %d seeds (%s): on those seeds the init-head readout did "
                       "not change E3's argmin on any paired tick even with authority ON and "
                       "gated_policy OFF. Green seeds: %s (their readings are carried in "
                       "arm_results and per_arm_gate and are NOT contradicted). Only %d seeds "
                       "passed every gate (need %d), so no H1 verdict is licensed. The "
                       "successor's lever is the AUTHORITY CONFIGURATION, not the training "
                       "budget. Red gates by seed: %s."
                       % (len(r5_red), n_rows, r5_red, green, n_ready, MIN_READY_SEEDS, counts))
        elif n_rows - len(r6_red) < MIN_READY_SEEDS:
            label = "instrument_refused_head_did_not_move"
            summary = ("R6 red on %d of %d seeds (%s): P1 did not perform >= %d updates or the "
                       "trained head did not move from its P0-end weights, so on those seeds "
                       "the trained pair and the init pair compare a head with itself. Only %d "
                       "seeds passed every gate (need %d). Instrument refusal, NOT a null about "
                       "H1. Red gates by seed: %s."
                       % (len(r6_red), n_rows, r6_red, MIN_UPDATES, n_ready, MIN_READY_SEEDS, counts))
        else:
            label = "instrument_refused_gate_red"
            summary = ("Only %d of %d seeds passed every readiness gate (need %d); refusal with "
                       "a record, no scientific leg adjudicated. Red gates by seed: %s."
                       % (n_ready, n_rows, MIN_READY_SEEDS, counts))
    elif c1:
        outcome, direction = "PASS", "supports"
        if not c3:
            label = "h1_consequence_trained_readout_more_argmin_consequential_lineage_seeds_only"
            summary = ("C1 passes on %d ready seeds (clearing: %s) but no FRESH seed (666/677) is "
                       "among them. The consequence-trained readout is more argmin-consequential "
                       "than the frozen-init readout on the lineage seeds only; report as "
                       "LINEAGE-SPECIFIC. Mean paired DV %.4f over %d ready seeds."
                       % (n_c1, clearing, mean_dv, n_ready))
        else:
            label = "h1_consequence_trained_readout_more_argmin_consequential"
            summary = ("The consequence-trained readout changes E3's TOP PREFERENCE (argmin of "
                       "post-bias scores) MORE than the same substrate's frozen-init readout, on "
                       "the same yoked ticks, beyond chance on %d seeds (clearing: %s, including "
                       "a fresh seed); mean paired DV %.4f over %d ready seeds. Registry leg "
                       "H1-trained-discriminating-readout SUPPORTED in the first regime able to "
                       "test it -- as a SHAPE effect (authority normalises magnitude). FORBIDDEN: "
                       "any claim that the readout discriminates the RIGHT rule, and any "
                       "behavioural claim -- E3 does not commit here; the DV is its ranking."
                       % (n_c1, clearing, mean_dv, n_ready))
    elif c1_neg:
        outcome, direction = "FAIL", "weakens"
        label = "h1_weakened_training_reduced_readout_consequence"
        summary = ("C1neg: the consequence-trained readout changes E3's top preference LESS "
                   "than the same substrate's frozen-init readout, beyond chance and by at "
                   "least %.2f, on %d of %d ready seeds; mean paired DV %.4f. Trained pair "
                   "divergent above the floor on %d ready seeds, init pair on %d (C2 %s). "
                   "Consequence training under an open loop made the readout LESS "
                   "argmin-consequential, not more: H1 WEAKENED, in the direction opposite to "
                   "its prediction, as operationalised by this lineage's training procedure. "
                   "WHY is for the autopsy -- trained_head_flip_fraction and |bias| are "
                   "recorded as its inputs."
                   % (m, n_c1neg, n_ready, mean_dv, n_c2, n_init_div,
                      "annotates: trained pair collapsed" if not c2 else "held"))
    elif c4:
        outcome, direction = "FAIL", "weakens"
        if n_sig_small >= SEED_MAJORITY:
            label = "h1_weakened_advantage_below_minimal_effect"
            adv_txt = ("A trained-over-init advantage IS detected (significant on %d seeds) but "
                       "it is smaller than the pre-registered minimal effect of interest "
                       "everywhere" % n_sig_small)
        else:
            label = "h1_weakened_no_detectable_trained_advantage"
            adv_txt = "No trained-over-init advantage of the size H1 would need exists"
        summary = ("R5 green on every ready seed -- the consequence loop WAS open -- and the "
                   "paired DV is bounded inside (-%.2f, +%.2f) at one-sided 95%% on EVERY one "
                   "of the %d ready seeds (|mean paired DV| = %.4f, sign %s). %s: H1 WEAKENED "
                   "(equivalence), as operationalised by this lineage's training procedure. "
                   "C1neg was reachable on %d of %d ready seeds (init reach >= %.2f inside the "
                   "joint table); trained pair above the floor on %d seeds."
                   % (m, m, n_ready, abs(mean_dv), ("positive" if mean_dv > 0 else "non-positive"),
                      adv_txt, n_ready - n_c1neg_unreachable, n_ready, m, n_c2))
    else:
        outcome, direction = "FAIL", "unknown"
        label = "inconclusive_no_seed_majority_resolution"
        summary = ("Adjudicated on %d ready seeds: C1 on %d, C1neg on %d (reachable on %d), "
                   "two-sided equivalence on %d of %d, significant-but-below-%.2f advantage "
                   "on %d; mean paired DV %.4f. Neither a majority advantage nor a majority "
                   "deficit nor universal equivalence: inconclusive at this power (or "
                   "heterogeneous across seeds), in either direction. Per-seed p-values, "
                   "bounds, run and episode counts are in criteria[0].per_seed and "
                   "arm_results[].paired. Trained pair above the floor on %d seeds."
                   % (n_ready, n_c1, n_c1neg, n_ready - n_c1neg_unreachable, n_c4, n_ready, m,
                      n_sig_small, mean_dv, n_c2))

    disc_total = sum(r["paired"]["n_discordant"] for r in ready)
    non_degenerate = {
        "C1_trained_over_init_paired_exact": bool(adjudicated and disc_total > 0),
        "C1neg_init_over_trained_paired_exact": bool(adjudicated and disc_total > 0),
        # C4 is informative whenever the init readout reached selection inside the joint
        # table on every ready seed -- which R5 (now measured on the joint table) guarantees
        # for a ready seed -- so `adjudicated` is the condition (pass-2 F-7).
        "C4_two_sided_equivalence_all_ready_seeds": bool(adjudicated),
        "C2_trained_pair_itself_divergent": bool(adjudicated),
        "C3_fresh_seed_reproduces": bool(c3_applies),
    }
    return {"outcome": outcome, "label": label, "summary": summary,
            "evidence_direction": direction, "adjudicated": adjudicated,
            "instrument_green": adjudicated, "n_ready_seeds": n_ready,
            "criteria": criteria, "combination_rule": combination_rule,
            "criteria_non_degenerate": non_degenerate, "agg": agg,
            "pooled": pooled, "mean_dv_ready_seeds": mean_dv,
            "n_sig_below_min_effect": n_sig_small, "n_c1neg_unreachable": n_c1neg_unreachable,
            "legacy_across_seed_sd_margin": {"sd_init_pair_across_ready_seeds": sd_init,
                                             "margin_would_have_been": legacy_margin,
                                             "gates": "nothing -- recorded for 1029 comparability"},
            "red_by_seed": {str(k): v for k, v in red_by_seed.items()}}


# --------------------------------------------------------------------------------------
# RUN
# --------------------------------------------------------------------------------------
def _flat_scalar(adj: Dict[str, Any], rows: List[Dict[str, Any]]) -> Dict[str, Any]:
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
        if math.isfinite(f):
            out[k] = f

    for c in adj["criteria"]:
        put(c["name"] + "__measured", c.get("measured"))
        put(c["name"] + "__threshold", c.get("threshold"))
        put(c["name"] + "__passed", c.get("passed"))
    put("adjudicated", adj["adjudicated"])
    put("n_ready_seeds", adj["n_ready_seeds"])
    put("n_seeds_significant_below_min_effect", adj["n_sig_below_min_effect"])
    put("n_ready_seeds_c1neg_unreachable", adj["n_c1neg_unreachable"])
    he = [r["pairs"]["head_effect"].get("raw_identical_fraction_of_cofresh") for r in rows
          if r.get("pairs") and r["pairs"]["head_effect"].get("raw_identical_fraction_of_cofresh") is not None]
    if he:
        put("min__head_effect_raw_identical_fraction_of_cofresh", float(min(he)))
    xm = [r["paired"].get("cross_pair_raw_mismatch") for r in rows if r.get("paired")]
    if xm:
        put("sum__cross_pair_raw_mismatch_ticks", float(sum(xm)))
    put("n_seeds", len(rows))
    put("mean_dv_ready_seeds", adj["mean_dv_ready_seeds"])
    for k in ("b", "c", "n"):
        put("pooled_" + k, adj["pooled"][k])
    put("legacy_sd_init_pair_across_ready_seeds",
        adj["legacy_across_seed_sd_margin"]["sd_init_pair_across_ready_seeds"])
    put("legacy_margin_would_have_been",
        adj["legacy_across_seed_sd_margin"]["margin_would_have_been"])
    for key in ("trained_pair_argmin_divergence", "init_pair_argmin_divergence",
                "dv_trained_minus_init", "head_move_l2", "n_updates",
                "init_head_flip_fraction", "trained_head_flip_fraction",
                "sole_channel_rel_median"):
        vals = [r.get(key) for r in rows if r.get(key) is not None]
        if vals:
            put("mean__" + key, float(np.mean(vals)))
    return out


def run_experiment(episodes: Dict[str, int], seeds: List[int], dry_run: bool
                   ) -> Dict[str, Any]:
    t0 = time.perf_counter()
    # Design-time gate check: no precondition may be structurally unsatisfiable for any
    # seed (none carries a structural bound, so this is the module's own consistency
    # check, run before compute as the skill requires).
    assert_no_structurally_unsatisfiable_gate(_readiness_specs(),
                                              [{"seed": int(s)} for s in seeds])
    zg = ZGoalStreamAccumulator()
    rows = [_run_seed(s, episodes, zg) for s in seeds]
    adj = _adjudicate(rows)
    agg = adj["agg"]

    full_config = {
        "schedule": dict(episodes), "seeds": list(seeds),
        "lineage_seeds": LINEAGE_SEEDS, "fresh_seeds": FRESH_SEEDS,
        "modulatory_authority_gain": MODULATORY_AUTHORITY_GAIN,
        "use_modulatory_selection_authority": True,
        "modulatory_authority_normalize_basis": "range (default)",
        "use_gated_policy": USE_GATED_POLICY,
        "lr_lpfc_bias": LR_LPFC_BIAS,
        "reinforce": {"lr": LR_LPFC_BIAS, "batch": x1020.REINFORCE_BATCH_SIZE,
                      "buf_max": x1020.OUTCOME_BUF_MAX,
                      "adv_min": x1020.ADV_MIN_THRESHOLD, "ema": x1020.EMA_DECAY,
                      "surrogate": "1020 _reinforce_step: log_softmax(-bias/T) over the head's "
                                   "own output, credited at the E3-selected candidate; "
                                   "unchanged from the lineage (scope, not a fix)"},
        "runners": {k: {"ablated": v[0], "trained_head": v[1]} for k, v in RUNNERS.items()},
        "pairs": {k: list(v) for k, v in PAIRS.items()},
        "criteria": {"delta_abs_floor": DELTA_ABS_FLOOR, "alpha_one_sided": ALPHA_ONE_SIDED,
                     "z_one_sided_95": Z_ONE_SIDED_95,
                     "min_effect_of_interest": MIN_EFFECT_OF_INTEREST,
                     "seed_majority": SEED_MAJORITY, "min_ready_seeds": MIN_READY_SEEDS,
                     "legacy_delta_sd_multiple_recorded_only": DELTA_SD_MULTIPLE},
        "readiness": {"flip_fraction_floor": FLIP_FRACTION_FLOOR,
                      "authority_active_floor": AUTHORITY_ACTIVE_FLOOR,
                      "min_paired_ticks": MIN_RAW_IDENTICAL_CO_FRESH,
                      "init_reach_floor": INIT_REACH_FLOOR, "min_updates": MIN_UPDATES,
                      "head_move_floor": HEAD_MOVE_FLOOR,
                      "sole_channel_rel_tol": SOLE_CHANNEL_REL_TOL},
        "p1_budget_source": P1_BUDGET_SOURCE_AUTOPSY,
        "p1_budget_source_status": P1_BUDGET_SOURCE_STATUS,
        "env_kwargs": dict(x1020.ENV_KWARGS), "dry_run": bool(dry_run),
    }

    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    if adj["adjudicated"]:
        precond_list = list(agg["adjudication_preconditions"])
    else:
        precond_list = [p for g in (r["readiness_gate"] for r in rows) for p in g["preconditions"]]
    manifest: Dict[str, Any] = {
        "run_id": "%s_%s_v3" % (EXPERIMENT_TYPE, ts),
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "architecture_epoch": ARCHITECTURE_EPOCH,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "sleep_driver_pattern": SLEEP_DRIVER_PATTERN,
        "claim_ids": list(CLAIM_IDS),
        "fanout_qid": FANOUT_QID,
        "fanout_hypotheses": list(FANOUT_HYPOTHESES),
        "fanout_axis": FANOUT_AXIS,
        "fanout_source_autopsy": FANOUT_SOURCE_AUTOPSY,
        "p1_budget_source_autopsy": P1_BUDGET_SOURCE_AUTOPSY,
        "p1_budget_source_status": P1_BUDGET_SOURCE_STATUS,
        "source_chip_ref": SOURCE_CHIP_REF,
        "outcome": adj["outcome"],
        "evidence_direction": adj["evidence_direction"],
        "timestamp_utc": ts,
        "dry_run": bool(dry_run),
        "non_degenerate": bool(adj["adjudicated"]),
        "degeneracy_reason": ("" if adj["adjudicated"] else agg["degeneracy_reason"]),
        "interpretation": {
            "label": adj["label"],
            "summary": adj["summary"],
            "combination_rule": adj["combination_rule"],
            "preconditions": precond_list,
            # The library's disclosure for a green-seeds-only list, at the key the indexer
            # reads (pass-2 red-team F-8), plus the partial-readiness text when adjudicated
            # on a subset.
            "preconditions_scope_note": agg["per_arm_gate"]["preconditions_scope_note"],
            "partial_readiness_note": (agg["degeneracy_reason"]
                                       if (adj["adjudicated"] and agg["red_arms"]) else ""),
            "criteria_non_degenerate": adj["criteria_non_degenerate"],
        },
        "criteria": adj["criteria"],
        "combination_rule": adj["combination_rule"],
        "per_arm_gate": agg["per_arm_gate"],
        "n_ready_seeds": adj["n_ready_seeds"],
        "seeds_intended": list(seeds),
        "arm_results": rows,
        "per_seed_results": [{k: v for k, v in r.items() if k not in ("runners", "readiness_gate")}
                             for r in rows],
        "diagnostics": {
            "x1029_on_pair_reference_by_seed": X1029_ON_PAIR_BY_SEED,
            "legacy_across_seed_sd_margin": adj["legacy_across_seed_sd_margin"],
            "pooled_paired_table_ready_seeds": adj["pooled"],
            "red_gates_by_seed": adj["red_by_seed"],
            "magnitude_arm_omitted_because": (
                "closed from above by the substrate's 0.1 tanh bound "
                "(failure_autopsy_V3-EXQ-1027-1029-cluster_2026-09-14) and moot under "
                "authority, which normalises the applied modulatory range per runner -- the "
                "DV is a bias SHAPE comparison"),
            "committed_fraction_by_runner_seed0": {
                k: v.get("committed_fraction") for k, v in rows[0]["runners"].items()} if rows else {},
        },
        "readout": {},
    }
    manifest["readout"] = _flat_scalar(adj, rows)

    out_path = write_flat_manifest(
        manifest, dry_run=dry_run, config=full_config, seeds=list(seeds),
        script_path=Path(__file__), started_at=t0,
        z_goal_stream_stats=zg.stats(),
    )
    return {"outcome": adj["outcome"], "manifest_path": out_path,
            "label": adj["label"], "summary": adj["summary"]}


# --------------------------------------------------------------------------------------
# DESIGN-TIME SELF-TEST (runs before any compute; also `--self-test`)
# --------------------------------------------------------------------------------------
def _synthetic_row(seed: int, b: int, c: int, n: int, t_div: float, i_div: float,
                   red: Optional[str] = None) -> Dict[str, Any]:
    """A row with exactly the fields `_adjudicate` reads, for grid checks."""
    gate = {"arm": "seed%d" % seed, "gate_green": red is None, "structurally_vacuous": False,
            "vacuity_reason": "", "preconditions": [], "scoped_out": [],
            "failed_preconditions": ([red] if red else []), "n_applied": 8, "n_scoped_out": 0}
    # `both` is set so that the init pair's in-joint reach equals i_div (reachability, F-4).
    both = max(int(round(i_div * n)) - c, 0)
    paired = _paired_stats({"n": n, "b": b, "c": c, "both": both,
                            "neither": max(n - b - c - both, 0)})
    return {"seed": seed, "paired": paired, **_seed_flags(paired), "readiness_gate": gate,
            "trained_pair_argmin_divergence": t_div, "init_pair_argmin_divergence": i_div,
            "seed_family": ("lineage" if seed in LINEAGE_SEEDS else "fresh")}


def _self_test() -> int:
    """Design-time arithmetic that must hold before any compute is spent."""
    fails: List[str] = []
    if QUEUE_ID.endswith("a") or QUEUE_ID.endswith("b"):
        fails.append("the cluster autopsy REFUSES a lettered successor by name")
    if len(SEEDS) != len(set(SEEDS)):
        fails.append("duplicate seed")
    if not set(LINEAGE_SEEDS).issubset(SEEDS) or not set(FRESH_SEEDS).issubset(SEEDS):
        fails.append("seed factor is not fully represented")
    if SEED_MAJORITY <= len(SEEDS) // 2:
        fails.append("SEED_MAJORITY %d is not a majority of %d" % (SEED_MAJORITY, len(SEEDS)))
    if MIN_READY_SEEDS != SEED_MAJORITY:
        fails.append("MIN_READY_SEEDS must equal SEED_MAJORITY (absolute bar)")
    if 44 in SEEDS:
        fails.append("seed 44 is a known-unstable reef seed; substitute 45")
    if PAIRS["trained_pair"][0] == PAIRS["init_pair"][0]:
        fails.append("trained and init pairs share their intact member")
    for p, (a, b) in PAIRS.items():
        for n in (a, b):
            if n not in RUNNERS:
                fails.append("pair %s names unknown runner %s" % (p, n))
    if RUNNERS[R_REF_SELF] != RUNNERS[REFERENCE]:
        fails.append("the self-yoke is not configured identically to the reference")
    declared = [k for k, v in RUNNERS.items() if v[1] and k != REFERENCE]
    if sorted(declared) != sorted(TRAINED_MIRRORS):
        fails.append("TRAINED_MIRRORS %r does not match the declared trained-head runners %r"
                     % (sorted(TRAINED_MIRRORS), sorted(declared)))
    if any("MAG" in k for k in RUNNERS):
        fails.append("a magnitude arm is present; the cluster autopsy closes it from above")
    if USE_GATED_POLICY:
        fails.append("gated_policy is ON; authority would amplify a second channel")
    if P1_EPISODES != 300:
        fails.append("P1_EPISODES %d is not the autopsy's 300" % P1_EPISODES)
    if not (0.0 < ALPHA_ONE_SIDED <= 0.1):
        fails.append("ALPHA_ONE_SIDED out of range")
    if MIN_EFFECT_OF_INTEREST <= DELTA_ABS_FLOOR:
        fails.append("MIN_EFFECT_OF_INTEREST must exceed the effect floor")
    if set(_PRECONDITION_COMPARATORS) != {s.name for s in _readiness_specs()}:
        fails.append("precondition comparator table does not match the specs")
    # Exact tail arithmetic.
    if abs(_binom_tail_ge(0, 5) - 1.0) > 1e-12 or abs(_binom_tail_ge(5, 5) - 1 / 32) > 1e-12 \
            or abs(_binom_tail_ge(3, 4) - 5 / 16) > 1e-12:
        fails.append("_binom_tail_ge arithmetic wrong")
    # Only C1 is load-bearing (pass-1 red-team F2-A: C2 is implied by C1 under R5).
    probe_rows = [_synthetic_row(s, 30, 5, 300, 0.12, 0.04) for s in SEEDS]
    adj = _adjudicate(probe_rows)
    lb = [c["name"] for c in adj["criteria"] if c.get("load_bearing")]
    if lb != ["C1_trained_over_init_paired_exact"]:
        fails.append("load-bearing set is %r, expected C1 only" % lb)
    # GRID CHECKS -- every branch reachable and routed as the NULL TABLE states.
    grid = [
        ("PASS main", [_synthetic_row(s, 30, 5, 300, 0.12, 0.04) for s in SEEDS],
         "PASS", "supports", "h1_consequence_trained_readout_more_argmin_consequential"),
        ("PASS lineage-only", [_synthetic_row(s, 30, 5, 300, 0.12, 0.04) for s in LINEAGE_SEEDS]
         + [_synthetic_row(s, 5, 5, 300, 0.04, 0.04) for s in FRESH_SEEDS],
         "PASS", "supports",
         "h1_consequence_trained_readout_more_argmin_consequential_lineage_seeds_only"),
        ("weakens training reduced (probe pattern)", [_synthetic_row(s, 0, 30, 300, 0.0, 0.10) for s in SEEDS],
         "FAIL", "weakens", "h1_weakened_training_reduced_readout_consequence"),
        ("weakens equivalence", [_synthetic_row(s, 3, 3, 300, 0.05, 0.05) for s in SEEDS],
         "FAIL", "weakens", "h1_weakened_no_detectable_trained_advantage"),
        ("weakens advantage below min effect (pass-2 F-3 shape)",
         [_synthetic_row(s, 8, 0, 300, 0.08, 0.05) for s in SEEDS],
         "FAIL", "weakens", "h1_weakened_advantage_below_minimal_effect"),
        ("unknown: 4 small-positive + 3 large-negative (pass-2 F-2 shape)",
         [_synthetic_row(s, 8, 0, 300, 0.08, 0.05) for s in SEEDS[:4]]
         + [_synthetic_row(s, 0, 40, 300, 0.0, 0.13) for s in SEEDS[4:]],
         "FAIL", "unknown", "inconclusive_no_seed_majority_resolution"),
        ("unknown inconclusive (positive near-miss)", [_synthetic_row(s, 12, 6, 60, 0.25, 0.15) for s in SEEDS],
         "FAIL", "unknown", "inconclusive_no_seed_majority_resolution"),
        ("insufficient seeds (3 all green)", [_synthetic_row(s, 30, 5, 300, 0.12, 0.04) for s in SEEDS[:3]],
         "FAIL", "non_contributory", "insufficient_seeds_run"),
        ("refusal gate red mix (3 green, 2 R5, 2 R7)",
         [_synthetic_row(s, 30, 5, 300, 0.12, 0.04) for s in SEEDS[:3]]
         + [_synthetic_row(s, 0, 0, 300, 0.0, 0.0, red="init_readout_reaches_selection") for s in SEEDS[3:5]]
         + [_synthetic_row(s, 30, 5, 300, 0.12, 0.04, red="sole_modulatory_channel") for s in SEEDS[5:]],
         "FAIL", "non_contributory", "instrument_refused_gate_red"),
        ("PASS on 4 ready of 7 (one fresh)",
         [_synthetic_row(s, 30, 5, 300, 0.12, 0.04) for s in (611, 622, 633, 666)]
         + [_synthetic_row(s, 0, 0, 300, 0.0, 0.0, red="init_readout_reaches_selection")
            for s in (644, 655, 677)],
         "PASS", "supports", "h1_consequence_trained_readout_more_argmin_consequential"),
        ("PASS on 4 ready of 7 (all lineage)", [_synthetic_row(s, 30, 5, 300, 0.12, 0.04) for s in SEEDS[:4]]
         + [_synthetic_row(s, 0, 0, 300, 0.0, 0.0, red="init_readout_reaches_selection") for s in SEEDS[4:]],
         "PASS", "supports",
         "h1_consequence_trained_readout_more_argmin_consequential_lineage_seeds_only"),
        ("refusal R5 on 4 of 7", [_synthetic_row(s, 30, 5, 300, 0.12, 0.04) for s in SEEDS[:3]]
         + [_synthetic_row(s, 0, 0, 300, 0.0, 0.0, red="init_readout_reaches_selection") for s in SEEDS[3:]],
         "FAIL", "non_contributory", "substrate_not_ready_requeue"),
        ("refusal R6", [_synthetic_row(s, 30, 5, 300, 0.12, 0.04) for s in SEEDS[:3]]
         + [_synthetic_row(s, 30, 5, 300, 0.12, 0.04, red="head_moved_during_p1") for s in SEEDS[3:]],
         "FAIL", "non_contributory", "instrument_refused_head_did_not_move"),
    ]
    for name, rows, exp_out, exp_dir, exp_label in grid:
        a = _adjudicate(rows)
        if (a["outcome"], a["evidence_direction"], a["label"]) != (exp_out, exp_dir, exp_label):
            fails.append("grid '%s': got (%s, %s, %s), expected (%s, %s, %s)"
                         % (name, a["outcome"], a["evidence_direction"], a["label"],
                            exp_out, exp_dir, exp_label))
    # A C1 pass carried by fewer than SEED_MAJORITY significant seeds must NOT pass.
    a = _adjudicate([_synthetic_row(s, 30, 5, 300, 0.12, 0.04) for s in SEEDS[:3]]
                    + [_synthetic_row(s, 8, 6, 300, 0.05, 0.04) for s in SEEDS[3:]])
    if a["outcome"] == "PASS":
        fails.append("C1 passed on 3 significant seeds")
    # Mutual exclusion of the per-seed flags, exhaustively over small tables (F-1).
    for n in (60, 300):
        for b in range(0, 41):
            for c in range(0, 41):
                f = _seed_flags(_paired_stats({"n": n, "b": b, "c": c, "both": 0,
                                               "neither": max(n - b - c, 0)}))
                if sum((f["c1_seed"], f["c1neg_seed"], f["c4_equiv_seed"])) > 1:
                    fails.append("per-seed flags not exclusive at n=%d b=%d c=%d" % (n, b, c))
                    break
    # A PASS manifest must never carry C4 passed:true (F-1).
    if any(c["name"].startswith("C4") and c["passed"] for c in adj["criteria"]):
        fails.append("C4 passed alongside a C1 PASS")
    for f in fails:
        print("  [self-test] FAIL %s" % f, flush=True)
    if not fails:
        print("  [self-test] PASS all design-time checks (incl. %d grid branches)" % len(grid),
              flush=True)
    return 1 if fails else 0


def main() -> Dict[str, Any]:
    ap = argparse.ArgumentParser(description="V3-EXQ-1046 SD-082 consequence-trained readout")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="*", default=None,
                    help="subset for probes ONLY; SEED_MAJORITY stays absolute (4 of 7)")
    args = ap.parse_args()

    if args.self_test:
        sys.exit(_self_test())
    if _self_test() != 0:
        raise RuntimeError("design-time self-test failed; refusing to spend compute")

    seeds = args.seeds if args.seeds else (DRY_RUN_SEEDS if args.dry_run else SEEDS)
    episodes = (DRY_RUN_EPISODES if args.dry_run else
                {"p0": P0_EPISODES, "p1": P1_EPISODES, "score": SCORE_EPISODES,
                 "steps": STEPS_PER_EPISODE})
    result = run_experiment(episodes, seeds, dry_run=args.dry_run)
    print("")
    print("outcome: %s" % result["outcome"], flush=True)
    print("label: %s" % result["label"], flush=True)
    print("summary: %s" % result["summary"], flush=True)
    print("manifest: %s" % result["manifest_path"], flush=True)
    result["dry_run"] = bool(args.dry_run)
    return result


if __name__ == "__main__":
    _result = main()
    _outcome_raw = str(_result["outcome"]).upper()
    emit_outcome(
        outcome=(_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL"),
        manifest_path=_result["manifest_path"],
        queue_id=QUEUE_ID,
        dry_run=_result["dry_run"],
    )
