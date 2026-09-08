"""V3-EXQ-1010 -- THE OVER-CAPACITY DECODER SWEEP: is the oracle's decision-relevant content
recoverable from the frozen 978-OFF z_world at ANY decoder capacity?

Adjudicates `H-F-content-discarded-at-encode`, the single leg left live on the frozen-ledger
question `zworld_actor_adequacy_locus` after V3-EXQ-1008 eliminated H-E and split H-C.

  MANIPULATION: the DECODER'S CAPACITY, and nothing else.
  HELD FIXED:   the representation, the dataset, the seeds, the held-out episode split, the
                standardiser, the fit protocol (Adam, ADAPTER_LR, ADAPTER_BATCH, ADAPTER_PASSES),
                and the oracle labels. All imported from V3-EXQ-1002 / V3-EXQ-1008, never
                re-defined.

Question (`hypothesis_space_registry.v1.json` qid `zworld_actor_adequacy_locus`):
  H-B eliminated (V3-EXQ-1002), H-D confirmed on one run (V3-EXQ-1002), H-E ELIMINATED and
  H-C SPLIT (V3-EXQ-1008, confirmed autopsy failure_autopsy_V3-EXQ-1008_2026-09-08, user-gated
  2026-09-08T06:18:57Z). H-F is that autopsy's own labelled fan-out growth (3a), axis
  `representation`, registered `alive` with `adjudicating_runs: []`. THIS RUN adjudicates it.
  Leg hid: H-F-content-discarded-at-encode.

Claims: NONE (claim_ids = []). Bears on INV-088 / MECH-457 exactly as V3-EXQ-1002 and
V3-EXQ-1008 wrote them -- both are PERIPHERAL co-tags per the 978 autopsy, MECH-457's re-derive
brake stands at 13 and INV-088's at 2, and re-attaching either here would increment a brake
counter on a run that exercises neither claim's mechanism. What this run adjudicates is a
hypothesis-space leg, not a claim.

EXPERIMENT_PURPOSE = "diagnostic"

SLEEP DRIVER: not applicable -- no sleep flag is set (the x734 all-ON stack at this rung enables
no sleep loop). Recorded as sleep_driver_pattern="none".

red-team (fable): see the REPAIR RECORD at the end of this docstring and the V3-EXQ-1010 queue
entry note.

=== WHY THIS RUN, AND WHY IT IS NOT A V3-EXQ-1008b ===

The confirmed V3-EXQ-1008 autopsy refuses, in its own section 7, "a lettered V3-EXQ-1008b that
re-runs this design with more adapter passes, a wider adapter, more seeds or more episodes".
This is not that run, and the distinction is structural rather than rhetorical:

    V3-EXQ-1008 held the READER'S CAPACITY FIXED at the consumer's exact width (a deliberate
    `capacity_match` requirement inherited from V3-EXQ-1002) and varied THE INPUT REPRESENTATION
    across ten arms.

    V3-EXQ-1010 holds THE REPRESENTATION FIXED and varies THE READER'S CAPACITY.

The DV, the question and the manipulated variable all change. Section 4(c) of that autopsy
states the gap in terms this run is built to close: "What no run in this lineage has done is
VARY THE READER'S CAPACITY: 1008 held it fixed at the consumer's exact width as a deliberate
capacity_match requirement inherited from 1002." And section 8: "the residual question is sharp,
single and unmeasured: is the decision-relevant content recoverable from the frozen latent by an
over-capacity decoder?"

That single fact gates two builds whose costs differ by more than an order of magnitude:
  H-F CONFIRMED -> the content is destroyed at encode time; the repair is at the ENCODER'S
                   OBJECTIVE (`sd_actor_critic_action_learning`'s co-shaping hint, or a separate
                   action-formatted stream). Expensive.
  H-F ELIMINATED -> the content survives but is inaccessible at the consumer's capacity/format;
                   the repair is INTERFACE REFORMATTING. Much cheaper.

=== WHAT "BANKED DATA" MEANS HERE, STATED HONESTLY (inherited from V3-EXQ-1008) ===

V3-EXQ-1002 and V3-EXQ-978 persisted no observations and no encoder weights. What IS banked is
the RECIPE, and it is deterministic: `x1002._collect_episodes` drives the oracle / a random
policy from env seeds seed*1000+ep with the policies seeded from `seed`, so re-collecting
reproduces the identical step-for-step dataset. The frozen 978-OFF latent is REPRODUCED by
re-running the imported 978 warmup (60 P0a + 200 P0 + 90 P1 episodes per seed, every constant
imported from x734/x808/x724 exactly as 1002 and 1008 did) -- the same warmup FAMILY, not a new
one, paid once per seed and shared by all five OFF-track rungs. V3-EXQ-1008 established that this
reproduction is BIT-IDENTICAL to V3-EXQ-1002's (`zworld_encoder_trained_in_p0` and
`zworld_not_collapsed` matching to full double precision, identical held-out split, bit-identical
`rawfield_ceiling` on 3/3 seeds), so the OFF latent this run reads is the same object 1002 and
1008 read. "No new rollouts" would be an over-compression; "no ree_core change and no new warmup
family" is exact.

NOTE ON PROVENANCE, CORRECTING V3-EXQ-1008 AT SOURCE: 1008's driver docstring justified "not a
bit-identical replay" partly on the ground that "the machine class differ". IT DOES NOT -- both
1002 and 1008 declare `linux-x86_64-py3.10-torch2.12.0+cpu`. That wrong string is not repeated
here (autopsy section 1, red-team FINDING 1; chip-20260908-exq1008-provenance-string).

=== THE CAPACITY LADDER (5 rungs, identical on both tracks) ===

Every rung is fitted by cross-entropy behaviour-cloning of the oracle's action, with x1002's
own optimiser settings and pass count, on x1002's own train/test episode split.

  `linear`   a fitted LINEAR softmax readout (in_dim -> action_dim, no hidden layer).
             THE RED-TEAM ADDITION THE 1008 AUTOPSY COMMISSIONED (section 4(c), finding 5):
             1008's closed-form content witness applies the oracle's HAND-SPECIFIED argmax rule
             to a linear decode and reaches 0.593-0.607, while the 32->128 adapter reaches 0.669.
             Those two readers differ in TWO ways -- learned-vs-fixed rule AND nonlinear-vs-linear
             -- so the 0.054-0.083 gap is "suggestive of, but does not isolate, nonlinear
             content". A FITTED LINEAR readout differs from the witness in exactly one of those
             two ways, which is what de-confounds them.
  `mlp128`   `x734.PPOPolicyNet` at `x734.PPO_TRUNK_HIDDEN` (128) -- the consumer's exact reader,
             the class V3-EXQ-978 instantiated. THE REPRODUCTION ANCHOR: 1008 measured
             0.6718/0.6735/0.6606 (OFF) and 0.8771/0.8578/0.8702 (PCA-32) at this rung.
  `mlp512`   the same class at hidden=512.
  `mlp2048`  the same class at hidden=2048.
  `deep2048x4` a deliberately OVER-PARAMETERISED decoder: four tanh hidden layers at 2048.

Rungs 2-4 use `x734.PPOPolicyNet` ITSELF (not a look-alike), so the `mlp128` rung is the
consumer's reader by construction and reproduces 1008's number rather than approximating it.

=== THE THREE TRACKS, AND WHY THE OTHER TWO ARE LOAD-BEARING ===

  `zworld_off`        the frozen 978-OFF z_world (32 dims), train-split diagonal z-score.
                      ** THE SUBJECT. ** 1008 at mlp128: 0.6686 mean, 0/3 seeds clear.
  `ws250_pca`         PCA-32 of the encoder's own 250-dim `world_state` input (32 dims), the
                      same train-split-fitted projection 1008 used, then the same guarded
                      z-score. ** THE CALIBRATION ANCHOR (certifies the PROTOCOL). ** 1008 at
                      mlp128: 0.8684 mean, 3/3 seeds clear.
  `zworld_untrained`  the SAME latent construction with the warmup SKIPPED (32 dims), same
                      z-score. ** THE NEGATIVE CONTROL (attributes an ELIMINATED verdict). **
                      1008 at mlp128: 0.699/0.681/0.704 -- note this is ABOVE the trained OFF
                      latent on every seed. Costs NO warmup, so the whole track is cheap.

All three tracks are 32-DIMENSIONAL, read the same stored steps, use the same standardiser and
the same fit protocol. The ONLY difference is which 32-dim representation the decoder reads.

Why each of the two controls is needed, stated separately because they guard opposite verdicts:

  THE ANCHOR guards CONFIRMED. Without it, a flat OFF track is uninterpretable: is the content
  absent, or does this fit protocol simply not support over-capacity decoders on this
  training-set size? The anchor answers that on a representation known to carry the content.

  THE NEGATIVE CONTROL guards ELIMINATED. Because 1008 measured the UNTRAINED latent ABOVE the
  trained one, "OFF clears at high capacity" is by itself indistinguishable from "any 250->32
  bottleneck preserves enough for a 4M-parameter decoder". If both clear, the finding is about
  the BOTTLENECK, not about what the encoder's training discarded -- and the interface-reformat
  build would be licensed on the wrong premise. `off_minus_untrained_best` is recorded per seed
  for exactly this read.

  A SIXTEENTH ARM, `rawfield_ceiling` at mlp128, is the instrument's POSITIVE CONTROL and gate
  (raw 25-dim resource field; 1008 measured 0.9791 mean). Floor 0.60.

  16 arms per seed = 1 instrument gate + 3 tracks x 5 capacity rungs.

=== THE PRE-REGISTERED VERDICT, AND ITS TWO ANTI-ALIASING GUARDS ===

VERDICT STATISTIC: `best_agreement_over_capacity` = per (track, seed), the MAXIMUM held-out
oracle-action agreement over the five rungs. NOT the largest rung -- an over-parameterised
decoder on a fixed dataset may overfit and fall, so "at any capacity" is a max, not a limit.

  H-F CONFIRMED   the OFF track's best rung stays BELOW `AGREEMENT_BAR` (0.80), or below the
                  `AGREEMENT_ELEVATION_MIN` (0.20) elevation over the strongest trivial
                  predictor, on the seed MAJORITY (>= 2 of 3) -- while both guards below hold.
                  Reading: the decision-relevant content is DESTROYED AT ENCODE TIME. Route to
                  a build at the encoder's objective.
  H-F ELIMINATED  some OFF rung clears bar AND elevation on the seed majority. Reading: the
                  content SURVIVES the encode and is inaccessible at the consumer's
                  capacity/format. Route to an interface-reformat build. The rung at which it
                  first clears is reported, because it sizes the reformat.
                  ** READ IT AGAINST THE NEGATIVE CONTROL. ** If the UNTRAINED track clears at
                  the same capacity (`off_minus_untrained_best` near zero or negative), the
                  finding is about the 250->32 BOTTLENECK rather than about what the encoder's
                  training discarded, and the reformat build is licensed on the wrong premise.
                  That is a reading the verdict does not gate on -- it is stated here, and the
                  number is recorded per seed, so the governance session cannot miss it.
  UNDETERMINED    a verdict arm is red (a failed encoder-health precondition or a failed
                  reproduction band), either guard fails, or fewer than SEED_MAJORITY seeds ran.

GUARD 1 -- `overcapacity_fit_protocol_sound` (the anchor certifies the PROTOCOL, not the subject).
  The ws250_pca track's agreement AT THE OVER-PARAMETERISED RUNG (`deep2048x4`) must not fall
  more than ANCHOR_DEGRADE_TOL (0.10) below its OWN `mlp128` value. If the anchor collapses
  under added capacity, this fit protocol cannot support over-capacity decoders at all and the
  OFF track's flatness is an instrument artifact, not a content finding.
  THE STATISTIC IS max-capacity MINUS consumer, NOT best MINUS consumer. The obvious spelling
  ("its best rung, relative to mlp128") is degenerate: `best` is a max over a set that CONTAINS
  the consumer rung, so that difference is >= 0 by construction and the guard could never fire.
  Note also WHY this is not a gate that certifies its own subject (the fourth red-team family):
  the anchor is a DIFFERENT REPRESENTATION at the SAME width under the SAME protocol, so it is
  a positive control for the manipulation class, and it is excluded from the verdict statistic
  entirely. Its known limit is recorded rather than smoothed: the anchor is an EASIER map than
  the OFF latent (1008 measured OFF train agreement 0.743-0.766 against PCA's 0.961-0.968 at
  the same width), so Guard 1 certifies "the protocol fits SOMETHING at every width", not "the
  protocol fits THIS latent at every width". That second question is Guard 2's, and it is
  measured on the OFF track itself.

GUARD 2 -- `overcapacity_decoder_can_memorise` (the decoder demonstrably HAS the fit power).
  The OFF track's TRAIN-split agreement, MAXIMISED OVER RUNGS, must reach MEMORISE_FLOOR (0.95)
  on the worst seed. This is the single strongest protection against the reading "the null was
  produced by an under-powered reader": if a decoder reading the frozen latent can memorise the
  training labels almost perfectly and still not generalise, its failure is about GENERALISABLE
  CONTENT IN THE REPRESENTATION, not about fit capacity. 1002 measured 0.828 train agreement for
  the UNTRAINED latent at mlp128, so this floor is a real ask that the ladder is built to meet,
  not a formality.
  MAX OVER RUNGS, not the top rung's own value: the claim Guard 2 needs is "the LADDER reached
  over-capacity", which any one rung memorising establishes, and keying it to a single rung
  would let one optimiser artifact discard an otherwise sound run. Per-rung train agreement AND
  per-rung final CE are recorded on every track either way (`off_train_agreement_by_rung`,
  `off_final_ce_by_rung`, `anchor_final_ce_by_rung`), plus an explicit per-rung `diverged` flag
  (final CE at or above the uniform-logit value ln(action_dim)) and the run-level
  `guards.memorising_rungs` / `guards.diverged_rungs` sets -- so a divergent rung is visible to
  a later reader rather than silently absorbed by the max.
  If Guard 2 fails, the run self-routes `substrate_not_ready_requeue` -- NEVER H-F CONFIRMED.

=== THE TOP-RUNG TRAINABILITY MEASUREMENT (authoring time, full scale) ===

The red-team asked, correctly, whether holding the optimiser fixed across a 77,000x parameter
range confounds CAPACITY with TRAINABILITY -- and pointed at the dry-run smoke, where
`deep2048x4` reached CE 4.38 against ln(5)=1.609 (worse than uniform logits, i.e. divergence).

That was settled by MEASUREMENT rather than argument, because the dry-run cannot answer it: the
smoke runs 3 passes over a handful of episodes, so a 12.67M-parameter net is nonsense there by
construction. `experiments/_scratch/exq1010_topring_trainability_probe.py` re-ran the ANCHOR
track -- which needs NO warmup, so it is cheap, and which is Guard 1's own subject -- at the
FULL dataset recipe and the real ADAPTER_PASSES, seed 42, with GRAD_CLIP_NORM in place:

  n_train_rows 5038, n_heldout_rows 2148   (1008 measured 5038/4997/4423 train -- reproduced)

  rung          params      final_ce   diverged   train_agree   heldout_agree   wall
  mlp128        21,381      0.1061     False      0.9653        0.8836          1.7 s
  mlp2048       4,274,181   0.0027     False      1.0000        0.8780          24.7 s
  deep2048x4    12,666,885  0.0111     False      0.9978        0.8641          66.1 s

Four things follow, and all four matter to how this run is read:
  (1) THE DIVERGENCE WAS A DRY-SCALE ARTIFACT. At full scale every rung fits; none diverges.
      The ladder's nominal top rung is also its real top rung.
  (2) THE OVER-CAPACITY RUNGS DEMONSTRABLY MEMORISE (train 0.9978-1.0000, against the 0.95
      MEMORISE_FLOOR), so Guard 2's mechanism is shown to be reachable rather than hoped for.
  (3) GUARD 1 IS COMFORTABLE BUT NOT VACUOUS: the anchor's max-capacity-minus-consumer delta is
      0.8641 - 0.8836 = -0.0195, inside the -0.10 tolerance with room, and NEGATIVE -- added
      capacity mildly overfits rather than helping, which is exactly the behaviour the guard
      exists to bound and confirms the tolerance is doing real work at a plausible value.
  (4) HELD-OUT AGREEMENT IS FLAT-TO-SLIGHTLY-DECLINING IN CAPACITY on a representation that
      DOES carry the content (0.8836 -> 0.8780 -> 0.8641). So "more capacity does not help" is
      the NORMAL shape here, not a signature of absent content -- which is precisely why the
      verdict is the absolute bar plus elevation, and never a within-track capacity trend.

Caveat recorded rather than smoothed: these numbers are close to but not identical with 1008's
(anchor mlp128 0.8836 here vs 0.8771 there, seed 42) because GRAD_CLIP_NORM is a real change to
the fit protocol. Every threshold in this run is applied to THIS run's own in-run values, never
to 1008's; `reference_values_1008` is recorded for orientation only and is used by nothing.

Both guards are readiness-kind preconditions with numeric measured/threshold, so the indexer
recomputes `met` from the numbers rather than trusting the author's boolean.

=== THE SECOND, READER-FREE CONTENT WITNESS (nonlinear, at every rung) ===

The autopsy's routing asks for "both oracle-action agreement AND a direct NONLINEAR decode of
the five `DECISION_WORLD_STATE_INDICES`". At every OFF-track rung this run additionally fits a
decoder OF THE SAME CAPACITY from the frozen latent to the 25-dim resource field, and then
applies THE ORACLE'S OWN ARGMAX RULE to the decoded field with no fitted action head at all
(`x1008._oracle_rule_actions` / `_decode_oracle_agreement`). That is the exact nonlinear
analogue of 1008's LINEAR closed-form content witness (which degraded to 0.5931/0.5910/0.6066
against a 0.5661/0.5803/0.5720 trivial baseline), and it is reader-free by construction: no
learned action mapping can launder content into it.

  Recorded per rung: `nonlinear_decode_oracle_agreement`, `nonlinear_decode_field_r2`,
  `decision_coord_r2` (the five oracle-read coordinates specifically, which is where the
  decision lives -- 1008's correction to V3-EXQ-978 was precisely that a bulk field-decode r2
  averaged over 25 coordinates is NOT evidence of decision-relevant content adequacy).

The witness is a RECORDED READOUT, not a verdict conjunct. The verdict is on agreement alone, as
pre-registered above; the witness is what makes a CONFIRMED verdict readable as "content absent"
rather than merely "this reader could not find it".

=== THE NULL TABLE: every branch is informative ===

  OFF best < bar, anchor sound, decoder memorises   -> H-F CONFIRMED. Build at the encoder
                                                       objective. The expensive branch, and the
                                                       one this run exists to be able to assert.
  OFF best >= bar on the majority                   -> H-F ELIMINATED. Build the interface
                                                       reformat. Report the first clearing rung.
  Anchor degrades past tolerance                    -> instrument not ready; requeue with a fit
                                                       protocol that supports these capacities.
                                                       NOT a content finding.
  Deep decoder cannot memorise                      -> substrate_not_ready_requeue. The ladder
                                                       did not reach over-capacity; extend it.
  Fewer than SEED_MAJORITY seeds                    -> no verdict (the majority is never lowered).

=== DV-SYMMETRY INVARIANCE (mandatory declaration, per arm) ===

DV = held-out per-state top-1 agreement between the decoder's argmax and the oracle's discrete
action. Symmetry group of that DV: any transform of the decoder's logits that preserves their
per-state argmax -- a broadcast additive constant, a positive monotone rescaling, and a
permutation of interchangeable units.

THE MANIPULATION IS NOT INVARIANT UNDER ANY OF THEM, on every arm. It is a change of the
decoder's FUNCTION CLASS (parameter count and depth), which changes which functions of the input
the fit can express and therefore which held-out predictions it makes. It is not a constant added
to the logits (a strictly larger function class is not an offset), not a monotone rescaling of a
ranked quantity (the ranking itself is what changes), and not a permutation of interchangeable
units (the rungs are ordered by capacity and are not exchangeable). The manipulation reaches the
DV through the fitted decoder's held-out argmax, which is the quantity the DV is.

Two per-arm notes, because the two tracks differ in what the manipulation can show:
  - `zworld_off` rungs: the representation is a FROZEN 32-dim latent, so the only thing capacity
    can add is nonlinear expressivity over that latent. That is precisely H-F's question.
  - `ws250_pca` rungs: identical statement, on a representation known to support 0.868 at the
    smallest MLP rung. The manipulation is equally live there; it is the control for whether
    added capacity HURTS under this protocol.
  - `rawfield_ceiling`: single rung, no capacity manipulation -- it is the instrument gate, and
    is excluded from every capacity comparison and from the verdict.

The reader-free nonlinear witness has NO fitted action head in its path at all, so it is not
subject to the logit-symmetry group in the first place.

=== WHAT THIS RUN CANNOT SAY, STATED BEFORE IT RUNS ===

"At ANY capacity" is delivered as "at any capacity IN THIS LADDER, under THIS fit protocol, on
THIS training-set size". The ladder spans a fitted linear readout to a four-layer 2048-wide MLP:
MEASURED at authoring time by `--self-test` at in_dim=32, action_dim=5, the action-path parameter
counts are

    linear 165 -> mlp128 21,381 -> mlp512 282,117 -> mlp2048 4,274,181 -> deep2048x4 12,666,885

i.e. a ~77,000x range, with the top rung carrying ~1,600x the consumer's parameters. Against
V3-EXQ-1008's OWN measured training-split sizes -- `n_train_steps` 5,038 / 4,997 / 4,423, NOT
the ~8,000 an earlier draft of this docstring guessed -- that is roughly 2,900 parameters per
training row at the top rung, over ~1,200 Adam steps. (The mlp128 count of 21,381 is exactly the
action-path parameter count V3-EXQ-1008 reports for every one of its 32-dim arms -- the
consumer's reader is reproduced by construction, not approximated.) Guard 2 additionally requires
the ladder to demonstrably MEMORISE the training split, which is what licenses reading a flat
HELD-OUT profile as a statement about content rather than about fit power. It is a bound, not a
proof, and this docstring says so rather than letting the queue title imply otherwise.

SECOND LIMIT, and the reason for the sample-saturation witness. Guard 2 excludes an UNDER-POWERED
reader; it does NOT exclude an UNDER-SAMPLED map. "Memorises 5,000 rows and does not generalise"
is also what a genuinely present but complex map looks like at this sample size. No arm varies n,
so the OFF track's `mlp512` rung is additionally refit on SATURATION_FRACTION (0.5) of the
training rows and the held-out delta recorded (`sample_saturation_witness`). A delta far beyond
the ~0.013 seed spread 1008 measured means the learning curve is still climbing and a CONFIRMED
verdict is partly confounded with training-set size. RECORDED, not a gate: it bounds the reading
rather than blocking it, because the autopsy's routing asked for a capacity sweep and this is a
different axis.

THIRD LIMIT. The run cannot separate "content absent" from "content present in a form no
feed-forward decoder of this family can express". The reader-free witness narrows that (it uses
the oracle's own rule rather than a learned one) but does not close it. The autopsy's SECONDARY
routing -- the layer-wise content probe on the observation->z_world path -- is the follow-on that
would, and it is explicitly gated on this run returning CONFIRMED.

=== REPAIR RECORD (red-team, Step 4.5) ===

red-team (fable): CONTESTED, no BLOCKING. One pass, not re-spawned. EIGHT findings, every one
verified against the source before acting; six APPLIED, two DISCLOSED.

  F1 (high) APPLIED. The frozen-latent REPRODUCTION BAND routed nowhere. `gate_green` read
     `per_arm_gate.non_degenerate`, which is ANY-arm-green by design (so that one red arm cannot
     vacate a good one) -- so the instrument-gate arm alone going green satisfied it while the
     band failed on `zworld_off__mlp128`, and a content verdict still fired. FIX: an explicit
     `verdict_ready` = every VERDICT ARM green by name (all five OFF rungs + the anchor's
     consumer and max rungs), checked in `_adjudicate` BEFORE both guards, with its own
     `C_verdict_arms_ready` load-bearing criterion and two self-test rows.
  F1b (high) APPLIED, found in the same finding. The first draft declared only its own five
     preconditions and had silently DROPPED x1002's -- `zworld_encoder_trained_in_p0`,
     `zworld_not_collapsed`, `oracle_labels_non_degenerate`, `heldout_split_sufficient`,
     `d3_local_view_greedy_clears_floor`. FIX: `PRECONDITION_SPECS = list(x1002.PRECONDITION_SPECS)
     + NEW_PRECONDITION_SPECS`, exactly as 1008 does, with `measured` now computed PER ARM (so a
     track's encoder health is certified by ITS OWN cells) and the two ctx keys those specs
     condition on (`has_encoder`, `trained_encoder`) added to `_arm_ctx`.
  F2 (high) APPLIED, then SETTLED BY MEASUREMENT. "Fixed Adam lr across 77,000x confounds
     capacity with trainability; the anchor cannot calibrate it because the instability is
     representation-dependent." Verified the evidence (the smoke's diverging deep rung) and the
     asymmetry it cites (1008: OFF train 0.743-0.766 vs PCA 0.961-0.968 at the same width) --
     both real. FIX: GRAD_CLIP_NORM applied IDENTICALLY at every rung of every track (so the
     manipulation stays capacity alone), a per-rung `diverged` flag, and Guard 2 as a max over
     rungs. Then the full-scale probe above showed the divergence was a DRY-SCALE ARTIFACT: at
     real scale every rung fits and the over-capacity rungs memorise. The reviewer's proposed
     per-rung learning rate was therefore NOT adopted -- it would vary two things at once to fix
     a problem that does not exist at the scale this runs at.
  F3 (medium) APPLIED. CONFIRMED was inseparable from "map present but under-sampled", and the
     docstring's "~7,900 training rows" was wrong. FIX: the row count corrected to 1008's own
     measured 5038/4997/4423 (independently reproduced by the probe at 5038), and a
     `sample_saturation_witness` added -- the OFF `mlp512` rung refit on 50% of the training
     rows with the held-out delta recorded. RECORDED, not a gate: it bounds the reading of
     CONFIRMED rather than blocking it, since sample size is a different axis from the one the
     autopsy's routing named.
  F4 (medium) APPLIED. No untrained-latent track, so ELIMINATED could not be attributed to the
     ENCODE rather than to the 250->32 BOTTLENECK -- and 1008 measured the untrained latent
     ABOVE the trained one on every seed (0.699/0.681/0.704 vs 0.672/0.674/0.661), so this is a
     live risk, not a theoretical one. FIX: a full third `zworld_untrained` track at all five
     rungs, which costs NO warmup, plus `off_minus_untrained_best` per seed and an explicit
     line in the ELIMINATED branch of the null table.
  F5 (low-medium) APPLIED. The docstring described Guard 1 in the DEGENERATE form ("its own best
     rung ... below its mlp128 value") that the code had already rejected. Both the docstring
     and the precondition text now state the max-capacity-minus-consumer statistic and say why
     the obvious spelling is >= 0 by construction.
  F6 (low) APPLIED. `x1002._worst_cell` returns `(0.0, None)` on an empty list, which would fail
     Guard 1 OPEN at 0.0; and the anchor rows carried no `cell_id`, so the offending seed always
     reported null. FIX: `cell_id` supplied on the anchor-delta rows (confirmed in the smoke
     manifest). The fail-open path is DISCLOSED rather than re-plumbed: it is unreachable here
     because the instrument gate and `verdict_ready` both precede Guard 1 and both require the
     very arms whose absence would empty that list.
  F7 (low) APPLIED. The reader-free witness had no baseline and no collapse guard -- and the
     smoke showed the DIVERGED rung producing the HIGHEST witness agreement (0.60) with
     decision-coordinate r2 of -50.9, i.e. a near-constant decode laundering the label marginal.
     FIX: `nonlinear_decode_margin_over_trivial` and a `nonlinear_decode_collapsed` flag
     (decision-coordinate r2 < 0), plus a run-level `collapsed_witness_rungs` set.
  F8 (low) APPLIED. Criteria whose statistic is a MAX OVER RUNGS were owned by the top rung, so
     a diverged top rung would report `non_degenerate: False` for a verdict that came cleanly
     off another rung. FIX: those criteria are now owned by the CONSUMER rung.

Additionally, `validate_experiments.py` caught a false-cache-HIT hazard the red-team did not:
GRAD_CLIP_NORM and SATURATION_FRACTION are readout-affecting constants that were absent from the
declared `config_slice` of a cross-driver-reusable fingerprint. Both are now declared.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from experiment_protocol import emit_outcome  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402
from experiments._metrics import (  # noqa: E402
    P0NotReady,
    dv_headroom_check,
    p0_readiness_gate,
)
from experiments._lib.arm_fingerprint import arm_cell, reset_all_rng  # noqa: E402
from experiments._lib.capability_eval import (  # noqa: E402
    LocalViewGreedyPolicy,
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
# IMPORTED, NEVER REDEFINED: the dataset recipe, the standardiser, the scoring, the 978-warmup
# reproduction, the PCA projection and the closed-form oracle rule are 1002's and 1008's own
# functions, so this run's anchors are reproductions of theirs and the capacity axis is read
# with their instrument.
import experiments.v3_exq_1002_zworld_actor_adequacy_oracle_adapter as x1002  # noqa: E402
import experiments.v3_exq_1008_zworld_adequacy_portfolio_ws250_rebasis as x1008  # noqa: E402
import experiments.v3_exq_724_competence_localization_diagnostic as x724  # noqa: E402
import experiments.v3_exq_734_env_difficulty_competence_recovery_sweep as x734  # noqa: E402
import experiments.v3_exq_737_ree_latent_policy_head_competence_probe as x737  # noqa: E402
import experiments.v3_exq_808_return_decomposition_objective_misspecification as x808  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_1010_zworld_overcapacity_decoder_sweep"
QUEUE_ID = "V3-EXQ-1010"
EXPERIMENT_PURPOSE = "diagnostic"
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"
CLAIM_IDS: List[str] = []          # deliberately empty -- see the docstring's Claims line
BEARS_ON = ["INV-088", "MECH-457"]

HYPOTHESIS_QID = x1008.HYPOTHESIS_QID
HYPOTHESIS_LEG = "H-F-content-discarded-at-encode"
DEVICE = x1002.DEVICE

# --------------------------------------------------------------------------------------
# SCHEDULE / INSTRUMENT CONSTANTS -- every one imported, none re-derived
# --------------------------------------------------------------------------------------
ZWORLD_P0_EPISODES = x1002.ZWORLD_P0_EPISODES      # 60
P0_WARMUP_EPISODES = x1002.P0_WARMUP_EPISODES      # 200
P1_REINFORCE_EPISODES = x1002.P1_REINFORCE_EPISODES  # 90
EVAL_EPISODES = x1002.EVAL_EPISODES                # 20
STEPS_PER_EPISODE = x1002.STEPS_PER_EPISODE        # 200
RUNG = x1002.RUNG
RUNG_ID = x1002.RUNG_ID
LEVEL_ID = x1002.LEVEL_ID
RESOURCE_FIELD_DIM = x1002.RESOURCE_FIELD_DIM      # 25
BC_EPISODES = x1002.BC_EPISODES                    # 40
BC_RANDOM_EPISODES = x1002.BC_RANDOM_EPISODES      # 20
ADAPTER_PASSES = x1002.ADAPTER_PASSES              # 60
ADAPTER_LR = x1002.ADAPTER_LR                      # 1e-3
ADAPTER_BATCH = x1002.ADAPTER_BATCH                # 256
SEED_MAJORITY = x1002.SEED_MAJORITY                # 2 of 3

AGREEMENT_BAR = x1002.AGREEMENT_BAR                        # 0.80
AGREEMENT_ELEVATION_MIN = x1002.AGREEMENT_ELEVATION_MIN    # 0.20
RAW_FIELD_CONTROL_FLOOR = x1002.RAW_FIELD_CONTROL_FLOOR    # 0.60

PROJECTION_DIM = x1008.PROJECTION_DIM              # 32
DECISION_WORLD_STATE_INDICES = x1008.DECISION_WORLD_STATE_INDICES
RESOURCE_FIELD_OFFSET = x1008.RESOURCE_FIELD_OFFSET

# ---- THIS RUN'S OWN PRE-REGISTERED CONSTANTS -----------------------------------------
# Guard 1: how far the CALIBRATION ANCHOR may fall under added capacity before the fit protocol
# is judged unable to support these capacities at all. 0.10 is one pre-registered
# AGREEMENT_ELEVATION_MIN half-step, and comfortably larger than the 0.02 seed spread 1008
# measured on the PCA arm (0.8578-0.8771) -- so it fires on protocol collapse, not on noise.
ANCHOR_DEGRADE_TOL = 0.10
# Guard 2: the train-split agreement the OVER-PARAMETERISED rung must reach on the frozen latent
# for the ladder to have demonstrably reached over-capacity. 1002 measured 0.828 train agreement
# for the UNTRAINED latent at the 128-wide rung, so 0.95 is a real ask.
MEMORISE_FLOOR = 0.95
# The band the mlp128 OFF rung must land in to certify that the 978 warmup reproduced the same
# regime 1002/1008 read (they measured 0.6606-0.6735). Wide enough to absorb an adapter-init
# re-draw (1002 vs 1008 differed by ~0.005), narrow enough to catch a regime drift.
OFF_REPRO_LOW = 0.60
OFF_REPRO_HIGH = 0.75

# --------------------------------------------------------------------------------------
# THE CAPACITY LADDER
# --------------------------------------------------------------------------------------
# (rung_id, kind, hidden, depth). depth counts TANH HIDDEN LAYERS.
# `mlp` rungs instantiate x734.PPOPolicyNet ITSELF at depth 2, so the 128 rung is the consumer's
# reader by construction rather than a transcribed look-alike.
CAPACITY_LADDER: List[Tuple[str, str, Optional[int], int]] = [
    ("linear", "linear", None, 0),
    ("mlp128", "mlp", 128, 2),
    ("mlp512", "mlp", 512, 2),
    ("mlp2048", "mlp", 2048, 2),
    ("deep2048x4", "deep", 2048, 4),
]
RUNG_IDS = [r[0] for r in CAPACITY_LADDER]
CONSUMER_RUNG = "mlp128"        # the reproduction anchor: x734.PPO_TRUNK_HIDDEN
MAX_CAPACITY_RUNG = "deep2048x4"   # the rung Guard 2 is measured on

TRACK_OFF = "zworld_off"
TRACK_PCA = "ws250_pca"
TRACK_UNT = "zworld_untrained"
SWEPT_TRACKS = [TRACK_OFF, TRACK_PCA, TRACK_UNT]
ARM_RAW = x1002.ARM_RAW          # "rawfield_ceiling" -- instrument gate, single rung, no sweep


def _arm_id(track: str, rung: str) -> str:
    return "%s__%s" % (track, rung)


SWEPT_ARM_IDS = [_arm_id(t, r) for t in SWEPT_TRACKS for r in RUNG_IDS]
ARM_IDS = [ARM_RAW] + SWEPT_ARM_IDS      # 1 + 15 = 16 arms per seed
OFF_ARM_IDS = [_arm_id(TRACK_OFF, r) for r in RUNG_IDS]
PCA_ARM_IDS = [_arm_id(TRACK_PCA, r) for r in RUNG_IDS]
UNT_ARM_IDS = [_arm_id(TRACK_UNT, r) for r in RUNG_IDS]
Z_ARM_IDS = OFF_ARM_IDS + UNT_ARM_IDS          # every arm reading a z_world latent
TRAINED_Z_ARM_IDS = list(OFF_ARM_IDS)          # ... of which these read a TRAINED one

# Gradient-norm clipping, applied IDENTICALLY at every rung of every track. It is part of the
# ONE fit protocol, not a per-rung tuning knob, so "only capacity varies" survives intact. 1.0
# is the same norm `experiments/_lib/allon_training.py` already clips every other head in this
# codebase to. Added after the red-team raised "fixed lr across a 77,000x parameter range
# confounds capacity with trainability" -- see THE TOP-RUNG TRAINABILITY MEASUREMENT in the
# docstring, which settles it at full scale rather than by argument.
GRAD_CLIP_NORM = 1.0
# A rung whose final CE exceeds the uniform-logit value has not fitted at all; recorded per rung
# as a divergence flag so a later reader sees it rather than having it absorbed by a max.
import math as _math  # noqa: E402
UNIFORM_LOGIT_CE = float(_math.log(5.0))   # action_dim is 5 on this rung; re-derived in-run
# Fraction of TRAINING EPISODES the sample-saturation witness re-fits on, to separate "content
# absent" from "map present but this training-set size is too small" (red-team finding 3).
SATURATION_FRACTION = 0.5
SATURATION_RUNG = "mlp512"

# ---- dry-run scale (imported where 1002/1008 defined it) -----------------------------
DRY_RUN_SEEDS = [42]
DRY_RUN_ZWORLD_P0 = x1002.DRY_RUN_ZWORLD_P0
DRY_RUN_P0 = x1002.DRY_RUN_P0
DRY_RUN_P1 = x1002.DRY_RUN_P1
DRY_RUN_EVAL = x1002.DRY_RUN_EVAL
DRY_RUN_STEPS = x1002.DRY_RUN_STEPS
DRY_RUN_BC_EPISODES = x1002.DRY_RUN_BC_EPISODES
DRY_RUN_BC_RANDOM_EPISODES = x1002.DRY_RUN_BC_RANDOM_EPISODES
DRY_RUN_ADAPTER_PASSES = x1002.DRY_RUN_ADAPTER_PASSES
# The over-parameterised rungs are the point of the run, so the smoke exercises them at their
# real widths -- a dry run that shrank them would not exercise the code under test. Only the
# episode counts and pass count shrink.

_EXTRA_SUBSTRATE_PATHS = [Path(m.__file__) for m in (x1002, x1008, x724, x734, x737, x808)]
_ZG = ZGoalStreamAccumulator()

SEEDS = [42, 43, 44]


# --------------------------------------------------------------------------------------
# ARM CONTEXTS + PRECONDITION SPECS
# --------------------------------------------------------------------------------------
def _arm_ctx(aid: str) -> Dict[str, Any]:
    track = (ARM_RAW if aid == ARM_RAW else aid.split("__", 1)[0])
    rung = (None if aid == ARM_RAW else aid.split("__", 1)[1])
    return {
        # `id` is the key `assert_no_structurally_unsatisfiable_gate` defaults to, and the key
        # x1002's own imported specs were written against; `arm_id` is kept as the readable
        # alias this driver uses internally.
        "id": aid,
        "arm_id": aid,
        "track": track,
        "rung": rung,
        "is_swept": bool(aid in SWEPT_ARM_IDS),
        "is_off_track": bool(aid in OFF_ARM_IDS),
        "is_anchor_track": bool(aid in PCA_ARM_IDS),
        "is_untrained_track": bool(aid in UNT_ARM_IDS),
        "is_max_capacity": bool(rung == MAX_CAPACITY_RUNG),
        "is_consumer_rung": bool(rung == CONSUMER_RUNG),
        # The two keys x1002's imported PRECONDITION_SPECS condition on.
        "has_encoder": bool(aid in Z_ARM_IDS),
        "trained_encoder": bool(aid in TRAINED_Z_ARM_IDS),
    }


def _arm_contexts() -> List[Dict[str, Any]]:
    return [_arm_ctx(a) for a in ARM_IDS]


NEW_PRECONDITION_SPECS = [
    # --- Guard 1: the anchor certifies the PROTOCOL (anchor track only) -------------------
    PreconditionSpec(
        name="overcapacity_fit_protocol_sound",
        description=("GUARD 1. The CALIBRATION ANCHOR (ws250_pca, a 32-dim representation known "
                     "to carry the content: 1008 measured 0.8684) must not lose more than "
                     "ANCHOR_DEGRADE_TOL of held-out agreement AT THE OVER-PARAMETERISED RUNG "
                     "relative to its own consumer-width rung. A collapse here means this fit "
                     "protocol cannot support over-capacity decoders on this training-set size, "
                     "so the OFF track's flatness would be an instrument artifact rather than a "
                     "content finding. Measured as anchor(deep2048x4) - anchor(mlp128), worst "
                     "seed; a FLOOR at -ANCHOR_DEGRADE_TOL. NOTE, because the obvious spelling "
                     "is the wrong one: the statistic is the MAX-CAPACITY rung minus the "
                     "consumer rung, NOT the BEST rung minus the consumer rung -- best is a max "
                     "OVER a set containing the consumer rung, so that difference is >= 0 by "
                     "construction and the gate could never fail."),
        threshold=float(-ANCHOR_DEGRADE_TOL),
        control="ws250_pca track -- a DIFFERENT representation at the SAME width under the SAME "
                "protocol; excluded from the verdict statistic entirely",
        applies_to=(lambda ctx: bool(ctx.get("is_anchor_track"))),
    ),
    # --- Guard 2: the decoder demonstrably HAS the fit power (OFF track, top rung only) ---
    PreconditionSpec(
        name="overcapacity_decoder_can_memorise",
        description=("GUARD 2. Some rung reading the FROZEN LATENT must reach MEMORISE_FLOOR "
                     "TRAIN-SPLIT agreement. This is what licenses reading a flat HELD-OUT "
                     "profile as a statement about generalisable content rather than about fit "
                     "power: a decoder that can memorise the labels and still not generalise "
                     "has not failed for want of capacity. 1002 measured 0.828 train agreement "
                     "for the UNTRAINED latent at the 128-wide rung, so this is a real ask. "
                     "STATISTIC: max over rungs of the OFF track's TRAIN agreement, then the "
                     "WORST SEED of that. Deliberately a MAX OVER RUNGS rather than the top "
                     "rung's own value -- the claim is 'the LADDER demonstrably reached "
                     "over-capacity', which one rung memorising establishes. The dry-run smoke "
                     "showed the deep2048x4 rung's loss and its field-decode MSE both DIVERGING "
                     "at 3 passes (OFF ce 4.57 and witness MSE 3.93, against 0.66 and 0.07 at "
                     "mlp2048), so a deep tanh stack at ADAPTER_LR is genuinely at risk of "
                     "instability; keying this guard to that single rung would let an optimiser "
                     "artifact discard an otherwise sound run. Per-rung train agreement and "
                     "final CE are recorded either way, so divergence stays auditable."),
        threshold=float(MEMORISE_FLOOR),
        control="zworld_off track, max over capacity rungs of TRAIN-split agreement, worst seed",
        applies_to=(lambda ctx: bool(ctx.get("is_off_track") and ctx.get("is_max_capacity"))),
    ),
    # --- regime reproduction (OFF track, consumer rung only) ------------------------------
    # A TWO-SIDED band, declared as TWO specs because PreconditionSpec carries a single
    # threshold + direction. Both legs are therefore independently recomputable by the indexer
    # from their own reported numbers -- which is the property the interval rule is protecting;
    # a single-bound declaration would leave the other leg absent from the manifest and let a
    # violation of it recompute as MET (V3-EXQ-779b `baseline_entropy_headroom`).
    PreconditionSpec(
        name="frozen_latent_regime_reproduced_floor",
        description=("The 978-OFF warmup reproduced the regime 1002 and 1008 read: the OFF "
                     "track at the consumer's exact width is NOT BELOW the band those runs "
                     "measured (0.6606-0.6735). Below it, the warmup or the substrate has "
                     "drifted and the capacity sweep is not being run on the latent the "
                     "question is about. LOWER leg of a two-sided band."),
        threshold=float(OFF_REPRO_LOW),
        direction="lower",
        control="zworld_off__mlp128 held-out agreement, worst (minimum) seed",
        applies_to=(lambda ctx: bool(ctx.get("is_off_track") and ctx.get("is_consumer_rung"))),
    ),
    PreconditionSpec(
        name="frozen_latent_regime_reproduced_ceiling",
        description=("UPPER leg of the same two-sided band: the OFF track at the consumer's "
                     "exact width is NOT ABOVE 0.75. Above it, the frozen latent is carrying "
                     "materially more of the mapping than 1002 and 1008 measured, which would "
                     "mean the regime moved and the H-F question is being asked of a different "
                     "object. Measured on the BEST (maximum) seed, since this leg's `met` is "
                     "the opposite worst case from the floor's."),
        threshold=float(OFF_REPRO_HIGH),
        direction="upper",
        control="zworld_off__mlp128 held-out agreement, worst (maximum) seed",
        applies_to=(lambda ctx: bool(ctx.get("is_off_track") and ctx.get("is_consumer_rung"))),
    ),
]

# x1002's OWN readiness specs are INHERITED, not re-derived -- they are what certify that the
# encoder actually trained (`zworld_encoder_trained_in_p0`), that the latent did not collapse
# (`zworld_not_collapsed`), that the oracle's labels are not degenerate, that the held-out split
# is large enough, and that the oracle itself clears the competence floor. V3-EXQ-1008 inherits
# them the same way. Dropping them (as this driver's first draft did) would have left the
# "same object 1002 and 1008 read" claim resting on nothing but the reproduction band.
PRECONDITION_SPECS = list(x1002.PRECONDITION_SPECS) + NEW_PRECONDITION_SPECS


# --------------------------------------------------------------------------------------
# THE DECODERS
# --------------------------------------------------------------------------------------
class _LinearReadout(nn.Module):
    """A fitted LINEAR softmax readout. Same (logits, value) contract as x734.PPOPolicyNet, so
    every x1002 scoring helper reads it unchanged. `trunk` is Identity: this rung's whole point
    is that there is no hidden layer."""

    def __init__(self, in_dim: int, action_dim: int) -> None:
        super().__init__()
        self.trunk = nn.Identity()
        self.policy_head = nn.Linear(in_dim, action_dim)
        self.value_head = nn.Linear(in_dim, 1)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(state)
        return self.policy_head(h), self.value_head(h).squeeze(-1)


class _DeepReadout(nn.Module):
    """`depth` tanh hidden layers at `hidden`. Identical in shape to x734.PPOPolicyNet at
    depth=2, generalised upward; used only for the over-parameterised rung."""

    def __init__(self, in_dim: int, action_dim: int, hidden: int, depth: int) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        d = int(in_dim)
        for _ in range(int(depth)):
            layers.append(nn.Linear(d, int(hidden)))
            layers.append(nn.Tanh())
            d = int(hidden)
        self.trunk = nn.Sequential(*layers)
        self.policy_head = nn.Linear(d, action_dim)
        self.value_head = nn.Linear(d, 1)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(state)
        return self.policy_head(h), self.value_head(h).squeeze(-1)


def _rung_spec(rung: str) -> Tuple[str, Optional[int], int]:
    for rid, kind, hidden, depth in CAPACITY_LADDER:
        if rid == rung:
            return kind, hidden, depth
    raise KeyError("unknown capacity rung: %s" % rung)


def _make_decoder(rung: str, in_dim: int, action_dim: int, out_dim: Optional[int] = None):
    """The capacity ladder's only construction site.

    `out_dim` is None for an action decoder (uses action_dim); the field-decode witness passes
    RESOURCE_FIELD_DIM so the witness runs at the SAME capacity as the action decoder it sits
    beside, which is what makes the two readouts comparable rung by rung.
    """
    kind, hidden, depth = _rung_spec(rung)
    od = int(action_dim if out_dim is None else out_dim)
    if kind == "linear":
        return _LinearReadout(int(in_dim), od)
    if kind == "mlp":
        # x734.PPOPolicyNet ITSELF -- at hidden=128 this is bit-identically the consumer's
        # reader and the class V3-EXQ-978 instantiated, not a look-alike.
        return x734.PPOPolicyNet(int(in_dim), od, hidden=int(hidden))
    if kind == "deep":
        return _DeepReadout(int(in_dim), od, int(hidden), int(depth))
    raise KeyError("unknown rung kind: %s" % kind)


def _capacity_report(net, rung: str, in_dim: int, action_dim: int) -> Dict[str, Any]:
    """Measure the capacity into the manifest rather than asserting it in prose.

    Deliberately NOT x1002._capacity_report: that helper hardcodes `trunk_hidden` to
    x734.PPO_TRUNK_HIDDEN, which is true only at this ladder's consumer rung and would silently
    mislabel every other rung.
    """
    kind, hidden, depth = _rung_spec(rung)
    trunk = int(sum(p.numel() for p in net.trunk.parameters()))
    phead = int(sum(p.numel() for p in net.policy_head.parameters()))
    vhead = int(sum(p.numel() for p in net.value_head.parameters()))
    return {
        "module_class": type(net).__name__,
        "rung": rung,
        "rung_kind": kind,
        "trunk_hidden": (int(hidden) if hidden is not None else 0),
        "trunk_depth": int(depth),
        "in_dim": int(in_dim),
        "action_dim": int(action_dim),
        "trunk_params": trunk,
        "policy_head_params": phead,
        "value_head_params_unused": vhead,
        "action_path_params": trunk + phead,
        "total_params": trunk + phead + vhead,
        "is_consumer_capacity": bool(rung == CONSUMER_RUNG),
        "consumer_trunk_hidden": int(x734.PPO_TRUNK_HIDDEN),
    }


def _train_decoder(net, x_tr: torch.Tensor, y_tr: torch.Tensor, passes: int,
                   seed: int, arm_id: str, action_dim: int = 5) -> Dict[str, Any]:
    """Behaviour-clone the oracle at the given capacity.

    x1002._train_adapter's loop (same optimiser, lr, batch size, pass count, shuffle and print
    cadence) with the network passed in rather than constructed inside, which is the single
    change the capacity sweep requires. Re-implemented rather than imported because
    x1002._train_adapter builds the net itself at the consumer's fixed width -- the very thing
    this run manipulates.

    ONE DELIBERATE PROTOCOL ADDITION, applied IDENTICALLY AT EVERY RUNG OF EVERY TRACK:
    gradient-norm clipping at GRAD_CLIP_NORM. Because it is identical everywhere, the sweep's
    "only capacity varies" claim is untouched -- what changes is that the deepest rung can
    actually fit instead of diverging. See GRAD_CLIP_NORM's own comment for the smoke
    measurement that motivated it.
    """
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
            torch.nn.utils.clip_grad_norm_(net.parameters(), GRAD_CLIP_NORM)
            opt.step()
            total += float(loss.item())
            nb += 1
        losses.append(total / max(nb, 1))
        if (p + 1) % 10 == 0 or (p + 1) == int(passes):
            # Deliberately NOT the "ep N/M" shape -- that pattern is the runner's episode
            # progress channel and its denominator belongs to _train_all_on_agent.
            print("  [decoder] %s seed=%d pass %d of %d ce_loss=%.4f"
                  % (arm_id, seed, p + 1, int(passes), losses[-1]), flush=True)
    final = (losses[-1] if losses else None)
    # A rung whose final CE is at or above the uniform-logit value has not fitted AT ALL, so its
    # readouts are an optimiser artifact rather than a capacity measurement. Recorded per rung
    # rather than silently absorbed by the max-over-rungs statistics.
    uniform_ce = float(np.log(max(int(action_dim), 2))) if action_dim else UNIFORM_LOGIT_CE
    return {"final_ce_loss": final,
            "first_ce_loss": (losses[0] if losses else None),
            "min_ce_loss": (min(losses) if losses else None),
            "ce_loss_by_pass_tail": [float(v) for v in losses[-5:]],
            "uniform_logit_ce": uniform_ce,
            "diverged": (bool(final >= uniform_ce) if final is not None else None),
            "grad_clip_norm": float(GRAD_CLIP_NORM),
            "n_passes": int(passes), "n_train_steps": n}


def _train_field_decoder(net, x_tr: torch.Tensor, t_tr: torch.Tensor, passes: int,
                         seed: int, arm_id: str) -> Dict[str, Any]:
    """The READER-FREE content witness's decoder: latent -> 25-dim resource field, MSE, at the
    SAME capacity as this rung's action decoder. No action head anywhere in its path."""
    opt = torch.optim.Adam(net.parameters(), lr=ADAPTER_LR)
    n = int(x_tr.shape[0])
    losses: List[float] = []
    for p in range(int(passes)):
        perm = torch.randperm(n)
        total, nb = 0.0, 0
        for i in range(0, n, ADAPTER_BATCH):
            idx = perm[i:i + ADAPTER_BATCH]
            pred, _v = net(x_tr[idx])
            loss = F.mse_loss(pred, t_tr[idx])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), GRAD_CLIP_NORM)
            opt.step()
            total += float(loss.item())
            nb += 1
        losses.append(total / max(nb, 1))
    if losses:
        print("  [witness] %s seed=%d field-decode mse %.5f -> %.5f over %d passes"
              % (arm_id, seed, losses[0], losses[-1], int(passes)), flush=True)
    return {"final_mse": (losses[-1] if losses else None),
            "first_mse": (losses[0] if losses else None), "n_passes": int(passes)}


def _agreement(net, x: torch.Tensor, y: torch.Tensor) -> Optional[float]:
    if int(x.shape[0]) == 0:
        return None
    with torch.no_grad():
        logits, _v = net(x)
        pred = torch.argmax(logits, dim=-1)
    return float((pred == y).float().mean().item())


def _field_targets(episodes: List[Dict[str, Any]]) -> torch.Tensor:
    """The 25-dim resource field, row-aligned with x1002's feature extractors."""
    x, _y = x1002._rawfield_features(episodes)
    return x


def _r2_per_coord(pred: torch.Tensor, target: torch.Tensor,
                  cols: List[int]) -> Optional[float]:
    """Mean held-out R2 over the named columns. `cols` in FIELD-LOCAL indices."""
    if int(pred.shape[0]) == 0 or not cols:
        return None
    p = pred[:, cols]
    t = target[:, cols]
    ss_res = float(((p - t) ** 2).sum().item())
    ss_tot = float(((t - t.mean(dim=0, keepdim=True)) ** 2).sum().item())
    if ss_tot <= 0.0:
        return None
    return float(1.0 - ss_res / ss_tot)


# The oracle's five decision cells, expressed in FIELD-LOCAL coordinates (the 25-dim resource
# field block), which is what the witness decodes.
DECISION_FIELD_INDICES = sorted(int(j) - RESOURCE_FIELD_OFFSET
                                for j in DECISION_WORLD_STATE_INDICES)


def _content_witness(net, x_te: torch.Tensor, f_te: torch.Tensor, y_te: torch.Tensor,
                     trivial_baseline: Optional[float]) -> Dict[str, Any]:
    """Decode the field at this rung's capacity, then apply THE ORACLE'S OWN ARGMAX RULE to the
    decoded field -- no fitted action head in the path. The nonlinear analogue of 1008's
    closed-form linear content witness (which degraded to 0.5931/0.5910/0.6066 against a
    0.5661/0.5803/0.5720 trivial baseline).

    THE RAW AGREEMENT IS NOT READABLE ON ITS OWN, and the authoring smoke proved it: the
    DIVERGED deep2048x4 rung produced the HIGHEST witness agreement of any rung (0.60) while its
    decision-coordinate r2 was -50.9 -- a near-constant decoded field whose argmax simply
    reproduces the label marginal. So this returns the baseline-relative margin alongside the
    raw number, and flags a negative decision-coordinate r2, which is what separates "the field
    was decoded and the oracle's rule on it recovers the action" from "the decode collapsed and
    the rule is reading a constant".
    """
    if int(x_te.shape[0]) == 0:
        return {"nonlinear_decode_oracle_agreement": None,
                "nonlinear_decode_margin_over_trivial": None,
                "nonlinear_decode_field_r2": None,
                "nonlinear_decode_decision_coord_r2": None,
                "nonlinear_decode_collapsed": None,
                "nonlinear_decoded_flat_window_frac": None}
    with torch.no_grad():
        pred, _v = net(x_te)
    rule = x1008._decode_oracle_agreement(pred, y_te)
    all_cols = list(range(int(f_te.shape[1])))
    agree = rule.get("linear_decode_oracle_agreement")
    dec_r2 = _r2_per_coord(pred, f_te, DECISION_FIELD_INDICES)
    return {
        "nonlinear_decode_oracle_agreement": agree,
        "nonlinear_decode_margin_over_trivial": (
            (float(agree) - float(trivial_baseline))
            if (agree is not None and trivial_baseline is not None) else None),
        "nonlinear_decode_field_r2": _r2_per_coord(pred, f_te, all_cols),
        "nonlinear_decode_decision_coord_r2": dec_r2,
        # A negative r2 on the five coordinates the oracle actually reads means the decode is
        # worse than predicting their mean: the witness number is an artifact, not content.
        "nonlinear_decode_collapsed": (bool(dec_r2 < 0.0) if dec_r2 is not None else None),
        "nonlinear_decoded_flat_window_frac": rule.get("decoded_flat_window_frac"),
    }


# --------------------------------------------------------------------------------------
# CONFIG SLICE
# --------------------------------------------------------------------------------------
def _config_slice(base: Dict[str, Any], arm_id: str) -> Dict[str, Any]:
    ctx = _arm_ctx(arm_id)
    kind, hidden, depth = ((None, None, None) if arm_id == ARM_RAW
                           else _rung_spec(str(ctx["rung"])))
    d = dict(base)
    d["arm_id"] = arm_id
    d["arm_track"] = ctx["track"]
    d["arm_capacity_rung"] = ctx["rung"] or CONSUMER_RUNG
    d["arm_decoder_kind"] = kind or "mlp"
    d["arm_decoder_hidden"] = (int(hidden) if hidden else (int(x734.PPO_TRUNK_HIDDEN)
                                                           if arm_id == ARM_RAW else 0))
    d["arm_decoder_depth"] = (int(depth) if depth is not None else 2)
    d["arm_input"] = ("resource_field_view" if arm_id == ARM_RAW
                      else "z_world" if ctx["is_off_track"] else "world_state")
    d["arm_projection"] = ("pca" if ctx["is_anchor_track"] else "none")
    d["arm_projection_dim"] = (int(PROJECTION_DIM) if ctx["is_anchor_track"] else None)
    d["arm_rebasis"] = "diag_zscore"
    d["decoder_init_reseeded_per_fit"] = True
    if ctx["is_off_track"]:
        d["arm_p0a_field_weight"] = 0.0    # 978's OFF arm
        d["arm_warmup_skipped"] = False
    # The two readout-affecting constants MUST be in the declared slice: an omitted one is a
    # false-cache-HIT (a consumer with a different value would hit these cells and read numbers
    # computed under a different scheme), which corrupts a conclusion rather than merely wasting
    # compute. Declared as `dict(...)` KEYWORDS, not `d[...] =` subscripts, on purpose: the
    # config_slice-declaration audit resolves a slice by absorbing the returned expression, and
    # it can follow a `dict()` passthrough call's keyword NAMES but not subscript assignments
    # into a local. Spelling it this way is what makes the declaration visible to the audit that
    # exists to catch exactly this mistake.
    return dict(d,
                grad_clip_norm=GRAD_CLIP_NORM,
                saturation_fraction=SATURATION_FRACTION,
                saturation_rung=SATURATION_RUNG)


# --------------------------------------------------------------------------------------
# CELLS
# --------------------------------------------------------------------------------------
def _fit_track_cell(arm_id: str, seed: int, data: Dict[str, Any], action_dim: int, passes: int,
                    x_tr: torch.Tensor, y_tr: torch.Tensor, x_te: torch.Tensor,
                    y_te: torch.Tensor, xr_te: torch.Tensor, yr_te: torch.Tensor,
                    transform: Any, f_tr: Optional[torch.Tensor],
                    f_te: Optional[torch.Tensor]) -> Dict[str, Any]:
    """Transform -> (re-seeded) decoder fit at this rung -> every agreement readout, plus the
    reader-free content witness where a field target is available."""
    ctx = _arm_ctx(arm_id)
    rung = str(ctx["rung"] or CONSUMER_RUNG)
    xs_tr, xs_te, xsr_te = transform(x_tr), transform(x_te), transform(xr_te)
    in_dim = int(xs_tr.shape[1]) if int(xs_tr.shape[0]) else 0

    # Same decoder init draw for every ARM at a given seed and rung (1008 red-team F8): the
    # paired differentials then carry no init-draw component.
    reset_all_rng(seed)
    net = _make_decoder(rung, in_dim, action_dim)
    train_stats = _train_decoder(net, xs_tr, y_tr, passes, seed, arm_id, action_dim)

    row: Dict[str, Any] = {
        "cell_id": "%s|seed%d" % (arm_id, seed),
        "arm_id": arm_id,
        "track": ctx["track"],
        "capacity_rung": rung,
        "seed": int(seed),
        "feature_dim": in_dim or None,
        "input_dim": (int(x_tr.shape[1]) if int(x_tr.shape[0]) else None),
        "capacity_report": _capacity_report(net, rung, in_dim, action_dim),
        "decoder_training": train_stats,
        "feature_transform": transform.report(),
    }
    row.update(x1002._score_cell(net, xs_tr, y_tr, xs_te, y_te, xsr_te, yr_te, action_dim,
                                 prev_te=x1002._prev_action_vector(data["test"])))

    # ---- the reader-free content witness, at the SAME capacity -------------------------
    if f_tr is not None and f_te is not None and int(xs_tr.shape[0]) > 0:
        reset_all_rng(seed + 10_000)     # a distinct, deterministic draw for the witness
        wnet = _make_decoder(rung, in_dim, action_dim, out_dim=int(f_tr.shape[1]))
        wstats = _train_field_decoder(wnet, xs_tr, f_tr, passes, seed, arm_id)
        row["content_witness_training"] = wstats
        row.update(_content_witness(wnet, xs_te, f_te, y_te, row.get("trivial_baseline")))
    else:
        row["content_witness_training"] = None
        row.update({"nonlinear_decode_oracle_agreement": None,
                    "nonlinear_decode_margin_over_trivial": None,
                    "nonlinear_decode_field_r2": None,
                    "nonlinear_decode_decision_coord_r2": None,
                    "nonlinear_decode_collapsed": None,
                    "nonlinear_decoded_flat_window_frac": None})

    # ---- the SAMPLE-SATURATION witness (one designated rung only) ----------------------
    # Separates "content absent" from "map present but this training-set size is too small".
    # Refits the SAME rung on SATURATION_FRACTION of the training ROWS and reports the
    # held-out delta. A drop far beyond the seed spread means the learning curve is still
    # climbing and a CONFIRMED verdict would be confounded with under-sampling. RECORDED, not
    # a gate -- it informs the reading of CONFIRMED rather than blocking it.
    if rung == SATURATION_RUNG and int(xs_tr.shape[0]) > 0:
        n_full = int(xs_tr.shape[0])
        n_half = max(int(n_full * SATURATION_FRACTION), 1)
        reset_all_rng(seed + 20_000)
        sub = torch.randperm(n_full)[:n_half]
        snet = _make_decoder(rung, in_dim, action_dim)
        sstats = _train_decoder(snet, xs_tr[sub], y_tr[sub], passes, seed,
                                arm_id + "@half", action_dim)
        half_agree = _agreement(snet, xs_te, y_te)
        full_agree = row.get("oracle_action_agreement")
        row["sample_saturation_witness"] = {
            "fraction": float(SATURATION_FRACTION),
            "n_train_rows_full": n_full,
            "n_train_rows_subset": n_half,
            "heldout_agreement_full": full_agree,
            "heldout_agreement_subset": half_agree,
            "delta_full_minus_subset": ((float(full_agree) - float(half_agree))
                                        if (full_agree is not None
                                            and half_agree is not None) else None),
            "training": sstats,
        }
    else:
        row["sample_saturation_witness"] = None
    return row


def _print_verdict(row: Dict[str, Any]) -> None:
    agree = row.get("oracle_action_agreement") or 0.0
    elev = row.get("agreement_elevation") or 0.0
    print("verdict: %s" % ("PASS" if (agree >= AGREEMENT_BAR and elev >= AGREEMENT_ELEVATION_MIN)
                           else "FAIL"), flush=True)


def _warm_off_agent(seed: int, env_kwargs: Dict[str, Any], sched: Dict[str, int],
                    dry_run: bool):
    """Reproduce 978's OFF-arm warmup exactly as 1002 and 1008 did (imports, not
    re-definitions). Paid ONCE per seed and shared by all five OFF-track rungs."""
    warm_env = x734._make_env(seed, env_kwargs)
    agent = x1002._make_agent(warm_env)
    before = latent_stack_snapshot(agent)
    stats = x734._train_all_on_agent(
        agent, warm_env, seed=seed, p0_episodes=sched["p0"], p1_episodes=sched["p1"],
        steps_per_episode=sched["steps"], rung_id=RUNG_ID,
        total_denominator=(sched["p0"] + sched["p1"]),
        zworld_p0_episodes=sched["zworld_p0"],
        zworld_p0_env=(x734._make_env(seed, env_kwargs) if sched["zworld_p0"] > 0 else None),
        zworld_p0_dry_run=dry_run,
        zworld_p0_resource_field_weight=0.0,   # 978's OFF arm
    )
    guard = latent_stack_weight_delta(agent, before)
    _ZG.observe(agent)
    return agent, stats, guard


def run_cell(arm_id: str, seed: int, data: Dict[str, Any], feats: Dict[str, Any],
             frozen: Dict[str, Any], action_dim: int, sched: Dict[str, int],
             env_kwargs: Dict[str, Any], cfg_base: Dict[str, Any],
             dry_run: bool) -> Dict[str, Any]:
    ctx = _arm_ctx(arm_id)
    print("Seed %d Condition %s:%s" % (seed, RUNG_ID, arm_id), flush=True)
    cfg_slice = _config_slice(cfg_base, arm_id)
    passes = sched["passes"]

    # An OFF-track cell shares the seed's frozen agent with its four sibling rungs, so the
    # cells are NOT independent -- stamped reuse-ineligible for exactly that reason.
    ineligible = (["frozen_agent_shared_across_capacity_rungs"] if ctx["is_off_track"] else [])

    with arm_cell(seed, config_slice=cfg_slice, script_path=Path(__file__),
                  config_slice_declared=True, include_driver_script_in_hash=False,
                  extra_substrate_paths=_EXTRA_SUBSTRATE_PATHS,
                  extra_ineligible_reasons=ineligible) as cell:
        y_tr, y_te, yr_te = feats["y_tr"], feats["y_te"], feats["yr_te"]
        f_tr = feats["field"]["tr"]
        f_te = feats["field"]["te"]

        if arm_id == ARM_RAW:
            # POSITIVE CONTROL: the raw field at the consumer's width. Warms no encoder.
            x_tr, x_te, xr_te = f_tr, f_te, feats["field"]["r"]
            transform = x1008._DiagZ(x_tr)
            row = _fit_track_cell(arm_id, seed, data, action_dim, passes,
                                  x_tr, y_tr, x_te, y_te, xr_te, yr_te,
                                  transform, None, None)
        elif ctx["is_anchor_track"]:
            # CALIBRATION ANCHOR: PCA-32 of the 250-dim input, 1008's own projection.
            x_tr, x_te, xr_te = feats["ws"]["tr"], feats["ws"]["te"], feats["ws"]["r"]
            if "pca" not in frozen:
                W, stats = x1008._world_state_pca_stats(x_tr, PROJECTION_DIM)
                frozen["pca"] = x1008._LinearProjection(x_tr, W, "pca_32", extra=stats)
            transform = frozen["pca"]
            row = _fit_track_cell(arm_id, seed, data, action_dim, passes,
                                  x_tr, y_tr, x_te, y_te, xr_te, yr_te,
                                  transform, f_tr, f_te)
        else:
            # A z_world track: the frozen 978-OFF latent (THE SUBJECT) or the UNTRAINED latent
            # (the negative control that makes an ELIMINATED verdict attributable). Each is
            # built once per seed and shared across that track's five rungs.
            key = "off" if ctx["is_off_track"] else "unt"
            if key not in frozen:
                if key == "unt":
                    # NO WARMUP -- the same construction with the warmup skipped, exactly as
                    # V3-EXQ-1008's ARM_UNT_DIAG. Costs no warmup time at all.
                    print("  [%s] no warmup (negative control)" % arm_id, flush=True)
                    warm_env = x734._make_env(seed, env_kwargs)
                    agent = x1002._make_agent(warm_env)
                    before = latent_stack_snapshot(agent)
                    wstats: Dict[str, Any] = {}
                    guard = latent_stack_weight_delta(agent, before)
                    _ZG.observe(agent)
                else:
                    agent, wstats, guard = _warm_off_agent(seed, env_kwargs, sched, dry_run)
                frozen[key] = {"agent": agent, "warm_stats": wstats, "warm_guard": guard,
                               "z": x1008._z_feats(agent, data)}
            fz = frozen[key]
            z = fz["z"]
            x_tr, x_te, xr_te = z["tr"], z["te"], z["r"]
            transform = x1008._DiagZ(x_tr)
            # The untrained track carries no field-decode witness: its purpose is the ACTION
            # comparison that attributes an ELIMINATED verdict, and a witness on it would
            # double the negative control's cost for a readout nothing routes on.
            row = _fit_track_cell(arm_id, seed, data, action_dim, passes,
                                  x_tr, y_tr, x_te, y_te, xr_te, yr_te,
                                  transform,
                                  (f_tr if ctx["is_off_track"] else None),
                                  (f_te if ctx["is_off_track"] else None))
            row["warmup_skipped"] = bool(key == "unt")
            row["warmup_stats"] = fz.get("warm_stats")
            row["zworld_weight_delta"] = fz.get("warm_guard")
            # The two quantities x1002's inherited encoder-health preconditions read.
            row["zworld_participation_ratio"] = x1002._participation_ratio(x_tr)

        cell.stamp(row)
    _print_verdict(row)
    return row


# --------------------------------------------------------------------------------------
# ADJUDICATION
# --------------------------------------------------------------------------------------
def _best_over_rungs(rows: List[Dict[str, Any]], track: str, seed: int,
                     key: str = "oracle_action_agreement") -> Tuple[Optional[float],
                                                                    Optional[str]]:
    """The verdict statistic: the MAXIMUM over capacity rungs, with the rung that produced it.
    A max, not the largest rung -- an over-parameterised decoder may overfit and fall, and 'at
    any capacity' is a claim about the best the ladder achieves."""
    best, best_rung = None, None
    for r in rows:
        if r.get("track") != track or int(r.get("seed", -1)) != int(seed):
            continue
        v = r.get(key)
        if v is None:
            continue
        if best is None or float(v) > best:
            best, best_rung = float(v), r.get("capacity_rung")
    return best, best_rung


def _cell(rows: List[Dict[str, Any]], arm_id: str, seed: int) -> Optional[Dict[str, Any]]:
    for r in rows:
        if r.get("arm_id") == arm_id and int(r.get("seed", -1)) == int(seed):
            return r
    return None


def _adjudicate(gate_green: bool, seeds_sufficient: bool, verdict_ready: bool,
                anchor_sound: bool, can_memorise: bool, n_seeds_clearing: int,
                majority: int) -> Tuple[str, str]:
    """The pre-registered verdict grid. Every branch is informative; see the docstring's null
    table. Order matters: seed sufficiency, then the instrument gate, then the readiness of the
    arms the verdict is actually read off, then the two guards -- ALL of them before the content
    verdict, so no instrument or regime failure can ever be reported as H-F CONFIRMED."""
    if not seeds_sufficient:
        return ("insufficient_seeds_for_majority",
                "fewer than SEED_MAJORITY seeds ran; the majority is never lowered")
    if not gate_green:
        return ("substrate_not_ready_requeue",
                "instrument gate not green; no capacity reading is interpretable")
    if not verdict_ready:
        return ("substrate_not_ready_requeue",
                "the arms the verdict is read off are not all green -- an encoder-health "
                "precondition or the frozen-latent reproduction band failed on a z_world arm, "
                "so the capacity sweep is not being run on the object the question is about. "
                "NOT a content finding")
    if not anchor_sound:
        return ("substrate_not_ready_requeue",
                "GUARD 1 failed: the calibration anchor degrades past ANCHOR_DEGRADE_TOL under "
                "added capacity, so this fit protocol cannot support these capacities. The OFF "
                "track's profile is an instrument artifact, NOT a content finding")
    if not can_memorise:
        return ("substrate_not_ready_requeue",
                "GUARD 2 failed: the over-parameterised rung does not reach MEMORISE_FLOOR "
                "train agreement on the frozen latent, so the ladder did not demonstrably "
                "reach over-capacity. Extend the ladder; do NOT read this as content absence")
    if n_seeds_clearing >= majority:
        return ("H-F-eliminated",
                "the decision-relevant content SURVIVES the encode and is recoverable at "
                "above-consumer capacity: the deficit is one of consumer capacity/format, and "
                "the repair is INTERFACE REFORMATTING")
    return ("H-F-confirmed",
            "the decision-relevant content is DESTROYED AT ENCODE TIME: no decoder in the "
            "capacity ladder recovers the oracle above the bar from the frozen latent, on a "
            "protocol the calibration anchor shows is sound and at a capacity that demonstrably "
            "memorises the training split. The repair is at the ENCODER'S OBJECTIVE, not at the "
            "consumer. READ THIS AGAINST `guards.memorising_rungs` AND `diverged_rungs`: the "
            "ladder's reach is the set of rungs that actually FITTED, not its nominal top rung, "
            "and against `sample_saturation_witness`, which bounds how much of the shortfall "
            "could instead be this training-set size")


# --------------------------------------------------------------------------------------
# SELF-TEST (adjudication logic, no compute)
# --------------------------------------------------------------------------------------
_SELF_TEST_ROWS = [
    # (gate, seeds_ok, verdict_ready, anchor_sound, memorise, n_clear, majority, expected)
    (True, True, True, True, True, 0, 2, "H-F-confirmed"),
    (True, True, True, True, True, 1, 2, "H-F-confirmed"),
    (True, True, True, True, True, 2, 2, "H-F-eliminated"),
    (True, True, True, True, True, 3, 2, "H-F-eliminated"),
    # guards precede the content verdict, in both directions
    (True, True, True, False, True, 0, 2, "substrate_not_ready_requeue"),
    (True, True, True, False, True, 3, 2, "substrate_not_ready_requeue"),
    (True, True, True, True, False, 0, 2, "substrate_not_ready_requeue"),
    (True, True, True, True, False, 3, 2, "substrate_not_ready_requeue"),
    (False, True, True, True, True, 0, 2, "substrate_not_ready_requeue"),
    (False, True, True, True, True, 3, 2, "substrate_not_ready_requeue"),
    # THE RED-TEAM'S FINDING 1: a red verdict arm (a failed reproduction band or a failed
    # inherited encoder-health precondition) must block the content verdict, even when the
    # instrument-gate arm alone is green and every guard holds.
    (True, True, False, True, True, 0, 2, "substrate_not_ready_requeue"),
    (True, True, False, True, True, 3, 2, "substrate_not_ready_requeue"),
    # seed sufficiency precedes everything
    (True, False, True, True, True, 3, 2, "insufficient_seeds_for_majority"),
    (False, False, False, False, False, 0, 2, "insufficient_seeds_for_majority"),
]


def _run_self_test() -> int:
    fails = 0
    for gate, sok, vready, anchor, mem, nclear, maj, expect in _SELF_TEST_ROWS:
        label, _why = _adjudicate(gate, sok, vready, anchor, mem, nclear, maj)
        ok = (label == expect)
        fails += (0 if ok else 1)
        print("  [self-test] %s gate=%s seeds=%s vready=%s anchor=%s mem=%s nclear=%d -> %s "
              "(want %s)" % ("ok " if ok else "FAIL", gate, sok, vready, anchor, mem, nclear,
                             label, expect), flush=True)

    # --- whole-grid invariants ---------------------------------------------------------
    # (1) A failed guard OR a red verdict arm NEVER yields a content verdict, at any count.
    for vready in (True, False):
        for anchor in (True, False):
            for mem in (True, False):
                for nclear in range(0, 4):
                    label, _ = _adjudicate(True, True, vready, anchor, mem, nclear, 2)
                    if (not vready or not anchor or not mem) and label.startswith("H-F-"):
                        print("  [self-test] FAIL guard-bypass: vready=%s anchor=%s mem=%s "
                              "nclear=%d -> %s" % (vready, anchor, mem, nclear, label),
                              flush=True)
                        fails += 1
    # (2) The content verdict is monotone in the clearing count.
    labels = [_adjudicate(True, True, True, True, True, n, 2)[0] for n in range(0, 4)]
    if labels != ["H-F-confirmed", "H-F-confirmed", "H-F-eliminated", "H-F-eliminated"]:
        print("  [self-test] FAIL monotonicity: %s" % labels, flush=True)
        fails += 1
    # (3) Every ladder rung is constructible at both a plausible input width and both head
    #     widths, and the consumer rung really is x734's class at its own hidden width.
    for rung in RUNG_IDS:
        for od in (5, RESOURCE_FIELD_DIM):
            net = _make_decoder(rung, 32, 5, out_dim=(None if od == 5 else od))
            rep = _capacity_report(net, rung, 32, 5)
            if rep["action_path_params"] <= 0:
                print("  [self-test] FAIL empty action path at rung %s" % rung, flush=True)
                fails += 1
    consumer = _make_decoder(CONSUMER_RUNG, 32, 5)
    if type(consumer).__name__ != "PPOPolicyNet":
        print("  [self-test] FAIL consumer rung is %s, not x734.PPOPolicyNet"
              % type(consumer).__name__, flush=True)
        fails += 1
    if _capacity_report(consumer, CONSUMER_RUNG, 32, 5)["trunk_hidden"] != int(
            x734.PPO_TRUNK_HIDDEN):
        print("  [self-test] FAIL consumer rung hidden != x734.PPO_TRUNK_HIDDEN", flush=True)
        fails += 1
    # (4) Capacity is strictly increasing along the ladder (the axis is an axis).
    widths = [_capacity_report(_make_decoder(r, 32, 5), r, 32, 5)["action_path_params"]
              for r in RUNG_IDS]
    if widths != sorted(widths) or len(set(widths)) != len(widths):
        print("  [self-test] FAIL capacity not strictly increasing: %s" % widths, flush=True)
        fails += 1
    else:
        print("  [self-test] ok  action-path params by rung: %s"
              % dict(zip(RUNG_IDS, widths)), flush=True)
    # (5) The five decision coordinates map into the 25-dim field block.
    if (len(DECISION_FIELD_INDICES) != 5
            or min(DECISION_FIELD_INDICES) < 0
            or max(DECISION_FIELD_INDICES) >= RESOURCE_FIELD_DIM):
        print("  [self-test] FAIL decision field indices out of range: %s"
              % DECISION_FIELD_INDICES, flush=True)
        fails += 1
    print("[self-test] %d failure(s)" % fails, flush=True)
    return fails


# --------------------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------------------
def run_experiment(seeds: List[int], dry_run: bool = False) -> Dict[str, Any]:
    sched = {
        "zworld_p0": DRY_RUN_ZWORLD_P0 if dry_run else ZWORLD_P0_EPISODES,
        "p0": DRY_RUN_P0 if dry_run else P0_WARMUP_EPISODES,
        "p1": DRY_RUN_P1 if dry_run else P1_REINFORCE_EPISODES,
        "eval_eps": DRY_RUN_EVAL if dry_run else EVAL_EPISODES,
        "steps": DRY_RUN_STEPS if dry_run else STEPS_PER_EPISODE,
        "bc_eps": DRY_RUN_BC_EPISODES if dry_run else BC_EPISODES,
        "bc_rand": DRY_RUN_BC_RANDOM_EPISODES if dry_run else BC_RANDOM_EPISODES,
        "passes": DRY_RUN_ADAPTER_PASSES if dry_run else ADAPTER_PASSES,
    }
    assert_no_structurally_unsatisfiable_gate(PRECONDITION_SPECS, _arm_contexts(),
                                              arm_id_key="arm_id")
    seeds_sufficient = bool(len(seeds) >= SEED_MAJORITY)
    majority = int(SEED_MAJORITY)

    env_kwargs = x734._env_kwargs_for_rung(RUNG)
    cfg_base = x1002._off_path_config_slice(
        dry_run, sched["zworld_p0"], sched["p0"], sched["p1"], sched["steps"],
        sched["bc_eps"], sched["bc_rand"], sched["passes"], sched["eval_eps"])
    probe_env = x734._make_env(seeds[0], env_kwargs)
    action_dim = int(probe_env.action_dim)

    # ---- the 1002 dataset, re-collected from its deterministic recipe, once per seed ----
    per_seed_data: Dict[int, Dict[str, Any]] = {}
    per_seed_feats: Dict[int, Dict[str, Any]] = {}
    anchor_rows: List[Dict[str, Any]] = []
    for s in seeds:
        torch.manual_seed(s)
        np.random.seed(s)
        oracle_eps = x1002._collect_episodes(s, env_kwargs, "oracle", sched["bc_eps"],
                                             sched["steps"])
        rand_eps = x1002._collect_episodes(s, env_kwargs, "random", sched["bc_rand"],
                                           sched["steps"])
        tr, te = x1002._split_episodes(oracle_eps)
        data = {"train": tr, "test": te, "random": rand_eps}
        per_seed_data[s] = data
        f_tr, y_tr = x1002._rawfield_features(tr)
        f_te, y_te = x1002._rawfield_features(te)
        fr_te, yr_te = x1002._rawfield_features(rand_eps)
        w_tr, _ = x1008._world_state_features(tr)
        w_te, _ = x1008._world_state_features(te)
        wr_te, _ = x1008._world_state_features(rand_eps)
        per_seed_feats[s] = {"y_tr": y_tr, "y_te": y_te, "yr_te": yr_te,
                             "field": {"tr": f_tr, "te": f_te, "r": fr_te},
                             "ws": {"tr": w_tr, "te": w_te, "r": wr_te}}
        ev = evaluate_seed(LocalViewGreedyPolicy(s), x734._make_env(s, env_kwargs),
                           sched["eval_eps"], sched["steps"])
        rule_true = x1008._decode_oracle_agreement(f_te, y_te)
        anchor_rows.append({
            "cell_id": "local_view_greedy|seed%d" % s, "anchor_id": "local_view_greedy",
            "seed": int(s), "foraging_competence": float(ev["foraging_competence"]),
            "competence_supra_floor": bool(ev["competence_supra_floor"]),
            "n_train_episodes": len(tr), "n_test_episodes": len(te),
            "n_random_episodes": len(rand_eps),
            "n_train_steps": int(f_tr.shape[0]), "n_heldout_steps": int(f_te.shape[0]),
            "world_state_dim": (int(w_tr.shape[1]) if int(w_tr.shape[0]) else None),
            "oracle_rule_on_true_field_agreement": rule_true["linear_decode_oracle_agreement"],
            "true_field_flat_window_frac": rule_true["decoded_flat_window_frac"],
        })

    # ---- positive control first, on every seed: it gates everything else ---------------
    frozen: Dict[int, Dict[str, Any]] = {s: {} for s in seeds}
    raw_rows = [run_cell(ARM_RAW, s, per_seed_data[s], per_seed_feats[s], frozen[s],
                         action_dim, sched, env_kwargs, cfg_base, dry_run) for s in seeds]

    # ---- run-level DV headroom, from the control's own values --------------------------
    state_blind_vals = [float(r["trivial_baseline"]) for r in raw_rows
                        if r.get("trivial_baseline") is not None]
    raw_agree_vals = [float(r["oracle_action_agreement"]) for r in raw_rows
                      if r.get("oracle_action_agreement") is not None]
    dv_checks = []
    if state_blind_vals:
        dv_checks.append(dv_headroom_check(
            "dv_headroom_agreement_elevation",
            dv_name="oracle_action_agreement_elevation_over_trivial",
            criterion_threshold=float(AGREEMENT_ELEVATION_MIN),
            control_values=state_blind_vals, statistic="ceiling_headroom",
            dv_bounds=(0.0, 1.0), margin=2.0,
            control=("strongest TRIVIAL predictor's held-out agreement per seed -- "
                     "max(state-blind majority class, repeat-previous-executed-action)"),
            description=("The elevation criterion must be reachable: 1.0 minus the worst "
                         "trivial baseline, margin 2.0. Inherited from 1002/1008 unchanged. "
                         "The true DV ceiling is <= 0.015 below 1.0 (flat-window random "
                         "moves, 1008 red-team F9); disclosed, not corrected.")))
    if raw_agree_vals:
        dv_checks.append(dv_headroom_check(
            "dv_headroom_agreement_absolute",
            dv_name="oracle_action_agreement",
            criterion_threshold=float(AGREEMENT_BAR),
            control_values=raw_agree_vals, statistic="max_abs", margin=1.0,
            control="rawfield_ceiling held-out agreement per seed",
            description=("The absolute bar must sit inside what the DV demonstrably reaches "
                         "on this dataset (1008: 0.9730-0.9846 vs 0.80).")))
    try:
        dv_preconditions = p0_readiness_gate(dv_checks) if dv_checks else []
        dv_gate_green, dv_gate_reason = True, ""
    except P0NotReady as e:
        dv_preconditions = list(e.preconditions)
        dv_gate_green = False
        dv_gate_reason = "dv_headroom unmet: " + ", ".join(
            str(p.get("name")) for p in dv_preconditions if not p.get("met"))

    raw_worst, raw_worst_cell = x1002._worst_cell(raw_rows, "oracle_action_agreement", "min")
    nte_worst, _nc = x1002._worst_cell(raw_rows, "n_heldout_steps", "min")
    lvg_worst, lvg_worst_cell = x1002._worst_cell(anchor_rows, "foraging_competence", "min")
    instrument_ready = bool(raw_worst is not None
                            and raw_worst >= RAW_FIELD_CONTROL_FLOOR) and dv_gate_green

    # ---- the ten swept cells, per seed (anchor track first: it is cheap and gates nothing,
    #      the OFF track pays the warmup once) -------------------------------------------
    # `or dry_run`: the smoke MUST execute every arm including the warmup and every rung; at
    # dry scale the control cannot reach its floor and the gate would otherwise short-circuit
    # past the code this run exists to exercise (V3-EXQ-591g).
    other_rows: List[Dict[str, Any]] = []
    _swept_order = PCA_ARM_IDS + UNT_ARM_IDS + OFF_ARM_IDS
    for s in seeds:
        if instrument_ready or dry_run:
            for aid in _swept_order:
                other_rows.append(run_cell(aid, s, per_seed_data[s], per_seed_feats[s],
                                           frozen[s], action_dim, sched, env_kwargs,
                                           cfg_base, dry_run))
        else:
            for aid in _swept_order:
                print("Seed %d Condition %s:%s" % (s, RUNG_ID, aid), flush=True)
                print("  [skip] instrument not certified; arm not run", flush=True)
                print("verdict: FAIL", flush=True)
        frozen[s].clear()      # release the agent and its cached features
    all_rows = raw_rows + other_rows

    # ---- the verdict statistic + the two guards ----------------------------------------
    per_seed_verdict: List[Dict[str, Any]] = []
    for s in seeds:
        off_best, off_best_rung = _best_over_rungs(other_rows, TRACK_OFF, s)
        pca_best, pca_best_rung = _best_over_rungs(other_rows, TRACK_PCA, s)
        off_c = _cell(other_rows, _arm_id(TRACK_OFF, CONSUMER_RUNG), s)
        pca_c = _cell(other_rows, _arm_id(TRACK_PCA, CONSUMER_RUNG), s)
        off_max = _cell(other_rows, _arm_id(TRACK_OFF, MAX_CAPACITY_RUNG), s)
        pca_max = _cell(other_rows, _arm_id(TRACK_PCA, MAX_CAPACITY_RUNG), s)
        # The elevation the best OFF rung achieved, read from that rung's own cell.
        off_best_cell = None
        for r in other_rows:
            if (r.get("track") == TRACK_OFF and int(r.get("seed", -1)) == int(s)
                    and r.get("capacity_rung") == off_best_rung):
                off_best_cell = r
                break
        off_elev = (off_best_cell or {}).get("agreement_elevation")
        clears = bool(off_best is not None and off_best >= AGREEMENT_BAR
                      and off_elev is not None and off_elev >= AGREEMENT_ELEVATION_MIN)
        # GUARD 1's statistic: the MAX-CAPACITY rung minus the CONSUMER rung. Deliberately NOT
        # (best - consumer): `best` is a max over a set that CONTAINS the consumer rung, so that
        # difference is >= 0 by construction and the guard could never fail.
        anchor_delta = ((float(pca_max["oracle_action_agreement"])
                         - float(pca_c["oracle_action_agreement"]))
                        if (pca_max is not None and pca_c is not None
                            and pca_max.get("oracle_action_agreement") is not None
                            and pca_c.get("oracle_action_agreement") is not None) else None)
        per_seed_verdict.append({
            "seed": int(s),
            "off_best_agreement": off_best,
            "off_best_rung": off_best_rung,
            "off_best_elevation": off_elev,
            "off_clears_bar_and_elevation": clears,
            "off_consumer_rung_agreement": (off_c or {}).get("oracle_action_agreement"),
            "off_max_capacity_train_agreement": (
                (off_max or {}).get("oracle_action_agreement_train")),
            # GUARD 2's statistic: the best TRAIN agreement the ladder reaches on this seed,
            # over ALL rungs -- see the precondition's description for why not the top rung.
            "off_best_train_agreement": _best_over_rungs(
                other_rows, TRACK_OFF, s, key="oracle_action_agreement_train")[0],
            "off_best_train_rung": _best_over_rungs(
                other_rows, TRACK_OFF, s, key="oracle_action_agreement_train")[1],
            "off_train_agreement_by_rung": {
                str(r.get("capacity_rung")): r.get("oracle_action_agreement_train")
                for r in other_rows
                if r.get("track") == TRACK_OFF and int(r.get("seed", -1)) == int(s)},
            "off_final_ce_by_rung": {
                str(r.get("capacity_rung")):
                    ((r.get("decoder_training") or {}).get("final_ce_loss"))
                for r in other_rows
                if r.get("track") == TRACK_OFF and int(r.get("seed", -1)) == int(s)},
            "anchor_final_ce_by_rung": {
                str(r.get("capacity_rung")):
                    ((r.get("decoder_training") or {}).get("final_ce_loss"))
                for r in other_rows
                if r.get("track") == TRACK_PCA and int(r.get("seed", -1)) == int(s)},
            "anchor_best_agreement": pca_best,
            "anchor_best_rung": pca_best_rung,
            "anchor_consumer_rung_agreement": (pca_c or {}).get("oracle_action_agreement"),
            "anchor_max_capacity_agreement": (pca_max or {}).get("oracle_action_agreement"),
            "anchor_capacity_delta": anchor_delta,
            "off_nonlinear_witness_by_rung": {
                str(r.get("capacity_rung")): r.get("nonlinear_decode_oracle_agreement")
                for r in other_rows
                if r.get("track") == TRACK_OFF and int(r.get("seed", -1)) == int(s)},
            "off_agreement_by_rung": {
                str(r.get("capacity_rung")): r.get("oracle_action_agreement")
                for r in other_rows
                if r.get("track") == TRACK_OFF and int(r.get("seed", -1)) == int(s)},
            "anchor_agreement_by_rung": {
                str(r.get("capacity_rung")): r.get("oracle_action_agreement")
                for r in other_rows
                if r.get("track") == TRACK_PCA and int(r.get("seed", -1)) == int(s)},
            # THE NEGATIVE CONTROL. An ELIMINATED verdict is only attributable to the ENCODE if
            # the UNTRAINED latent does NOT also clear at some capacity: 1008 measured the
            # untrained latent at 0.699/0.681/0.704 against the trained OFF latent's
            # 0.672/0.674/0.661 -- ABOVE it on every seed -- so "OFF clears at high capacity"
            # would otherwise be indistinguishable from "any 250->32 bottleneck does".
            "untrained_best_agreement": _best_over_rungs(other_rows, TRACK_UNT, s)[0],
            "untrained_best_rung": _best_over_rungs(other_rows, TRACK_UNT, s)[1],
            "untrained_agreement_by_rung": {
                str(r.get("capacity_rung")): r.get("oracle_action_agreement")
                for r in other_rows
                if r.get("track") == TRACK_UNT and int(r.get("seed", -1)) == int(s)},
            "off_minus_untrained_best": (
                (float(_best_over_rungs(other_rows, TRACK_OFF, s)[0])
                 - float(_best_over_rungs(other_rows, TRACK_UNT, s)[0]))
                if (_best_over_rungs(other_rows, TRACK_OFF, s)[0] is not None
                    and _best_over_rungs(other_rows, TRACK_UNT, s)[0] is not None) else None),
            "sample_saturation": next(
                (r.get("sample_saturation_witness") for r in other_rows
                 if r.get("track") == TRACK_OFF and int(r.get("seed", -1)) == int(s)
                 and r.get("capacity_rung") == SATURATION_RUNG), None),
        })

    n_seeds_clearing = sum(1 for v in per_seed_verdict if v["off_clears_bar_and_elevation"])

    # WORST CELL, not the mean -- `met` is a worst-case claim and the indexer recomputes it.
    # `cell_id` is supplied so the offending seed is NAMED rather than reported as null.
    anchor_worst, anchor_worst_seed = x1002._worst_cell(
        [{"cell_id": "ws250_pca__%s_vs_%s|seed%d" % (MAX_CAPACITY_RUNG, CONSUMER_RUNG,
                                                     v["seed"]),
          "anchor_capacity_delta": v["anchor_capacity_delta"]}
         for v in per_seed_verdict if v["anchor_capacity_delta"] is not None],
        "anchor_capacity_delta", "min")
    mem_worst, mem_worst_seed = x1002._worst_cell(
        [{"cell_id": "zworld_off__%s|seed%d" % (v["off_best_train_rung"], v["seed"]),
          "off_best_train_agreement": v["off_best_train_agreement"]}
         for v in per_seed_verdict
         if v["off_best_train_agreement"] is not None],
        "off_best_train_agreement", "min")
    # The band's two legs take OPPOSITE worst cases: the floor is worst at the minimum seed,
    # the ceiling at the maximum. Reporting one number for both would let an out-of-band cell
    # recompute as MET on whichever leg it did not extremise.
    _repro_rows = [{"cell_id": "zworld_off__mlp128|seed%d" % v["seed"],
                    "off_consumer_rung_agreement": v["off_consumer_rung_agreement"]}
                   for v in per_seed_verdict if v["off_consumer_rung_agreement"] is not None]
    repro_worst, repro_worst_seed = x1002._worst_cell(
        _repro_rows, "off_consumer_rung_agreement", "min")
    repro_high, repro_high_seed = x1002._worst_cell(
        _repro_rows, "off_consumer_rung_agreement", "max")

    anchor_sound = bool(anchor_worst is not None and anchor_worst >= -ANCHOR_DEGRADE_TOL)
    can_memorise = bool(mem_worst is not None and mem_worst >= MEMORISE_FLOOR)

    # ---- per-arm precondition gates -----------------------------------------------------
    # `measured` is computed PER ARM (x1002's and 1008's own pattern), because the encoder-health
    # preconditions read THAT ARM's own cells -- a globally-computed value would let one track's
    # latent certify another's.
    maj_worst, _mc = x1002._worst_cell(raw_rows, "train_majority_class_share", "max")
    arm_gates = []
    for ctx in _arm_contexts():
        aid = str(ctx["id"])
        rows_a = [r for r in all_rows if r.get("arm_id") == aid]
        measured: Dict[str, Any] = {
            # x1002's inherited specs
            "adapter_capacity_sufficient_on_raw_field": raw_worst,
            "oracle_labels_non_degenerate": maj_worst,
            "heldout_split_sufficient": nte_worst,
            "d3_local_view_greedy_clears_floor": lvg_worst,
            # this run's own specs
            "overcapacity_fit_protocol_sound": anchor_worst,
            "overcapacity_decoder_can_memorise": mem_worst,
            "frozen_latent_regime_reproduced_floor": repro_worst,
            "frozen_latent_regime_reproduced_ceiling": repro_high,
        }
        if ctx["has_encoder"]:
            prmin, _pc = (x1002._worst_cell(rows_a, "zworld_participation_ratio", "min")
                          if rows_a else (0.0, None))
            measured["zworld_not_collapsed"] = prmin
        if ctx["trained_encoder"]:
            dmin, _dc = (x1002._worst_cell(
                [{"cell_id": r.get("cell_id"),
                  "d": float((r.get("zworld_weight_delta") or {}).get(
                      "world_encoder_max_abs_delta", 0.0) or 0.0)}
                 for r in rows_a], "d", "min") if rows_a else (0.0, None))
            measured["zworld_encoder_trained_in_p0"] = dmin
        arm_gates.append(evaluate_arm_gate(aid, ctx, PRECONDITION_SPECS, measured))
    per_arm_gate = aggregate_arm_gates(arm_gates)
    green = set(per_arm_gate.get("green_arms") or [])
    gate_green = bool(per_arm_gate.get("non_degenerate")) and dv_gate_green

    # THE VERDICT'S OWN READINESS. `non_degenerate` is `any arm green` -- correct, because one
    # red arm must never vacate a good one -- but it is NOT sufficient to license THIS verdict:
    # the instrument-gate arm alone going green would satisfy it while the arms the verdict is
    # actually read off sat red. So require the specific arms by name, exactly as V3-EXQ-1008's
    # `leg1_ready` does. This is what gives the reproduction band (and the inherited
    # encoder-health preconditions, which are scoped to the z_world arms) a route to the verdict.
    verdict_arms = ([_arm_id(TRACK_OFF, r) for r in RUNG_IDS]
                    + [_arm_id(TRACK_PCA, CONSUMER_RUNG),
                       _arm_id(TRACK_PCA, MAX_CAPACITY_RUNG)])
    verdict_ready = bool(all(a in green for a in verdict_arms))
    verdict_red_arms = [a for a in verdict_arms if a not in green]

    label, why = _adjudicate(gate_green, seeds_sufficient, verdict_ready, anchor_sound,
                             can_memorise, n_seeds_clearing, majority)
    outcome = "PASS" if (gate_green and seeds_sufficient and verdict_ready and anchor_sound
                         and can_memorise and label.startswith("H-F-")) else "FAIL"

    criteria = [
        {"name": "C_instrument_gate_green", "load_bearing": False, "passed": bool(gate_green)},
        {"name": "C_verdict_arms_ready", "load_bearing": True, "passed": bool(verdict_ready)},
        {"name": "C_guard1_anchor_protocol_sound", "load_bearing": True,
         "passed": bool(anchor_sound)},
        {"name": "C_guard2_overcapacity_memorises", "load_bearing": True,
         "passed": bool(can_memorise)},
        {"name": "C_hf_adjudicated", "load_bearing": True,
         "passed": bool(label.startswith("H-F-"))},
        {"name": "C_off_clears_at_some_capacity", "load_bearing": False,
         "passed": bool(n_seeds_clearing >= majority)},
    ]
    # arm_id -> the criterion names that arm OWNS. A criterion whose statistic is a MAX OVER
    # RUNGS is owned by the CONSUMER rung, not the top rung: the top rung is one input to that
    # max and may legitimately have diverged (see GRAD_CLIP_NORM), so keying the criterion there
    # would report non_degenerate=False for a run whose verdict came cleanly off another rung.
    crit_non_degenerate = arm_criteria_non_degenerate(
        {
            ARM_RAW: ["C_instrument_gate_green"],
            _arm_id(TRACK_PCA, CONSUMER_RUNG): ["C_guard1_anchor_protocol_sound"],
            _arm_id(TRACK_OFF, CONSUMER_RUNG): ["C_verdict_arms_ready",
                                                "C_guard2_overcapacity_memorises",
                                                "C_hf_adjudicated",
                                                "C_off_clears_at_some_capacity"],
        },
        per_arm_gate,
    )

    interpretation = {
        "label": label,
        "why": why,
        "hypothesis_qid": HYPOTHESIS_QID,
        "hypothesis_leg": HYPOTHESIS_LEG,
        "hypothesis_verdict": (label if label.startswith("H-F-") else "no_verdict"),
        "combination_rule": ("outcome PASS iff the instrument gate is green AND both guards "
                             "hold AND the H-F leg reaches a verdict. The SCIENCE is in "
                             "hypothesis_verdict, not in PASS/FAIL: both H-F-confirmed and "
                             "H-F-eliminated are PASS, and each routes a different build."),
        "preconditions": list(per_arm_gate.get("adjudication_preconditions") or [])
                         + list(dv_preconditions),
        "criteria_non_degenerate": crit_non_degenerate,
        "seeds_sufficient": bool(seeds_sufficient),
        "n_seeds_clearing_at_some_capacity": int(n_seeds_clearing),
        "seed_majority": majority,
        "verdict_statistic": "best_agreement_over_capacity (max over rungs, per track per seed)",
    }

    # NOTE: run_id / timestamp_utc / architecture_epoch / queue_id are stamped in __main__ and
    # elapsed_seconds by the writer's `started_at`; they are deliberately NOT set twice here.
    manifest: Dict[str, Any] = {
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "architecture_epoch": ARCHITECTURE_EPOCH,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "claim_ids": list(CLAIM_IDS),
        "bears_on": list(BEARS_ON),
        "level_id": LEVEL_ID,
        "rung_id": RUNG_ID,
        "sleep_driver_pattern": "none",
        "outcome": outcome,
        "dry_run": bool(dry_run),
        "interpretation": interpretation,
        "criteria": criteria,
        "per_arm_gate": per_arm_gate,
        "capacity_ladder": [{"rung": r, "kind": k, "hidden": h, "depth": d}
                            for r, k, h, d in CAPACITY_LADDER],
        "tracks": {"subject": TRACK_OFF, "calibration_anchor": TRACK_PCA,
                   "instrument_gate": ARM_RAW},
        "per_seed_verdict": per_seed_verdict,
        "arm_results": all_rows,
        "oracle_anchor_results": anchor_rows,
        "instrument": {
            "rawfield_worst_seed_agreement": raw_worst,
            "rawfield_worst_cell": raw_worst_cell,
            "rawfield_control_floor": float(RAW_FIELD_CONTROL_FLOOR),
            "heldout_steps_worst_seed": nte_worst,
            "oracle_competence_worst_seed": lvg_worst,
            "oracle_competence_worst_cell": lvg_worst_cell,
            "dv_gate_green": bool(dv_gate_green),
            "dv_gate_reason": dv_gate_reason,
        },
        "guards": {
            "verdict_ready": bool(verdict_ready),
            "verdict_arms": list(verdict_arms),
            "verdict_arms_red": list(verdict_red_arms),
            # Which rungs actually FITTED, and which diverged -- the ladder's real reach, which
            # a CONFIRMED verdict must be read against rather than against its nominal top rung.
            "memorising_rungs": sorted({
                str(r.get("capacity_rung")) for r in other_rows
                if r.get("track") == TRACK_OFF
                and r.get("oracle_action_agreement_train") is not None
                and float(r["oracle_action_agreement_train"]) >= MEMORISE_FLOOR}),
            "diverged_rungs": sorted({
                "%s/%s" % (r.get("track"), r.get("capacity_rung")) for r in other_rows
                if (r.get("decoder_training") or {}).get("diverged")}),
            "collapsed_witness_rungs": sorted({
                "%s/%s" % (r.get("track"), r.get("capacity_rung")) for r in other_rows
                if r.get("nonlinear_decode_collapsed")}),
            "grad_clip_norm": float(GRAD_CLIP_NORM),
            "anchor_capacity_delta_worst_seed": anchor_worst,
            "anchor_capacity_delta_worst_cell": anchor_worst_seed,
            "anchor_degrade_tolerance": float(-ANCHOR_DEGRADE_TOL),
            "anchor_sound": bool(anchor_sound),
            "overcapacity_best_train_agreement_worst_seed": mem_worst,
            "overcapacity_best_train_agreement_worst_cell": mem_worst_seed,
            "memorise_floor": float(MEMORISE_FLOOR),
            "can_memorise": bool(can_memorise),
            "memorise_statistic": ("max over capacity rungs of the OFF track's TRAIN-split "
                                   "agreement, then the worst seed of that"),
            "off_consumer_rung_min_seed": repro_worst,
            "off_consumer_rung_min_cell": repro_worst_seed,
            "off_consumer_rung_max_seed": repro_high,
            "off_consumer_rung_max_cell": repro_high_seed,
            "off_reproduction_band": [float(OFF_REPRO_LOW), float(OFF_REPRO_HIGH)],
        },
        "pre_registered": {
            "agreement_bar": float(AGREEMENT_BAR),
            "agreement_elevation_min": float(AGREEMENT_ELEVATION_MIN),
            "seed_majority": majority,
            "anchor_degrade_tol": float(ANCHOR_DEGRADE_TOL),
            "memorise_floor": float(MEMORISE_FLOOR),
            "off_reproduction_band": [float(OFF_REPRO_LOW), float(OFF_REPRO_HIGH)],
        },
        "reference_values_1008": {
            "note": ("recorded so a later reader can see what this run reproduces; NEVER used "
                     "as a threshold -- every gate is applied to this run's own in-run values"),
            "zworld_off_diag_mlp128": [0.6718, 0.6735, 0.6606],
            "ws250_pca_mlp128": [0.8771, 0.8578, 0.8702],
            "rawfield_ceiling": [0.9846, 0.9796, 0.9730],
            "linear_content_witness": [0.5931, 0.5910, 0.6066],
            "trivial_baseline": [0.5661, 0.5803, 0.5720],
        },
    }
    return manifest


if __name__ == "__main__":
    import argparse
    import time
    from datetime import datetime

    parser = argparse.ArgumentParser(description=EXPERIMENT_TYPE)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--self-test", action="store_true",
                        help="Push the pre-registered synthetic rows through the verdict grid "
                             "and the ladder invariants, then exit. Seconds, no env.")
    args = parser.parse_args()

    if args.self_test:
        sys.exit(_run_self_test())

    t0 = time.perf_counter()
    seeds = args.seeds if args.seeds else (DRY_RUN_SEEDS if args.dry_run else SEEDS)
    print("%s: seeds=%s dry_run=%s" % (EXPERIMENT_TYPE, seeds, bool(args.dry_run)), flush=True)

    result = run_experiment(list(seeds), dry_run=bool(args.dry_run))

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    result["run_timestamp"] = ts
    result["timestamp_utc"] = ts
    result["run_id"] = "%s_%s_v3" % (EXPERIMENT_TYPE, ts)
    result["architecture_epoch"] = ARCHITECTURE_EPOCH
    result["queue_id"] = QUEUE_ID

    full_config = {
        "rung": RUNG, "level_id": LEVEL_ID,
        "env_kwargs": x734._env_kwargs_for_rung(RUNG),
        "zworld_p0_episodes": (DRY_RUN_ZWORLD_P0 if args.dry_run else ZWORLD_P0_EPISODES),
        "p0_warmup_episodes": (DRY_RUN_P0 if args.dry_run else P0_WARMUP_EPISODES),
        "p1_reinforce_episodes": (DRY_RUN_P1 if args.dry_run else P1_REINFORCE_EPISODES),
        "eval_episodes": (DRY_RUN_EVAL if args.dry_run else EVAL_EPISODES),
        "steps_per_episode": (DRY_RUN_STEPS if args.dry_run else STEPS_PER_EPISODE),
        "bc_episodes": (DRY_RUN_BC_EPISODES if args.dry_run else BC_EPISODES),
        "bc_random_episodes": (DRY_RUN_BC_RANDOM_EPISODES if args.dry_run
                               else BC_RANDOM_EPISODES),
        "bc_train_frac": x1002.BC_TRAIN_FRAC,
        "adapter_passes": (DRY_RUN_ADAPTER_PASSES if args.dry_run else ADAPTER_PASSES),
        "adapter_batch": ADAPTER_BATCH, "adapter_lr": ADAPTER_LR,
        # THE MANIPULATED VARIABLE, declared in the config the manifest carries.
        "capacity_ladder": [{"rung": r, "kind": k, "hidden": h, "depth": d}
                            for r, k, h, d in CAPACITY_LADDER],
        "capacity_consumer_rung": CONSUMER_RUNG,
        "capacity_max_rung": MAX_CAPACITY_RUNG,
        "consumer_trunk_hidden": int(x734.PPO_TRUNK_HIDDEN),
        "tracks": list(SWEPT_TRACKS),
        "decoder_init_reseeded_per_fit": True,
        "projection_dim": PROJECTION_DIM,
        "agreement_bar": AGREEMENT_BAR, "agreement_elevation_min": AGREEMENT_ELEVATION_MIN,
        "anchor_degrade_tol": ANCHOR_DEGRADE_TOL, "memorise_floor": MEMORISE_FLOOR,
        "off_reproduction_band": [OFF_REPRO_LOW, OFF_REPRO_HIGH],
        "decision_world_state_indices": DECISION_WORLD_STATE_INDICES,
        "decision_field_indices": DECISION_FIELD_INDICES,
        "raw_field_control_floor": RAW_FIELD_CONTROL_FLOOR,
        "heldout_min_steps": x1002.HELDOUT_MIN_STEPS,
        "seed_majority": SEED_MAJORITY, "arms": ARM_IDS,
        "hypothesis_qid": HYPOTHESIS_QID, "hypothesis_leg": HYPOTHESIS_LEG,
        "dry_run": bool(args.dry_run),
    }
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
