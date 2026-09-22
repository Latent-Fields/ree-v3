#!/opt/local/bin/python3
"""
V3-EXQ-1073 -- MECH-572: does precision-PROVENANCE-conditioned sleep plasticity
gain beat a matched generic gain reduction, without self-sealing?

SLEEP DRIVER: manual-cycle-loop (force_cycle() called once per cycle in a
              dedicated N_CYCLES wake-sleep-test loop)

WHY THIS RUN
-------------
User brief (2026-09-22): behavioural precision provenance and sleep plasticity
gain. The intake asks whether the organism PRESERVES, from waking experience,
enough epistemic provenance to regulate how strongly later sleep rewrites the
world model -- and whether doing so helps, or merely builds a self-sealing
protection for whatever the model already believes.

The route into it is V3-EXQ-1063's Result 6. On a CONVERGED base a sleep cycle
made the frozen held-out world-forward MSE WORSE in 9 of 9 probe cells, and the
world-head displacement sat at 98.7% of the fresh-Adam bound 8*CMC_LR. That is
MECH-572's phenotype: a FRESH torch.optim.Adam per consolidate() call, whose
bias-corrected first step is ~lr*sign(g) REGARDLESS of |g|, so the consolidator
perturbs a converged head by far more than that head's own residual and cannot
be talked out of it by any per-row loss weight (Adam normalises the magnitude
away).

The user's decision (recommendation ledger entry 520) was "build full producers
first, then experiment". SD-PP-1..4 are that build. This run is the experiment.

THE AUDIT THAT MOTIVATED THE PRODUCERS (one paragraph)
-------------------------------------------------------
Verified live against ree-v3 a38a834 (substrate spec section 0): on the
consolidation path that produces the MECH-572 phenotype, NONE of the four
quantities the intake distinguishes survives from waking into replay. The
experience buffers hold only (z_world, action_one_hot); no historical
prediction precision is preserved; no evidence/sensory precision producer
existed anywhere in ree_core (the grep was empty); under the V3-EXQ-1063
configuration nothing even COMPUTES the world-forward prediction for the
executed action at test time (the three paths that do -- MECH-353
blocked-agency, SD-063 deficit, the escape linker -- are all default-off);
behavioural success/failure is not preserved; and none of it reaches replay
SELECTION (torch.randperm) or consolidation GAIN (fixed lr). The hippocampal
AnchorSet / SleepReplaySampler path the intake inspected is not on this path at
all -- MECH-285 Phase B is a no-op consumer feeding SWS aggregation, while the
weight-consolidation pass draws from the raw experience buffers. So the
"episodic trace" for this experiment IS the world experience buffer, and
provenance had to be built to ride alongside it: SD-PP-1 (evidence precision),
SD-PP-2 (model precision), SD-PP-3 (the per-transition packet), SD-PP-4 (the
gain rule + the consolidator's per-step lr hook).

THE THREE PRECISION QUANTITIES, KEPT DISTINCT (preregistration section 2)
---------------------------------------------------------------------------
  quantity                      producer                 read when
  ----------------------------  -----------------------  --------------------
  historical model precision    SD-PP-2 precision_at()   at the waking test,
    pi_hist                                              BEFORE the outcome
  evidence precision  pi_e      SD-PP-1                  at the outcome
    (z units)                   evidence_precision_z
  current model precision       SD-PP-2 current_read()   at sleep entry
    pi_cur

Historical precision is PROVENANCE, NOT PROTECTION. It enters the gain rule
ONLY through the surprise reopen factor r_i >= 1. Any implementation in which
high pi_hist REDUCES learning is invalid for the main hypothesis.

ARMS (preregistration section 6) -- all fork from the identical pre-sleep state
--------------------------------------------------------------------------------
  arm               metadata  gain                            purpose
  ----------------  --------  ------------------------------  -----------------
  ARM_A_BASELINE    no        fresh Adam, fixed lr            current behaviour
  ARM_B_STORE_ONLY  yes       as A                            storage neutrality
  ARM_C_PROVENANCE  yes       rule, mode "provenance"         treatment
  ARM_C_NOHIST      yes       mode "provenance_nohist" (r=1)  is pi_hist
                                                              load-bearing (F1)?
  ARM_D_GLOBAL      yes       mode "global", scale c_seed     matched-budget
                                                              rival: any lr cut
  ARM_D_RESIDUAL    yes       mode "residual_only" (BATCH 4:  rival scheduler:
                              per-row g_i = clip(gain_max *  CURRENT residual
                              sqrt(pe_cur_i/v_ref), min, max) only (F2)
                              from the CURRENT residual;
                              IGNORES global_scale)

c_seed = the mean realised per-step gain of ARM C for that seed, pooled over ALL
conditions and cycles (weighted by the number of e2_world steps in each cycle),
read from the gain diagnostics and NEVER from any DV. ARM_D_GLOBAL therefore
runs AFTER C for a seed. Its budget equivalence is exact in TOTAL and
deliberately not per condition -- the rival hypothesis is a single global
reduction -- so in any GIVEN condition Dg runs at a different gain from C, and
the realised ratios are recorded as p6_budget_ratio_<cond>_s<seed> and MUST be
read alongside P6 (see the PRE-FREEZE PROBE section: Dg runs ~12x C's gain in
cond 1 and ~7x in cond 3).

ARM_D_RESIDUAL is a DIFFERENT KIND of rival and is NOT budget-matched (BATCH 4).
It is the genuine current-residual scheduler: the same magnitude factor m_i that
C uses, but with K = 1, r = 1, and the CURRENT per-row MSE of e2.world_forward
on the replayed triple in place of the STORED innovation. So it reallocates
plasticity across conditions from the current residual alone, which is exactly
the intake's F2 question -- does evidence precision add anything beyond raw
error? -- and it needs no c_seed, so it runs in phase 1.

CONDITIONS (preregistration section 7) -- seed-paired across arms
-------------------------------------------------------------------
  cond                      base       P1 waking          batteries
  ------------------------  ---------  -----------------  -------------------
  1 COND_CONVERGED_CLEAN    converged  clean              R0 (retention)
  2 COND_UNDERFIT           fresh      clean              R0
  3 COND_CONFIDENTLY_WRONG  converged  action map         R0 ("letting go")
                                       inverted, but      + R1 (correction)
                                       reliable
  4 COND_NOISY_CONTRADICT.  converged  additive obs       R0
                                       noise sigma 0.12

Factorial mapping (hist x ev x cur): (H,H,H), (L,H,L), (H,H,cur drops), (H,L,H).
A reduced matched subset, chosen because the full 2x2x2 would need evidence
precision manipulated INSIDE the converged-clean regime, which the same noise
hook provides only as condition 4. The three quantities stay separately
identified: ev by cond 4 vs 3, cur by cond 1 vs 2, hist by ARM C vs C0 in cond 3.

THE GAIN RULE (preregistration section 5 / SD-PP-4)
----------------------------------------------------
    K_i    = pi_e_i / (pi_e_i + pi_cur)                   write authority (Kalman)
    m_i    = sqrt( max(pe_i - NOISE_GAIN*ev_var_i, 0) / V_REF )   innovation size
    r_i    = min( R_MAX, 1 + BETA * max(0, ln(surprise_i)) )      reopen factor
    gain_i = clip( G_MAX * K_i * m_i * r_i, G_MIN, G_MAX )
    loss   = sum_i gain_i*l_i / sum_i gain_i ;  lr_step = lr * mean_i gain_i
    NOISE_GAIN 2.0   V_REF 1e-2   BETA 0.5   R_MAX 3.0   G_MIN 0.02   G_MAX 2.0

The gain acts in TWO places and both are pre-registered: per-row weights set the
DIRECTION; the per-step lr scale moves the DISPLACEMENT. Per-row weights alone
would be normalised away by Adam -- that is the whole MECH-572 point. Replay
CONTENT, ORDER and COUNT are untouched: same randperm draw, same K rows, same 8
steps.

PRE-RUN GATES (preregistration section 9; STOP conditions)
------------------------------------------------------------
  G1 liveness       world_head_max_abs_delta strictly increasing over
                    global_scale in (0.1, 0.5, 1.0, 2.0) at a pinned seed.
                    Also runnable standalone via --liveness.
  G2 A/B neutrality world buffer + world-head params + the torch RNG state
                    entering force_cycle are BITWISE identical across all six
                    arms of a (seed, condition) at cycle 1.
  G3 evidence chan. evidence_precision_z cond4 < 0.25 x cond1; and
                    |log2(cond3/cond1)| <= 1 (it must NOT alias the rule shift).
  G4 hist. precis.  sd(pi_hist) > 0 in every condition; cond3's first 30 real
                    post-shift packets carry surprise > 10 on >= 50% of them
                    while cond1's median surprise is < 3.
  G5 curr. precis.  pi_cur cond1 > 10 x pi_cur cond2.
  G6 displacement   world_head_max_abs_delta > 0 in ARM A of every condition.
  G7 readout var.   ARM A across-sleep MSE delta not identically 0; no nan; no
                    post-sleep MSE at the 1e-8 floor in any arm.
  G8 readability    converged-base skill (1 - mse/identity_mse) > 0 on >= 2/3
                    seeds; else the retention contrasts are non_degenerate=False.
  G9 PE separation  cond3 mean waking pe > 2 x the converged residual on >= 2/3
                    seeds; else cond3 is non_degenerate=False.

G1-G7 ROUTE PASS/FAIL, exactly as V3-EXQ-1063's C1..C6 do. G1-G6 failing gives
"no P read at all" (label R5 / R6 / substrate_not_ready_requeue as the matrix
directs); G7 failing alone gives a measurement-invalid FAIL. G8/G9 route NO
pass/fail -- they mark specific contrasts non-citable without vacating the rest
(failure_autopsy_V3-EXQ-785_2026-07-19 sections 2a/8).

PRIMARY CONTRASTS (preregistration section 11). ret = post/pre - 1 on the
retention battery; corr = (pre - post)/pre on the R1 battery. MARGIN = 0.02
absolute on these relative quantities; where the per-seed SD of the paired
deltas exceeds it the SD-scaled margin (1 SD) is reported alongside AND THE
DECISION USES THE LARGER. The pre-registered decision rule is a 2-of-3 seed sign
count; the t-based 95% CI (n=3) is descriptive and flagged low-power.
  P1  A vs B          bitwise identical (G2); readout tolerance 0.
  P2  B vs C, cond 1  ret(C) < ret(B) - MARGIN  AND  ret(C) < 0.10.
  P3  B vs C, cond 2  ret(C) <= ret(B) + MARGIN  (C still learns when underfit).
  P4  anti-self-seal  cond 3: corr(C) >= corr(B) - MARGIN  AND  corr(C) > 0.
  P5  evidence prec.  cond 4: ret(C) < ret(Dr) - MARGIN.
  P6  C vs Dg         cond 1 ret(C) < ret(Dg) - MARGIN  AND  cond 3
                      corr(C) > corr(Dg) + MARGIN, simultaneously, per seed.
  P7  C vs C0         cond 3: corr(C) > corr(C0) + MARGIN. A null HERE IS A
                      RESULT (historical precision not shown to add information).
NO P ROUTES PASS/FAIL. P2, P4 and P6 are load_bearing because they route
interpretation.label.

INTERPRETATION MATRIX (preregistration section 12, pre-registered)
-------------------------------------------------------------------
  R1  P2,P3,P4,P6 pass        -> provenance_beyond_generic_gain
  R2  P2/P4 pass, P6 fails    -> generic_gain_correction_sufficient
  R3  P2 pass, P4 fails       -> self_sealing_failure
  R5  G1 or G6 fails          -> instrument_inert_substrate_not_ready
  R6  G3/G4/G5 fails          -> precision_channel_degenerate
  R7  P5 fails                -> evidence_precision_not_load_bearing
  else                        -> mixed_inconclusive
R4 (mechanistic/local success only, no behavioural endpoint) is not a label --
it is the CEILING of this run by construction and is appended to every
evidence_direction_note.

EVIDENCE LEVEL -- THE CEILING, STATED UP FRONT (preregistration section 14)
-----------------------------------------------------------------------------
No default-on consumer of e2.world_forward exists in E3 (necessities register
row B1), so this run CANNOT produce an organism-level endpoint. Its ceiling is
Result 4: mechanistic/local. experiment_purpose is "diagnostic" and
evidence_direction is "non_contributory" on every route. Governance may cite
the mechanistic pattern but must NOT promote MECH-572 or register a new claim
from this run; the organism-level follow-up is a separate chip on B1.

PRE-FREEZE PROBE (seed 42, 180 waking steps, full budget) AND WHAT IT CHANGED
-------------------------------------------------------------------------------
A full-budget channel probe was run before the freeze record. Four findings, and
the three amendments they forced (the preregistration's section 15 explicitly
permits a constant or a gate's scope to change when the non-degeneracy probe
shows a channel degenerate, PROVIDED it is recorded here before any DV is read):

  1. G5 CLEARS at full budget: pi_cur 2758 (converged) vs 171.7 (underfit),
     a 16x separation against the pre-registered 10x. The dry-run's failure was
     pure EMA warm-up, as suspected.
  2. pi_hist_min == 100.0 == 1/v_init in EVERY condition. The estimators start
     each arm cell COLD, because the cached-head pairing carries the P0 WEIGHTS
     but no calibration history -- so the floor of the pi_hist distribution was
     a warm-up artefact posing as "historical precision". FIXED by AMENDMENT 1.
  3. The evidence channel separates cleanly in VARIANCE and not in the packet
     MEAN of precision: sigma_obs 0.005 (the instrument floor) clean vs 0.1205
     under noise; kappa 0.136 clean vs 0.0064 under noise (kappa correctly
     self-measures on the noise); evidence_variance_z mean 3.4e-6 clean vs
     1.1e-4 noisy, a 32x separation. But evidence_precision_z is a 1/x quantity,
     so its packet MEAN (4.8e5 clean vs 3.2e5 noisy) is dominated by the few
     cold-start frames and G3 leg 1 read 0.66 against a 0.25 ceiling. That was a
     STATISTIC failure, not a channel failure. FIXED by AMENDMENT 2.
  4. COND_CONFIDENTLY_WRONG IS DEGENERATE ON THIS HEAD, by construction rather
     than by any channel failing. The converged head's MSE on the INVERTED-rule
     battery is 1.13e-5 against 1.49e-5 on the original rule, and packet pe_mean
     is 2.3e-4 (cond 3) against 3.8e-4 (cond 1) -- the inverted rule is, if
     anything, marginally EASIER. Surprise max 1.02, fraction above 10 = 0. The
     reason is MECH-573 / substrate necessity B5: the head sits near
     copy-the-input (skill -0.071) and barely reads the action, so permuting the
     action map is not a CONTRADICTION for it. G9 and G4's surprise leg
     therefore cannot be met on this substrate at this operating point, and
     failing the whole run on them would report a substrate limit as a
     measurement error. RECLASSIFIED by AMENDMENT 3.

  AMENDMENT 1 -- a CALIBRATION WINDOW, cached with the head and restored into
    every arm. _build_base now runs CALIB_STEPS clean, non-inverted, noise-free
    waking steps (env seed+2, a seed the measurement phase never uses) on a
    separate ARM_B-flagged agent holding the same P0 weights, with NO sleep
    cycle, and caches the two producers' EMA state. _load_base restores it into
    arms B..D. This is PROVENANCE, NOT A LEAK: the window is clean, it is
    strictly pre-test, it uses an env seed no condition uses, it sees no
    manipulation of any kind, and EVERY arm receives the IDENTICAL state -- so
    it cannot differentiate the arms, which is what a leak would have to do.
    ARM_A has no producers and receives nothing. G2's bit-identity comparison is
    unaffected: the calibration agent is a separate object whose RNG draws all
    happen inside _build_base, before every per-arm reset_all_rng.
  AMENDMENT 2 -- G3 reads the packet MEDIAN of evidence_variance_z, not the mean
    of evidence_precision_z. Leg 1 is var_median(cond1)/var_median(cond4) < 0.25
    (equivalently the precision ratio, same threshold); leg 2 is
    |log2(var_median(cond3)/var_median(cond1))| <= 1. THRESHOLDS ARE UNCHANGED;
    only the statistic the same threshold is applied to has changed, and the
    reason is that a mean of a reciprocal is dominated by its smallest
    denominators.
  AMENDMENT 2b (BATCH 2, pre-freeze) -- G3's SECOND leg is rescoped the same
    way. Leg 1 (evidence precision moves with NOISE) stays a verdict-routing
    gate and clears emphatically. Leg 2 (it must NOT move with the RULE SHIFT)
    measured |log2(var_median(cond3)/var_median(cond1))| = 3.795 against a
    threshold of 1, and the entire gap is kappa: SD-PP-1 measures kappa on real
    MOTION, and a permuted action map changes the motion statistics, so the two
    CLEAN conditions differ for a NON-SENSORY reason. That is a scope limit on
    cond-3 contrasts, not a failure of the evidence channel, whose clean-vs-
    noisy separation is ~1000x. So leg 2 becomes G3b, routes_verdict False, and
    when unmet marks P4, P6 and P7 non_degenerate=False and nothing else.
    Registered as item B7. kappa is deliberately NOT frozen: its self-
    measurement under noise in cond 4 is the channel doing its job, and
    freezing it would break leg 1 to rescue leg 2.
  AMENDMENT 2c (BATCH 3 / RED-TEAM F2) -- G3b's threshold widens 1.0 -> 2.0
    (a 2x kappa drift perturbs the Kalman weight K by ~1% at the measured
    operating point, pi_e ~2.9e5 against pi_cur ~2758, so it is not a material
    change to the gain), and G3 gains a DIRECT verdict-routing leg 2: between
    the two CLEAN regimes (cond 1 and cond 3) the SENSORY statistic sigma_obs
    must be IDENTICAL and pinned at sigma_floor. That is the assertion the old
    rule-shift leg was reaching for and could not make, because it read
    evidence_variance_z = kappa * sigma_obs_sq and kappa is a MOTION statistic.
MEASURED TABLES (--probe, seed 42, FULL P0 + FULL calibration, 1 cycle; and
--liveness). These are the section-15 freeze numbers, quoted verbatim.

  base[converged]   conv_rel_drop=0.9980 mse_after_p0=1.48681e-05
                    identity_mse=1.38821e-05 calib v_tot=5.10488e-06
                    kappa=0.000146373 kappa_ready=1
  base[unconverged] conv_rel_drop=0.0000 mse_after_p0=0.00750742
                    identity_mse=1.38821e-05 calib v_tot=0.00793077
                    kappa=0.000493816 kappa_ready=1

  per condition (ARM_B_STORE_ONLY, cycle 1):
  cond                      ev_var_med   sigma_med   kappa_med    pi_hist_sd   surprise_med  surpr>10   pe_mean      pi_cur      mse_pre      skill
  COND_CONVERGED_CLEAN      3.533e-09       0.005   0.0001413     1.836e+05         0.6835  0.005587  4.527e-06    6.22e+05    1.487e-05   -0.07103
  COND_UNDERFIT             1.158e-08       0.005   0.0004632         7.206          1.018         0   0.007613       134.4     0.007507     -539.8
  COND_CONFIDENTLY_WRONG    5.276e-09       0.005   0.0002111     3.651e+04         0.7936  0.005587  1.107e-05    1.84e+05    1.487e-05   -0.07103
  COND_NOISY_CONTRADICTION  1.587e-06      0.1193   0.0001077     2.749e+05         0.4936  0.005587  6.964e-06   7.106e+05    1.487e-05   -0.07103

  gate statistics at full budget:
    G3  leg1 var_med(c1)/var_med(c4) = 0.00222557        (< 0.25)      PASS
    G3  leg2 sigma_obs med c1=0.005 c3=0.005 floor=0.005 (EQUAL, AT floor) PASS
    G3b |log2(var_med(c3)/var_med(c1))| = 0.578785       (<= 2.0)      PASS
    G4  min pi_hist_sd over conditions = 7.2056          (> 0)         PASS
    G4b cond3 early surprise frac = 0.0333333            (>= 0.5)      FAIL
        cond1 surprise median = 0.683481                 (< 3.0)       ok
    G5  pi_cur(c1)/pi_cur(c2) = 4628.09                  (> 10.0)      PASS
    G8  converged-base skill = -0.071027                 (> 0)         FAIL
    G9  cond3 pe_mean / cond1 residual = 0.744699        (> 2.0)       FAIL

  ARM_C_PROVENANCE gain rule, per condition:
  cond                        k_mean      m_mean      r_mean      r_max    gain_min    gain_max  step_scale
  COND_CONVERGED_CLEAN          0.9972     0.02185       1.064      1.975     0.02164      0.4183     0.03845
  COND_UNDERFIT                      1      0.8603       1.014      1.065       1.488        1.91       1.783
  COND_CONFIDENTLY_WRONG        0.9988     0.03068       1.017      1.275     0.04472     0.09879     0.07169
  COND_NOISY_CONTRADICTION      0.4403     0.01488       1.118      1.939        0.02      0.1527     0.02661

  ARM_D_RESIDUAL (current-residual scheduler, BATCH 4), per condition:
  cond                      gain_min    gain_max  step_scale   step_scale_C  dr/C ratio
  COND_CONVERGED_CLEAN       0.02405      0.2037     0.03322        0.03845      0.8641
  COND_UNDERFIT                1.319       1.419        1.42          1.783      0.7962
  COND_CONFIDENTLY_WRONG     0.07196      0.1101     0.07665        0.07169       1.069
  COND_NOISY_CONTRADICTION   0.03644      0.1953     0.04496        0.02661        1.69
  P5 DISCRIMINATION CHECK (cond 4): Dr 0.04496 vs C 0.02661 -> Dr runs 1.69x
  HOTTER than C on the noisy channel. P5 DISCRIMINATES.

  attainability ladder (--liveness, converged base, ARM_D_GLOBAL flags):
  global_scale | realised_step_scale | world_head_max_abs_delta | retention_r0
          0.02 |                0.02 |              0.000158912 |     0.077999
          0.05 |                0.05 |              0.000398606 |     0.471372
           0.1 |                 0.1 |              0.000801712 |       1.8015
           0.5 |                 0.5 |               0.00401959 |      22.0688
             1 |                   1 |               0.00804061 |      19.8443
             2 |                   2 |                0.0160124 |      18.8331
  strictly_increasing=True (5/5 rungs increased)

  FIVE READINGS THESE TABLES PIN, and they are why the amendments above exist.
  (i)   The RULE SEPARATES THE FOUR REGIMES ~67x in realised plasticity: 1.783
        on the underfit base (learn hard, m = 0.860) against 0.0266 under noise
        and 0.0385 on the converged-clean base (protect, m = 0.0219 at K ~ 1).
        Under noise K_mean falls to 0.4403 -- the Kalman term discounting
        unreliable evidence, which is the cleanest single demonstration that
        evidence precision is LOAD-BEARING IN THE RULE and not merely recorded.
  (ii)  ret < 0.10 needs a gain budget below ~0.022, against ARM_C's realised
        cond-1 gain of 0.0385 and c_seed ~0.48. The preregistration's absolute
        ceiling was therefore UNATTAINABLE as a leg of P2 (AMENDMENT 1b).
  (iii) RETENTION IS NON-MONOTONE IN GAIN above scale ~0.5 (22.07 at 0.5, then
        19.84 at 1.0 and 18.83 at 2.0) while DISPLACEMENT keeps rising
        strictly. "More gain is monotonically worse for retention" is FALSE in
        that range, which matters for reading any P6 split whose budget ratio
        lands above 0.5.
  (iv)  G3b reads 0.578785 at FULL calibration against 3.795 at the 30-step
        dry-run window: the kappa-aliasing concern (B7) is largely an artefact
        of an un-warmed kappa, and G5 separates 4628x rather than the 16x
        measured without a calibration window at all.
  (vi)  ARM_D_RESIDUAL BEHAVES AS F2 REQUIRES, and P5 therefore has a real
        contrast. Its gain tracks the CURRENT residual with no evidence term,
        so where the observation noise INFLATES that residual (cond 4) it runs
        1.69x HOTTER than C -- precisely because C's Kalman weight drops to
        0.4403 there and Dr has no such term. In the three conditions where the
        residual is honest it sits at 0.86x / 0.80x / 1.07x of C. So a P5 win
        for C cannot be explained by C merely running colder everywhere: on the
        noisy channel Dr is the hotter arm, which is the only configuration in
        which "C ignores the noise" is a testable claim rather than a restated
        budget difference.
  (v)   c_seed ~0.48 is dominated by COND_UNDERFIT's 1.783, so ARM_D_GLOBAL
        runs at ~12x ARM_C's gain in cond 1 and ~7x in cond 3 -- exactly where
        P6 is taken. That is the matched-in-TOTAL design working as specified,
        NOT a defect, but P6 MUST be read against p6_budget_ratio_<cond>_s<seed>
        and the freeze record says so.

  AMENDMENT 1b (BATCH 3 / RED-TEAM F1) -- P2's ABSOLUTE leg is demoted to a
    REPORT. V3-EXQ-1063's LANDED seed-42 numbers (battery pre 1.4868e-5,
    3-cycle mse_delta_sum -7.142e-5) put ret(B) near 4.8, not the ~1.0 the
    preregistration's "rise at most 10% of residual" ceiling assumed; at the
    realised cond-1 gain (~0.39) that ceiling is UNATTAINABLE, so keeping it as
    a leg of P2 would have failed the load-bearing contrast for a SCALING
    reason and sent the matrix to R2/mixed. P2 is now the RELATIVE leg alone
    (ret(C) < ret(B) - margin, protection against the baseline); the absolute
    reading survives as P2b, which routes nothing and feeds no label. NO GAIN
    CONSTANT WAS CHANGED. Instead the liveness ladder extends DOWN to 0.02 and
    records retention per rung, so the manifest carries the measured
    ATTAINABILITY CURVE -- what retention each budget actually buys -- rather
    than an assumption about it.
  AMENDMENT 3 -- the by-construction failures become NON-DEGENERACY MARKERS
    rather than FAIL routes. G4 keeps ONLY its "pi_hist varies" leg as a
    verdict-routing gate. Its surprise leg becomes G4b (routes_verdict False),
    which together with G9 marks COND_CONFIDENTLY_WRONG unreadable and flags
    P4, P7 and P6's correction leg non_degenerate=False. Cond-3 cells are still
    RUN and fully RECORDED across all three seeds -- that record IS the
    instrument measurement that B5 is real, and it is the run's contribution on
    that point even when the contrasts built on it cannot be read.

RED-TEAM (Step 4.5): pending -- to be filled by the session

Run with:
  /opt/local/bin/python3 experiments/v3_exq_1073_mech572_precision_provenance_gain.py --dry-run
  /opt/local/bin/python3 experiments/v3_exq_1073_mech572_precision_provenance_gain.py --liveness
  /opt/local/bin/python3 experiments/v3_exq_1073_mech572_precision_provenance_gain.py --probe
  /opt/local/bin/python3 experiments/v3_exq_1073_mech572_precision_provenance_gain.py
"""
from __future__ import annotations

import argparse
import hashlib
import math
import random
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig
from experiment_protocol import emit_outcome
from experiments._lib.arm_fingerprint import arm_cell, reset_all_rng
from experiments._lib.precondition_gate import (
    PreconditionSpec,
    aggregate_arm_gates,
    assert_no_structurally_unsatisfiable_gate,
    evaluate_arm_gate,
)
from experiments.pack_writer import write_flat_manifest

EXPERIMENT_TYPE = "v3_exq_1073_mech572_precision_provenance_gain"
QUEUE_ID = "V3-EXQ-1073"
CLAIM_IDS: List[str] = ["MECH-572"]
EXPERIMENT_PURPOSE = "diagnostic"

# ---------------------------------------------------------------------------
# arms (preregistration section 6)
# ---------------------------------------------------------------------------
ARM_A_BASELINE = "ARM_A_BASELINE"
ARM_B_STORE_ONLY = "ARM_B_STORE_ONLY"
ARM_C_PROVENANCE = "ARM_C_PROVENANCE"
ARM_C_NOHIST = "ARM_C_NOHIST"
ARM_D_GLOBAL = "ARM_D_GLOBAL"
ARM_D_RESIDUAL = "ARM_D_RESIDUAL"

# Run order MATTERS for ARM_D_GLOBAL ONLY: c_seed is ARM C's realised mean
# gain, so the arm that consumes it as a matched budget must run after C for
# that seed. BATCH 4: ARM_D_RESIDUAL no longer needs it -- the substrate's
# "residual_only" mode now schedules from the CURRENT per-row residual and
# ignores global_scale entirely -- so it moves into phase 1 and its budget is
# its own, reallocated across conditions by the residual alone.
ARMS_PHASE1: Tuple[str, ...] = (
    ARM_A_BASELINE, ARM_B_STORE_ONLY, ARM_C_PROVENANCE, ARM_C_NOHIST,
    ARM_D_RESIDUAL)
ARMS_PHASE2: Tuple[str, ...] = (ARM_D_GLOBAL,)
ARMS: Tuple[str, ...] = ARMS_PHASE1 + ARMS_PHASE2

# record_metadata -> the SD-PP-1/2/3 producers + recorder are constructed.
# gain_mode None -> use_provenance_conditioned_consolidation_gain stays False,
# so provenance_gain_config is None and compute_e2_world_loss takes the plain
# (pre-SD-PP-4) branch -- structural absence, not a neutral weight.
ARM_SPEC: Dict[str, Dict[str, Any]] = {
    ARM_A_BASELINE: {"record_metadata": False, "gain_mode": None,
                     "uses_c_seed": False,
                     "purpose": "current behaviour; reproduces the MECH-572 phenotype"},
    ARM_B_STORE_ONLY: {"record_metadata": True, "gain_mode": None,
                       "uses_c_seed": False,
                       "purpose": "storage neutrality -- recording must not itself change anything"},
    ARM_C_PROVENANCE: {"record_metadata": True, "gain_mode": "provenance",
                       "uses_c_seed": False,
                       "purpose": "treatment"},
    ARM_C_NOHIST: {"record_metadata": True, "gain_mode": "provenance_nohist",
                   "uses_c_seed": False,
                   "purpose": "is historical precision load-bearing (intake F1)?"},
    ARM_D_GLOBAL: {"record_metadata": True, "gain_mode": "global",
                   "uses_c_seed": True,
                   "purpose": "matched-budget rival: any uniform lr reduction"},
    ARM_D_RESIDUAL: {"record_metadata": True, "gain_mode": "residual_only",
                     "uses_c_seed": False,
                     "purpose": ("rival scheduler: CURRENT per-row residual only "
                                 "-- g_i = clip(gain_max * sqrt(pe_cur_i / "
                                 "v_ref), gain_min, gain_max), i.e. C's "
                                 "magnitude factor m_i with K=1, r=1 and the "
                                 "CURRENT residual in place of the stored "
                                 "innovation. Ignores global_scale (intake F2)")},
}
GAIN_ARMS: Tuple[str, ...] = (
    ARM_C_PROVENANCE, ARM_C_NOHIST, ARM_D_GLOBAL, ARM_D_RESIDUAL)
RECORDER_ARMS: Tuple[str, ...] = (ARM_B_STORE_ONLY,) + GAIN_ARMS

# ---------------------------------------------------------------------------
# conditions (preregistration section 7)
# ---------------------------------------------------------------------------
COND_CONVERGED_CLEAN = "COND_CONVERGED_CLEAN"
COND_UNDERFIT = "COND_UNDERFIT"
COND_CONFIDENTLY_WRONG = "COND_CONFIDENTLY_WRONG"
COND_NOISY_CONTRADICTION = "COND_NOISY_CONTRADICTION"
CONDITIONS: Tuple[str, ...] = (
    COND_CONVERGED_CLEAN, COND_UNDERFIT,
    COND_CONFIDENTLY_WRONG, COND_NOISY_CONTRADICTION)

# sigma 0.12 is V3-EXQ-798a's MEL-match to its HIGH shift arm, carried by 1071.
OBS_NOISE_SIGMA = 0.12

COND_SPEC: Dict[str, Dict[str, Any]] = {
    COND_CONVERGED_CLEAN: {
        "base": "converged", "invert_action_map": False, "obs_noise": 0.0,
        "correction_battery": False,
        "regime": "hist HIGH, ev HIGH, cur HIGH -- avoid destructive displacement"},
    COND_UNDERFIT: {
        "base": "unconverged", "invert_action_map": False, "obs_noise": 0.0,
        "correction_battery": False,
        "regime": "hist LOW, ev HIGH, cur LOW -- keep learning"},
    COND_CONFIDENTLY_WRONG: {
        "base": "converged", "invert_action_map": True, "obs_noise": 0.0,
        "correction_battery": True,
        "regime": "hist HIGH, ev HIGH, cur DROPS -- must reopen"},
    COND_NOISY_CONTRADICTION: {
        "base": "converged", "invert_action_map": False,
        "obs_noise": OBS_NOISE_SIGMA, "correction_battery": False,
        "regime": "hist HIGH, ev LOW, cur epi HIGH -- must not learn the noise"},
}

# Seed 44 is deliberately absent (recurring reef-config early-death instability,
# EXQ-539/540, V3-EXQ-538a). V3-EXQ-1063's set, fixed here by preregistration
# section 10.
SEEDS: Tuple[int, ...] = (42, 123, 456)

# ---------------------------------------------------------------------------
# substrate constants, carried verbatim from V3-EXQ-1063
# ---------------------------------------------------------------------------
GRID_SIZE = 5
N_HAZARDS = 1
N_RESOURCES = 1
SELF_DIM = 16
WORLD_DIM = 16
STEPS_PER_EPISODE = 90

P0_STEPS = 3600
P0_EPISODE_EQUIV = P0_STEPS // STEPS_PER_EPISODE          # 40
P0_TRAJECTORY_AT = (0, 30, 90, 180, 360, 720, 1440, 2400, 3600)
E2_LR = 1e-3
BATCH_K = 8
BUF_MAX = 256
MIN_BUF_BEFORE_TRAIN = 16
MAX_GRAD_NORM = 1.0

N_CYCLES = 2
WAKE_EPS_PER_CYCLE = 2
EPISODES_PER_RUN = P0_EPISODE_EQUIV + N_CYCLES * WAKE_EPS_PER_CYCLE   # 44

CMC_STEPS = 8
CMC_LR = 1e-3
CMC_BATCH = 16

BATTERY_SIZE = 64
# AMENDMENT 1: the pre-test calibration window. 180 = the full-budget probe's
# waking length, which is what the probe measured the producers' EMAs to need.
CALIB_STEPS = 180
BATTERY_FLOOR = 32.0
WORLD_BUFFER_FLOOR = float(CMC_BATCH + 1)

# ---------------------------------------------------------------------------
# gain-rule constants (preregistration section 5). PRE-REGISTERED -- never
# derived from a run statistic. They are the ree_core defaults; they are named
# here so the manifest carries them and a later reader can see what ran.
# ---------------------------------------------------------------------------
GAIN_NOISE_GAIN = 2.0
GAIN_V_REF = 1e-2
GAIN_SURPRISE_BETA = 0.5
GAIN_REOPEN_MAX = 3.0
GAIN_MIN = 0.02
GAIN_MAX = 2.0
OBS_EMA_ALPHA = 0.2
KAPPA_EMA_ALPHA = 0.05
SIGMA_FLOOR = 0.005
PE_EMA_ALPHA = 0.05
V_FLOOR = 1e-6
V_INIT = 1e-2
PRECISION_SOURCE = "sd063_or_ema"

# ---------------------------------------------------------------------------
# pre-registered thresholds (constants; NEVER derived from run statistics)
# ---------------------------------------------------------------------------
MIN_CONV_REL_DROP = 0.90        # a CONVERGED base must have converged
MAX_UNCONV_REL_DROP = 0.02      # an UNDERFIT base must not have
SEEDS_REQUIRED = 2              # the 2-of-3 pre-registered sign count
MARGIN = 0.02                   # absolute floor on the relative quantities
RET_C_CEILING = 0.10            # P2: post-sleep error rise <= 10% of residual
T_CRIT_95_DF2 = 4.302653        # two-sided t, 95%, n=3 -- DESCRIPTIVE only

# G1 liveness ladder (SD-PP-4 test 10; preregistration section 9).
# RED-TEAM F1: extended DOWNWARD. The two smallest rungs exist to measure the
# ATTAINABILITY of a given retention, not just the monotonicity of displacement.
LIVENESS_SCALES: Tuple[float, ...] = (0.02, 0.05, 0.1, 0.5, 1.0, 2.0)
# G3
G3_NOISE_RATIO_MAX = 0.25       # cond4 evidence precision < 0.25 x cond1
# RED-TEAM F2: widened 1.0 -> 2.0. kappa moving 2x perturbs the Kalman weight
# K by ~1% at the measured operating point (pi_e ~2.9e5 against pi_cur ~2758),
# so a 2x kappa drift is not a material change to the gain. G3b routes no
# verdict in any case (BATCH 2 EDIT 1).
G3_SHIFT_LOG2_MAX = 2.0         # |log2(cond3/cond1)| <= 2
# G4
G4_SURPRISE_HIGH = 10.0
G4_EARLY_PACKETS = 30
G4_EARLY_FRAC_FLOOR = 0.5
G4_CLEAN_MEDIAN_MAX = 3.0
# G5
G5_PI_CUR_RATIO = 10.0
# G7
MSE_FLOOR = 1e-8
# G9
G9_PE_RATIO = 2.0

EPS = 1e-12
WORLD_HEAD_MODULES = ("world_transition", "world_action_encoder")

# The SD-056 rollout clamp is deliberately NOT enabled, for V3-EXQ-1063's three
# reasons, unchanged: (1) REACH -- this driver runs no imagination rollout at
# all; world_forward_contrastive_loss is called as a read-only no_grad readout
# on a fixed battery, and the only contrastive TRAINING is the sleep pass's 8
# steps at lr 1e-3 scoped to the two world heads. (2) COMPARABILITY -- these
# readouts extend V3-EXQ-1063's, which declined the same clamp. (3) SCOPE -- an
# unratified substrate-behaviour change inside a diagnostic is exactly the drift
# that makes a later reader unable to attribute a difference.
SD056_ROLLOUT_CLAMP_EXEMPT = (
    "No imagination rollout is run: world_forward_contrastive_loss is a "
    "read-only no_grad readout on fixed batteries, and the only contrastive "
    "training is the sleep pass's 8 steps at lr 1e-3 scoped to the two world "
    "heads. Divergence is DETECTED (all e1/e2 params asserted finite after "
    "every cycle) rather than assumed absent. Enabling the clamp would break "
    "comparability with V3-EXQ-1063, whose readouts this run extends."
)

ANCHOR_REACHABILITY_EXEMPT = (
    "world_experience_buffer floor (17) is >20x cleared by the fixed P1 "
    "rollout (N_CYCLES 2 x WAKE_EPS 2 x STEPS 90 = 360 entries/cell at "
    "e1_steps_per_tick=1); not a hand-tuned degeneracy definition."
)

ETHICS_PREFLIGHT = {
    "involves_negative_valence": False,
    "involves_suffering_like_state": False,
    "involves_self_model": False,
    "involves_inescapability_or_helplessness": False,
    "involves_offline_replay_over_harm": False,
    "involves_social_mind_or_language": False,
    "involves_human_data_or_clinical_context": False,
    "decision": "allow",
}


# ---------------------------------------------------------------------------
# preconditions -- regime-conditioned per CELL (experiments/_lib/precondition_gate)
# No cell's gate may vacate another cell's.
# ---------------------------------------------------------------------------
def _ctx_is_converged(ctx: Dict[str, Any]) -> bool:
    return ctx["base"] == "converged"


def _ctx_is_unconverged(ctx: Dict[str, Any]) -> bool:
    return ctx["base"] == "unconverged"


def _ctx_has_correction_battery(ctx: Dict[str, Any]) -> bool:
    return bool(ctx["correction_battery"])


def _ctx_has_recorder(ctx: Dict[str, Any]) -> bool:
    return bool(ctx["record_metadata"])


def _ctx_has_gain(ctx: Dict[str, Any]) -> bool:
    return ctx["gain_mode"] is not None


PRECONDITIONS: Tuple[PreconditionSpec, ...] = (
    PreconditionSpec(
        name="world_replay_pairs_populated",
        description="replay pairs available to the world trainer at cycle time",
        control="real P1 waking rollout via _e1_tick, not synthetic buffers",
        threshold=WORLD_BUFFER_FLOOR, direction="lower"),
    PreconditionSpec(
        name="world_loss_non_sentinel",
        description=("compute_e2_world_loss returns an exactly-zero graph-anchored "
                     "sentinel when n_pairs < 2, and consolidate()'s contract says "
                     "such a step does NOT count as touching the module"),
        control="probed on the same real buffers the sleep cycle draws from",
        threshold=0.0, direction="lower"),
    PreconditionSpec(
        name="world_grad_nonzero",
        description="a non-zero gradient reaches the two world heads",
        control=("backward() from the realised world loss; run in EVERY arm so "
                 "the arms stay RNG-matched entering the cycle"),
        threshold=0.0, direction="lower"),
    PreconditionSpec(
        name="action_buffer_non_vacuous",
        description=("fraction of replay action entries that are not all-zero; "
                     "world_action_encoder is nn.Linear so a zero input gives "
                     "dL/dW = 0 EXACTLY and the channel is untrainable"),
        control=("agent.e1_action_buffer_stats(), the substrate's own "
                 "non-vacuity detector for this failure mode"),
        threshold=0.5, direction="lower"),
    PreconditionSpec(
        name="retention_battery_populated",
        description="held-out one-step transitions backing the retention MSE",
        control="held-out env (seed+9973) + fixed action policy, pure read",
        threshold=BATTERY_FLOOR, direction="lower",
        structural_max=lambda ctx: float(BATTERY_SIZE)),
    PreconditionSpec(
        name="correction_battery_populated",
        description=("held-out one-step transitions under the SHIFTED rule, "
                     "backing the correction DV"),
        control=("a SECOND held-out env instance (seed+9973) with the same "
                 "inversion applied BEFORE sampling"),
        threshold=BATTERY_FLOOR, direction="lower",
        applies_to=_ctx_has_correction_battery,
        applies_note=("only COND_CONFIDENTLY_WRONG reads a correction battery; "
                      "asserting it elsewhere would be structurally un-passable"),
        structural_max=lambda ctx: float(BATTERY_SIZE)),
    PreconditionSpec(
        name="sleep_cycles_fired",
        description=("cycles whose merged metrics carry post_sleep_z_goal_retention, "
                     "which _run_cycle merges unconditionally at the end of every "
                     "completed cycle"),
        control="a key no internal gate can suppress on a completed cycle",
        threshold=0.5, direction="lower",
        structural_max=lambda ctx: float(ctx["n_cycles"])),
    PreconditionSpec(
        name="base_converged",
        description=("frozen-battery MSE drop across P0 on the SAME retention "
                     "battery the DV is later read on"),
        control=("recon-only Adam on agent.e2.parameters() over buffered one-step "
                 "transitions from the no-shift env -- V3-EXQ-798a's P0 form"),
        threshold=MIN_CONV_REL_DROP, direction="lower",
        applies_to=_ctx_is_converged,
        applies_note=("an UNDERFIT base is DEFINED by running no optimiser, so "
                      "asserting convergence there would be structurally "
                      "un-passable and would collapse the two-base design")),
    PreconditionSpec(
        name="base_unconverged",
        description="the same drop, bounded ABOVE, for the base that must not train",
        control=("identical P0 rollout with the optimiser withheld; the bound is "
                 "a positive check that the regime label is true"),
        threshold=MAX_UNCONV_REL_DROP, direction="upper",
        applies_to=_ctx_is_unconverged,
        applies_note="a CONVERGED base is required to exceed this bound, not stay under it",
        # A CEILING's satisfiability bound is the MINIMUM attainable value. With
        # the optimiser withheld the battery tensors are fixed and world_forward
        # is untouched, so the drop is EXACTLY 0.
        structural_min=lambda ctx: 0.0),
    PreconditionSpec(
        name="provenance_packets_recorded",
        description=("FRACTION of world-buffer entries carrying a provenance "
                     "packet at sleep entry. SD-PP-3 appends one packet per "
                     "buffer entry and trims in lockstep, so this is 1.0 BY "
                     "CONSTRUCTION -- it is an ALIGNMENT assertion, and a value "
                     "below 1 means the carrier has drifted off the buffer it "
                     "indexes (RED-TEAM F3: the old 'count > 0' form was "
                     "vacuous)"),
        control=("SD-PP-3 recorder driven by the real _e1_tick path, compared "
                 "against len(agent._world_experience_buffer)"),
        threshold=0.99, direction="lower",
        applies_to=_ctx_has_recorder,
        applies_note=("ARM_A constructs no recorder at all (structural absence), "
                      "so asserting packets there would be un-passable")),
    PreconditionSpec(
        name="gain_rows_with_provenance",
        description=("fraction of the replay rows the gain rule scored that "
                     "carried a usable packet (1 - n_missing/n_rows). The ONLY "
                     "legitimate misses are episode-start placeholders, a "
                     "handful out of ~360 buffer entries, so a floor of 0.9 is "
                     "reachable by construction while still catching a gain "
                     "running mostly over missing rows -- which would be inert "
                     "at 1.0 (RED-TEAM F3: the old '> 0' form was vacuous)"),
        control="SD-PP-4's own n_missing diagnostic, read from the last cycle",
        threshold=0.9, direction="lower",
        applies_to=_ctx_has_gain,
        applies_note=("ARM_A / ARM_B build no ProvenanceGainConfig, so no rows "
                      "are ever scored there")),
)


# ---------------------------------------------------------------------------
# small helpers (V3-EXQ-1063 forms)
# ---------------------------------------------------------------------------
def _finite_or_none(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _finite_vals(xs: Sequence[Any]) -> List[float]:
    out: List[float] = []
    for x in xs:
        v = _finite_or_none(x)
        if v is not None:
            out.append(v)
    return out


def _mean(xs: Sequence[Any]) -> float:
    vals = _finite_vals(xs)
    return sum(vals) / len(vals) if vals else float("nan")


def _sd(xs: Sequence[Any]) -> float:
    vals = _finite_vals(xs)
    if len(vals) < 2:
        return float("nan")
    m = sum(vals) / len(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))


def _median(xs: Sequence[Any]) -> float:
    vals = sorted(_finite_vals(xs))
    if not vals:
        return float("nan")
    n = len(vals)
    mid = n // 2
    return vals[mid] if n % 2 else 0.5 * (vals[mid - 1] + vals[mid])


def _ci95(xs: Sequence[Any]) -> Tuple[float, float]:
    """t-based 95% CI at n=3. DESCRIPTIVE ONLY -- flagged low_power everywhere."""
    vals = _finite_vals(xs)
    if len(vals) < 2:
        return (float("nan"), float("nan"))
    m = sum(vals) / len(vals)
    s = _sd(vals)
    half = T_CRIT_95_DF2 * s / math.sqrt(len(vals))
    return (m - half, m + half)


def _to_batched(x: Any, device: Any) -> torch.Tensor:
    if not torch.is_tensor(x):
        x = torch.tensor(x, dtype=torch.float32, device=device)
    else:
        x = x.to(device)
    return x.unsqueeze(0) if x.dim() == 1 else x


def _hash_tensor_list(tensors: Sequence[torch.Tensor]) -> str:
    """sha256 over the raw bytes of a tensor list. Empty list -> a fixed tag."""
    if not tensors:
        return "empty"
    h = hashlib.sha256()
    for t in tensors:
        h.update(t.detach().cpu().contiguous().to(torch.float32).numpy().tobytes())
    return h.hexdigest()


def _hash_world_buffer(agent: REEAgent) -> str:
    buf = list(getattr(agent, "_world_experience_buffer", []))
    if not buf:
        return "empty"
    flat = torch.cat([b.detach().reshape(1, -1) for b in buf], dim=0)
    return hashlib.sha256(
        flat.cpu().contiguous().to(torch.float32).numpy().tobytes()).hexdigest()


def _hash_rng_state() -> str:
    """The global torch RNG state. compute_e2_world_loss's randperm draw is a
    PURE FUNCTION of this, so equality here IS the replay-batch equality G2
    asserts (preregistration section 9 / red-team 9)."""
    return hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()


def _make_env(seed: int, invert_action_map: bool = False) -> CausalGridWorldV2:
    """No world_rule_shift schedule in ANY arm. The COND_CONFIDENTLY_WRONG
    inversion is a ONE-SHOT driver poke of the effective action map (substrate
    necessity (d)), applied at construction and never re-permuted."""
    env = CausalGridWorldV2(
        seed=seed, size=GRID_SIZE, num_hazards=N_HAZARDS,
        num_resources=N_RESOURCES, use_proxy_fields=True)
    if invert_action_map:
        _invert_action_map(env)
    return env


def _invert_action_map(env: CausalGridWorldV2) -> None:
    """Rule R1 = R0 with the action map inverted: swap 0<->1 and 2<->3.

    The canonical map is {0:(-1,0), 1:(1,0), 2:(0,-1), 3:(0,1), 4:(0,0)}
    (causal_grid_world.py ACTIONS, copied into the effective _action_map at
    construction), so this swaps up<->down and left<->right and leaves stay
    (and any appended consummatory action, which is not a map member) alone.
    RELIABLE, not noisy: the new rule is deterministic and fully learnable.
    """
    am = env._action_map
    env._action_map = {
        0: am[1], 1: am[0], 2: am[3], 3: am[2],
        **{k: v for k, v in am.items() if k > 3},
    }


def _make_agent(env: CausalGridWorldV2, arm: str,
                global_scale: float = 1.0) -> REEAgent:
    """The V3-EXQ-1063 base configuration, plus the SD-PP-1..4 flags for the arm.

    EVERY arm runs the full MECH-423 consolidation pass with
    use_sleep_world_forward_consolidation=True -- the manipulation is the GAIN,
    never whether the module is consolidated at all.
    """
    spec = ARM_SPEC[arm]
    record = bool(spec["record_metadata"])
    gain_mode = spec["gain_mode"]
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=SELF_DIM,
        world_dim=WORLD_DIM,
        sws_enabled=True,
        rem_enabled=True,
        use_sleep_loop=True,
        # A huge K keeps notify_episode_end's automatic cadence from ever firing,
        # so the N_CYCLES deliberate force_cycle() calls are the only sleep cycles.
        sleep_loop_episodes_K=1_000_000,
        use_sleep_aggregation_cluster=True,
        # IDENTICAL IN EVERY ARM.
        use_cross_module_consolidation=True,
        cross_module_consolidation_schedule="interleaved",
        cross_module_consolidation_steps=CMC_STEPS,
        cross_module_consolidation_lr=CMC_LR,
        cross_module_consolidation_batch=CMC_BATCH,
        use_sleep_world_forward_consolidation=True,
        cross_module_consolidation_record_trace=True,
        # SD-PP-1/2/3: the producers and the carrier.
        use_observation_reliability=record,
        observation_reliability_obs_ema_alpha=OBS_EMA_ALPHA,
        observation_reliability_kappa_ema_alpha=KAPPA_EMA_ALPHA,
        observation_reliability_sigma_floor=SIGMA_FLOOR,
        use_world_forward_epistemic_precision=record,
        world_forward_precision_source=PRECISION_SOURCE,
        world_forward_precision_pe_ema_alpha=PE_EMA_ALPHA,
        world_forward_precision_v_floor=V_FLOOR,
        world_forward_precision_noise_gain=GAIN_NOISE_GAIN,
        use_replay_precision_provenance=record,
        # SD-PP-4: the consumer.
        use_provenance_conditioned_consolidation_gain=gain_mode is not None,
        provenance_gain_mode=(gain_mode or "provenance"),
        provenance_gain_min=GAIN_MIN,
        provenance_gain_max=GAIN_MAX,
        provenance_gain_surprise_beta=GAIN_SURPRISE_BETA,
        provenance_gain_reopen_max=GAIN_REOPEN_MAX,
        provenance_gain_v_ref=GAIN_V_REF,
        provenance_gain_global_scale=float(global_scale),
        # MECH-205 instrument, live in every arm so waking is one configuration.
        surprise_gated_replay=True,
        pe_ema_alpha=0.02,
        # The MEL CONSUMER stays absent, so the offline budget is
        # scheduler-pinned and identical across arms.
        use_mel_consumer=False,
        use_entry_pressure=False,
        use_within_life_sleep_trigger=False,
    )
    return REEAgent(cfg)


def _world_head_params(agent: REEAgent) -> List[torch.Tensor]:
    out: List[torch.Tensor] = []
    for name in WORLD_HEAD_MODULES:
        mod = getattr(agent.e2, name, None)
        if mod is not None:
            out.extend(mod.parameters())
    return out


def _snapshot(params: Sequence[torch.Tensor]) -> List[torch.Tensor]:
    return [p.detach().clone() for p in params]


def _max_abs_delta(before: Sequence[torch.Tensor],
                   after: Sequence[torch.Tensor]) -> float:
    worst = 0.0
    for b, a in zip(before, after):
        if b.numel() == 0:
            continue
        worst = max(worst, float((a - b).abs().max().item()))
    return worst


def _all_finite(params: Sequence[torch.Tensor]) -> bool:
    return all(bool(torch.isfinite(p).all().item()) for p in params)


def _obs(d: Dict[str, Any], key: str) -> Optional[torch.Tensor]:
    v = d.get(key)
    if v is None:
        return None
    return v if torch.is_tensor(v) else torch.as_tensor(v, dtype=torch.float32)


def _apply_obs_noise(obs_dict: Dict[str, Any], sigma: float,
                     gen: Optional[torch.Generator]) -> Dict[str, Any]:
    """V3-EXQ-798a :565-577 / V3-EXQ-1071 :387-399 verbatim. Additive Gaussian
    noise on the exteroceptive channel, deliberately UN-LEARNABLE -- there is no
    structure here to re-learn, so a model cannot reduce the PE it induces no
    matter how long it trains. That is exactly what makes COND_NOISY_CONTRADICTION
    the "large but UNRELIABLE" partner to COND_CONFIDENTLY_WRONG's "large but
    reliable". Drawn from a DEDICATED generator, so it consumes no global RNG and
    the arms stay bit-identical."""
    if sigma <= 0.0:
        return obs_dict
    out = dict(obs_dict)
    ws = _obs(obs_dict, "world_state")
    if ws is not None:
        out["world_state"] = ws + torch.randn(
            ws.shape, generator=gen, dtype=ws.dtype) * sigma
    return out


def _sense(agent: REEAgent, obs_dict: Dict[str, Any]):
    device = agent.device
    obs_harm = obs_dict.get("harm_obs", None)
    return agent.sense(
        _to_batched(obs_dict["body_state"], device),
        _to_batched(obs_dict["world_state"], device),
        obs_harm=_to_batched(obs_harm, device) if obs_harm is not None else None,
    )


# ---------------------------------------------------------------------------
# the frozen held-out batteries and their readouts
# ---------------------------------------------------------------------------
def _sample_probe_battery(agent: REEAgent, seed: int, n_transitions: int,
                          invert_action_map: bool = False
                          ) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """V3-EXQ-701b / 1060 / 1063 form. HELD OUT: a distinct env instance and a
    FIXED action policy independent of training. PURE READ: senses and steps
    only; it never calls _e1_tick, so it appends nothing to the replay buffers
    the trainer draws from.

    invert_action_map builds the R1 (correction) battery: the SAME held-out
    seed, with the rule inversion applied BEFORE any transition is sampled, so
    R0 and R1 differ by the action map and by nothing else.
    """
    env = _make_env(seed + 9973, invert_action_map=invert_action_map)
    _, obs_dict = env.reset()
    act_rng = random.Random(seed + 9973)
    battery: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    prev: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    guard = 0
    max_guard = max(n_transitions, 1) * 8
    with torch.no_grad():
        while len(battery) < n_transitions and guard < max_guard:
            guard += 1
            z_now = _sense(agent, obs_dict).z_world.detach().reshape(1, -1).clone()
            if not bool(torch.isfinite(z_now).all().item()):
                break
            if prev is not None:
                battery.append((prev[0], prev[1], z_now))
            idx = act_rng.randrange(env.action_dim)
            action = torch.zeros(1, env.action_dim, device=agent.device)
            action[0, idx] = 1.0
            _, _, done, _, obs_dict = env.step(action)
            prev = (z_now, action)
            if done:
                _, obs_dict = env.reset()
                prev = None
    return battery


def _battery_tensors(battery: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
                     device: Any
                     ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    if len(battery) < 2:
        return None
    z0 = torch.cat([b[0] for b in battery], dim=0).to(device)
    acts = torch.cat([b[1] for b in battery], dim=0).to(device)
    z1 = torch.cat([b[2] for b in battery], dim=0).to(device)
    return z0, acts, z1


def _identity_predictor_mse(
        tensors: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
) -> float:
    """mean((z1 - z0)^2): what a predictor that simply COPIES its input scores.
    MECH-573 measures the converged head to sit near copy-the-input, so this is
    the reference skill = 1 - mse/identity_mse is taken against (G8)."""
    if tensors is None:
        return float("nan")
    z0, _acts, z1 = tensors
    with torch.no_grad():
        return float((z1 - z0).pow(2).mean().item())


def _battery_readouts(agent: REEAgent,
                      tensors: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
                      ) -> Dict[str, float]:
    """mse -- per-element reconstruction error (701b _frozen_probe_pe form).
    infonce -- the SD-056 objective compute_e2_world_loss actually minimises,
    called exactly as the trainer's call site does (min_batch_classes=1)."""
    out = {"mse": float("nan"), "infonce": float("nan")}
    if tensors is None:
        return out
    z0, acts, z1 = tensors
    with torch.no_grad():
        pred = agent.e2.world_forward(z0, acts)
        out["mse"] = float((pred - z1).pow(2).mean().item())
        try:
            loss = agent.e2.world_forward_contrastive_loss(
                z_world_0=z0, actions=acts, z_world_1_targets=z1,
                min_batch_classes=1, simulation_mode=False,
            )
            out["infonce"] = (float(loss.detach().item())
                              if torch.is_tensor(loss) else float("nan"))
        except (RuntimeError, ValueError):
            out["infonce"] = float("nan")
    return out


# ---------------------------------------------------------------------------
# P0 -- the base. Identical rollout in both regimes; the optimiser is the ONLY
# difference (V3-EXQ-1063's FRESH regime, cited by preregistration section 7).
# ---------------------------------------------------------------------------
def _p0_train_step(agent: REEAgent, buf: Deque, opt: torch.optim.Optimizer,
                   rng: random.Random) -> Optional[float]:
    """RECON-ONLY reconstruction MSE on buffered one-step transitions. The SD-056
    contrastive auxiliary is omitted -- a CONFIRMED P0 destabiliser (V3-EXQ-701b
    ablation, carried by 798a)."""
    if len(buf) < MIN_BUF_BEFORE_TRAIN:
        return None
    pool = list(buf)
    batch = pool if len(pool) <= BATCH_K else rng.sample(pool, BATCH_K)
    z0 = torch.stack([t[0] for t in batch]).to(agent.device)
    acts = torch.stack([t[1] for t in batch]).to(agent.device)
    z1 = torch.stack([t[2] for t in batch]).to(agent.device)
    opt.zero_grad(set_to_none=True)
    loss = F.mse_loss(agent.e2.world_forward(z0, acts), z1)
    val = float(loss.detach().item())
    if not math.isfinite(val):
        return val
    loss.backward()
    torch.nn.utils.clip_grad_norm_(agent.e2.parameters(), max_norm=MAX_GRAD_NORM)
    opt.step()
    return val


def _run_p0(agent: REEAgent, seed: int, label: str, train: bool,
            battery_tensors: Any, p0_steps: int) -> Dict[str, Any]:
    env = _make_env(seed)
    _, obs_dict = env.reset()
    agent.reset()
    agent.e1.reset_hidden_state()
    opt = torch.optim.Adam(agent.e2.parameters(), lr=E2_LR)
    buf: Deque = deque(maxlen=BUF_MAX)
    rng = random.Random(seed)
    device = agent.device
    prev: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    trajectory: List[Dict[str, float]] = []
    losses: List[float] = []

    for step in range(p0_steps + 1):
        if step in P0_TRAJECTORY_AT or step == p0_steps:
            r = _battery_readouts(agent, battery_tensors)
            trajectory.append({"p0_step": float(step), "mse": r["mse"],
                               "infonce": r["infonce"]})
        if step == p0_steps:
            break
        if step % STEPS_PER_EPISODE == 0:
            ep = step // STEPS_PER_EPISODE
            print(f"  [train] {label} seed={seed} ep {ep + 1}/{EPISODES_PER_RUN} "
                  f"phase=P0 train={int(train)}", flush=True)
        z_now = _sense(agent, obs_dict).z_world.detach().reshape(-1).clone()
        if prev is not None and bool(torch.isfinite(z_now).all().item()):
            buf.append((prev[0], prev[1], z_now))
        idx = rng.randrange(env.action_dim)
        a_vec = torch.zeros(env.action_dim, dtype=torch.float32)
        a_vec[idx] = 1.0
        prev = (z_now, a_vec)
        _, _, done, _, obs_dict = env.step(a_vec.unsqueeze(0).to(device))
        if train:
            lv = _p0_train_step(agent, buf, opt, rng)
            if lv is not None:
                losses.append(lv)
        if done:
            _, obs_dict = env.reset()
            agent.e1.reset_hidden_state()
            prev = None

    return {"trajectory": trajectory, "p0_loss_mean": _mean(losses),
            "p0_n_train_steps": float(len(losses))}


# ---------------------------------------------------------------------------
# the base cache -- the mechanism that makes arm PAIRING exact
# ---------------------------------------------------------------------------
# The agent cannot be deep-copied (torch non-leaf tensors in the live graph), so
# the arms are paired by RE-RUN FROM A CACHED P0 HEAD rather than by forking a
# process state. Per (seed, base) we build the agent under a complete RNG reset,
# capture BOTH batteries with the (never-trained) encoder, run P0 once, and cache
# agent.e2.state_dict() plus the battery tensors. Each arm cell then re-seeds
# identically, builds a FRESH agent carrying that arm's flags, loads the cached
# state_dict (strict=True) and skips P0 entirely.
#
# Why that gives bit-identical pre-sleep state: agent construction draws its RNG
# from module init only, and every SD-PP object is plain Python (no nn.Module, no
# RNG draw -- pinned by the SD-PP-1/2 "no RNG consumption" contract tests), so
# all six arms leave the global RNG in the same place after construction. The
# waking loop then consumes the same draws in the same order in every arm. G2
# asserts this FROM OUTPUT rather than trusting the argument: the world buffer
# hash, the world-head parameter hash, and the torch RNG state entering
# force_cycle must all be equal across the six arms at cycle 1.
#
# CONSEQUENCE, stated rather than hidden: an arm cell's agent has no P0 SENSORY
# history -- only the e2 weights carry the base. That differs from a V3-EXQ-1063
# cell, which runs P0 on the same agent. It is uniform across arms, which is what
# the pairing needs, and it is the correct waking regime for the producers: the
# SD-PP-1/2 estimators must warm up on P1's regime, not on P0's.
# ---------------------------------------------------------------------------
def _build_base(seed: int, base: str, p0_steps: int, battery_size: int,
                calib_steps: int) -> Dict[str, Any]:
    label = f"BASE[{base}]"
    print(f"Seed {seed} {label}", flush=True)
    # LOAD-BEARING, and the reason is not hygiene. torch seeds its default
    # generator NON-DETERMINISTICALLY at first use, and only arm_cell.__enter__
    # reseeds -- so without this the base agent is built at a random RNG state
    # while every ARM cell builds its agent at reset_all_rng(seed). The encoder
    # is never trained, so that mismatch would mean the P0 head was fitted (and
    # the batteries captured) in ONE latent space while P1 consolidates it on
    # z_world from ANOTHER, with the DV read in the first. Resetting here makes
    # the base-build agent bit-identical to every arm cell's agent, which is
    # what the cached-head pairing assumes, and makes the whole run a pure
    # function of (substrate, config, seed) as the arm fingerprint claims.
    reset_all_rng(seed)
    env0 = _make_env(seed)
    # ARM_A flags: the base head must be a plain V3-EXQ-1063 head, and the
    # producers are not constructed during P0 in any case.
    agent = _make_agent(env0, ARM_A_BASELINE)
    device = agent.device

    battery_r0 = _sample_probe_battery(agent, seed, battery_size,
                                       invert_action_map=False)
    battery_r1 = _sample_probe_battery(agent, seed, battery_size,
                                       invert_action_map=True)
    t_r0 = _battery_tensors(battery_r0, device)
    t_r1 = _battery_tensors(battery_r1, device)
    agent.reset()
    agent.e1.reset_hidden_state()

    p0 = _run_p0(agent, seed, label, base == "converged", t_r0, p0_steps)
    traj = p0["trajectory"]
    mse_before = traj[0]["mse"]
    mse_after = traj[-1]["mse"]
    conv_rel_drop = (((mse_before - mse_after) / mse_before)
                     if _finite_or_none(mse_before) is not None and mse_before > EPS
                     else 0.0)

    state = {k: v.detach().clone() for k, v in agent.e2.state_dict().items()}
    # RED-TEAM F5. The batteries were encoded by THIS agent's latent stack, and
    # the encoder is never trained, so every arm agent must carry a bit-identical
    # one or its P1 z_world stream lives in a foreign latent space from the one
    # the P0 head was fitted in and the DV is read in. run_cell asserts this
    # hash rather than trusting the RNG argument.
    encoder_hash = _hash_tensor_list(list(agent.latent_stack.parameters()))
    out = {
        "seed": seed,
        "base": base,
        "e2_state": state,
        "encoder_hash": encoder_hash,
        "battery_r0": t_r0,
        "battery_r1": t_r1,
        "n_battery_r0": float(len(battery_r0)),
        "n_battery_r1": float(len(battery_r1)),
        "identity_mse_r0": _identity_predictor_mse(t_r0),
        "identity_mse_r1": _identity_predictor_mse(t_r1),
        "p0_trajectory": traj,
        "p0_loss_mean": p0["p0_loss_mean"],
        "p0_n_train_steps": p0["p0_n_train_steps"],
        "p0_battery_mse_before": mse_before,
        "p0_battery_mse_after": mse_after,
        "conv_rel_drop": float(conv_rel_drop),
    }
    # AMENDMENT 1: the calibration window, run AFTER P0 and after the batteries
    # are captured, on a separate ARM_B-flagged agent holding these same P0
    # weights. The UNCONVERGED base gets its own window, which is the point --
    # its calibration must reflect the fresh head's much larger residual, or
    # pi_cur could not separate the two bases (G5) for the right reason.
    calib = _run_calibration(seed, state, calib_steps)
    out["calibration_state"] = calib
    _wf, _or = calib["wf_precision"], calib["obs_reliability"]
    out["calib_v_tot"] = float(_wf.get("v_tot", float("nan")))
    out["calib_v_noise"] = float(_wf.get("v_noise", float("nan")))
    out["calib_n_obs"] = float(_wf.get("n_obs", float("nan")))
    out["calib_kappa"] = float(_or.get("kappa", float("nan")))
    out["calib_kappa_ready"] = float(_or.get("kappa_ready", float("nan")))
    out["calib_sigma_sq_ema"] = float(_or.get("sigma_sq_ema", float("nan")))
    out["calib_n_frames"] = float(_or.get("n_frames", float("nan")))
    out["calib_steps"] = float(calib_steps)
    print(f"  {label} seed={seed} CALIBRATION ({calib_steps} clean steps, env "
          f"seed+2): v_tot={out['calib_v_tot']:.6g} "
          f"v_noise={out['calib_v_noise']:.6g} n_obs={out['calib_n_obs']:.0f} "
          f"kappa={out['calib_kappa']:.6g} "
          f"kappa_ready={out['calib_kappa_ready']:.0f} "
          f"sigma_sq_ema={out['calib_sigma_sq_ema']:.6g} "
          f"n_frames={out['calib_n_frames']:.0f}", flush=True)
    print(f"  {label} seed={seed} conv_rel_drop={conv_rel_drop:.4f} "
          f"mse_p0 {mse_before:.6g} -> {mse_after:.6g} "
          f"identity_mse={out['identity_mse_r0']:.6g} "
          f"n_r0={out['n_battery_r0']:.0f} n_r1={out['n_battery_r1']:.0f}",
          flush=True)
    return out


def _load_base(agent: REEAgent, base_cache: Dict[str, Any]) -> None:
    """Load the cached P0 head AND, for the arms that have producers, the
    cached calibration state (AMENDMENT 1).

    ARM_A constructs neither producer, so it receives nothing -- structural
    absence, exactly as its gain is. The `.get` guard is load-bearing rather
    than defensive: _run_calibration itself calls into a cache that has no
    calibration_state yet, and the calibration agent MUST start cold.
    """
    agent.e2.load_state_dict(base_cache["e2_state"], strict=True)
    calib = base_cache.get("calibration_state") or {}
    wf = getattr(agent, "world_forward_precision", None)
    if wf is not None and calib.get("wf_precision"):
        wf.set_state(dict(calib["wf_precision"]))
    orel = getattr(agent, "observation_reliability", None)
    if orel is not None and calib.get("obs_reliability"):
        orel.set_state(dict(calib["obs_reliability"]))


# ---------------------------------------------------------------------------
# P1 -- the waking + sleep measurement loop
# ---------------------------------------------------------------------------
def _packet_summary(packets: Sequence[Any]) -> Dict[str, float]:
    """Distributional summary over the has_prev packets (preregistration
    section 8). Placeholders (episode starts, hypothesis-tagged ticks) carry
    pe=nan by SD-PP-3 contract and are excluded by has_prev."""
    real = [p for p in packets if bool(getattr(p, "has_prev", False))]
    # AMENDMENT 2: evidence_variance_z joins the full mean/min/max/sd/MEDIAN
    # summary. G3 reads its MEDIAN, because evidence_precision_z is 1/variance
    # and the MEAN of a reciprocal is dominated by its smallest denominators --
    # i.e. by the handful of cold-start frames, not by the channel.
    # RED-TEAM F2 adds kappa and sigma_obs: G3's new verdict-routing leg 2 is a
    # DIRECT assertion on sigma_obs (the sensory statistic), which is what the
    # old rule-shift leg was trying to say and could not, because it read
    # evidence_variance_z = kappa * sigma_obs_sq and kappa moves with motion.
    keys = ("pe", "pi_hist", "evidence_precision_z", "evidence_variance_z",
            "kappa", "sigma_obs", "surprise")
    out: Dict[str, float] = {"n_packets": float(len(packets)),
                             "n_has_prev": float(len(real))}
    for key in keys:
        vals = _finite_vals([getattr(p, key, float("nan")) for p in real])
        out[f"{key}_mean"] = _mean(vals)
        out[f"{key}_min"] = (min(vals) if vals else float("nan"))
        out[f"{key}_max"] = (max(vals) if vals else float("nan"))
        out[f"{key}_sd"] = _sd(vals)
        out[f"{key}_median"] = _median(vals)
    surprises = _finite_vals([getattr(p, "surprise", float("nan")) for p in real])
    out["surprise_frac_above_10"] = (
        (sum(1 for s in surprises if s > G4_SURPRISE_HIGH) / len(surprises))
        if surprises else float("nan"))
    return out


def _early_surprise_fraction(packets: Sequence[Any], n_early: int
                             ) -> Tuple[float, float]:
    """G4's post-shift check. The rule inversion is present from P1 step 0 in
    COND_CONFIDENTLY_WRONG, so "the first 30 real transitions after the shift"
    are simply the first n_early has_prev packets of P1. Returns
    (fraction with surprise > 10, n considered)."""
    real = [p for p in packets if bool(getattr(p, "has_prev", False))][:n_early]
    vals = _finite_vals([getattr(p, "surprise", float("nan")) for p in real])
    if not vals:
        return (float("nan"), 0.0)
    return (sum(1 for s in vals if s > G4_SURPRISE_HIGH) / len(vals),
            float(len(vals)))


def _trace_summary(trace: Sequence[Dict[str, Any]], module: str
                   ) -> Dict[str, float]:
    rows = [r for r in trace if str(r.get("module")) == module]
    scales = _finite_vals([r.get("step_scale") for r in rows])
    grads = _finite_vals([r.get("grad_norm") for r in rows])
    losses = _finite_vals([r.get("loss") for r in rows])
    return {
        "n_steps": float(len(rows)),
        "step_scale_mean": _mean(scales),
        "step_scale_min": (min(scales) if scales else float("nan")),
        "step_scale_max": (max(scales) if scales else float("nan")),
        "step_scale_sd": _sd(scales),
        "grad_norm_mean": _mean(grads),
        "loss_mean": _mean(losses),
    }


def _wake_step(agent: REEAgent, env: CausalGridWorldV2,
               obs_dict: Dict[str, Any], sigma: float,
               noise_gen: Optional[torch.Generator],
               act_rng: torch.Generator, device: Any
               ) -> Tuple[Dict[str, Any], bool]:
    """ONE waking tick, shared VERBATIM by the P1 measurement loop and by the
    _build_base calibration window (AMENDMENT 1).

    It is a single function precisely so the calibration window cannot drift
    from the regime it is calibrating for: if the two were written out twice,
    a later edit to one would silently teach the producers a different waking
    process than the one they are later read in.

    Returns (obs_dict, ok); ok is False on a non-finite action.
    """
    latent = _sense(agent, _apply_obs_noise(obs_dict, sigma, noise_gen))
    ticks = agent.clock.advance()
    wdim = latent.z_world.shape[-1]
    e1_prior = (agent._e1_tick(latent) if ticks.get("e1_tick", False)
                else torch.zeros(1, wdim, device=device))
    candidates = agent.generate_trajectories(latent, e1_prior, ticks)
    action = agent.select_action(candidates, ticks)
    if action is None:
        idx = int(torch.randint(0, env.action_dim, (1,),
                                generator=act_rng).item())
        action = torch.zeros(1, env.action_dim, device=device)
        action[0, idx] = 1.0
        agent._last_action = action
    if not bool(torch.isfinite(action).all().item()):
        return obs_dict, False
    # Without this the replay action buffer is all zeros and
    # world_action_encoder is structurally untrainable (nn.Linear, zero input
    # -> dL/dW = 0 EXACTLY).
    agent.record_executed_action(action)
    _, harm_signal, done, _info, obs_dict = env.step(action)
    with torch.no_grad():
        agent.update_residue(harm_signal=float(harm_signal), world_delta=None,
                             hypothesis_tag=False, owned=True)
    if done:
        _, obs_dict = env.reset()
        agent.e1.reset_hidden_state()
        agent.notify_env_reset()
    return obs_dict, True


def _run_calibration(seed: int, e2_state: Dict[str, torch.Tensor],
                     calib_steps: int) -> Dict[str, Dict[str, float]]:
    """AMENDMENT 1. Teach the SD-PP-1/2 EMAs what this head's clean waking
    residual and this channel's clean observation noise actually look like, and
    return their state so every arm can start from it instead of from v_init.

    WHY THIS EXISTS. The cached-head pairing carries the P0 WEIGHTS but no
    calibration history, so before this the full-budget probe measured
    pi_hist_min == 100.0 == 1/v_init in EVERY condition -- the floor of the
    "historical precision" distribution was the estimator's own cold start, not
    a fact about the model. An estimator that has never seen this head cannot
    report its precision.

    WHY IT IS PROVENANCE AND NOT A LEAK, in four parts, each independently
    sufficient: (a) the window is CLEAN -- non-inverted action map, zero
    observation noise -- so it contains no information about any manipulation;
    (b) it is strictly PRE-TEST, ending before the first measurement tick;
    (c) it runs on env seed+2, which no condition's measurement phase uses
    (those are seed+1) and which is not the battery seed (seed+9973); and
    (d) EVERY arm receives the IDENTICAL state, so it cannot differentiate the
    arms -- which is what a leak would have to do to matter.

    RNG: reset_all_rng(seed) then the same construct-env-then-agent order an
    arm cell uses, so the calibration agent's encoder is bit-identical to every
    arm agent's. Its draws all happen inside _build_base, before any per-arm
    reset_all_rng, so G2's bit-identity comparison is untouched.
    """
    reset_all_rng(seed)
    env_dims = _make_env(seed)
    agent = _make_agent(env_dims, ARM_B_STORE_ONLY)
    agent.e2.load_state_dict(e2_state, strict=True)
    device = agent.device

    env = _make_env(seed + 2)
    _, obs_dict = env.reset()
    agent.reset()
    agent.e1.reset_hidden_state()
    agent.notify_env_reset()
    act_rng = torch.Generator(device="cpu").manual_seed(seed + 7717)
    noise_gen = torch.Generator(device="cpu").manual_seed(seed + 31337)

    for _step in range(calib_steps):
        obs_dict, ok = _wake_step(agent, env, obs_dict, 0.0, noise_gen,
                                  act_rng, device)
        if not ok:
            break

    return {
        "wf_precision": dict(agent.world_forward_precision.get_state()),
        "obs_reliability": dict(agent.observation_reliability.get_state()),
        "calib_steps": float(calib_steps),
        "calib_env_seed": float(seed + 2),
    }


def _run_p1(agent: REEAgent, seed: int, arm: str, condition: str,
            base_cache: Dict[str, Any], n_cycles: int, wake_eps: int,
            steps: int) -> Dict[str, Any]:
    """V3-EXQ-1063's P1 loop verbatim, plus three additions, all applied in
    EVERY arm so the arms differ by the gain alone:
      (1) agent.notify_env_reset() immediately after every env.reset(), so the
          SD-PP-1 estimator never differences across an episode boundary and the
          next SD-PP-3 packet is a placeholder rather than a bogus cross-episode
          prediction error (a no-op when nothing is wired -- ARM_A);
      (2) the COND_NOISY_CONTRADICTION obs-noise hook on the world_state handed
          to _sense, drawn from a DEDICATED generator (seed+31337) so it consumes
          no global RNG;
      (3) the G2 hashes (world buffer, world-head params, torch RNG state)
          captured immediately before each force_cycle.
    """
    spec = COND_SPEC[condition]
    sigma = float(spec["obs_noise"])
    invert = bool(spec["invert_action_map"])
    device = agent.device
    world_params = _world_head_params(agent)
    t_r0 = base_cache["battery_r0"]
    t_r1 = base_cache["battery_r1"] if spec["correction_battery"] else None

    env = _make_env(seed + 1, invert_action_map=invert)
    _, obs_dict = env.reset()
    agent.reset()
    agent.e1.reset_hidden_state()
    agent.notify_env_reset()
    act_rng = torch.Generator(device="cpu").manual_seed(seed + 7717)
    noise_gen = torch.Generator(device="cpu").manual_seed(seed + 31337)

    per_cycle: List[Dict[str, Any]] = []
    traces: List[Dict[str, float]] = []
    packet_summaries: List[Dict[str, float]] = []
    cycles_fired = 0
    world_delta_total = 0.0
    finite_ok = True
    early_frac = float("nan")
    early_n = 0.0
    packets_seen = 0

    for cyc in range(n_cycles):
        for ep in range(wake_eps):
            ep_global = P0_EPISODE_EQUIV + cyc * wake_eps + ep
            print(f"  [train] {arm}|{condition} seed={seed} "
                  f"ep {ep_global + 1}/{EPISODES_PER_RUN} "
                  f"phase=P1 cycle={cyc + 1}/{n_cycles}", flush=True)
            for _step in range(steps):
                obs_dict, ok = _wake_step(agent, env, obs_dict, sigma,
                                          noise_gen, act_rng, device)
                if not ok:
                    finite_ok = False
                    break
            if not finite_ok:
                break
        if not finite_ok:
            break

        recorder = getattr(agent, "replay_provenance", None)
        all_packets = list(getattr(recorder, "packets", [])) if recorder else []
        if cyc == 0 and all_packets:
            early_frac, early_n = _early_surprise_fraction(
                all_packets, G4_EARLY_PACKETS)
        new_packets = all_packets[packets_seen:]
        packets_seen = len(all_packets)
        cycle_packets = _packet_summary(new_packets)
        cumulative_packets = _packet_summary(all_packets)
        packet_summaries.append(cycle_packets)

        wf = getattr(agent, "world_forward_precision", None)
        wf_snap = (dict(wf.snapshot()) if wf is not None else {})
        orel = getattr(agent, "observation_reliability", None)
        orel_snap = (dict(orel.snapshot()) if orel is not None else {})
        prov_stats = (dict(recorder.stats()) if recorder is not None else {})

        pre_r0 = _battery_readouts(agent, t_r0)
        pre_r1 = (_battery_readouts(agent, t_r1) if t_r1 is not None
                  else {"mse": float("nan"), "infonce": float("nan")})
        before = _snapshot(world_params)
        buffer_hash = _hash_world_buffer(agent)
        # RED-TEAM F5: the ACTION buffer is half the replay triple, and nothing
        # above hashed it -- an arm whose action stream drifted would have shown
        # an identical world-buffer hash.
        action_hash = _hash_tensor_list(
            [a.detach().reshape(1, -1)
             for a in getattr(agent, "_action_experience_buffer", [])])
        head_hash = _hash_tensor_list(before)
        rng_hash = _hash_rng_state()

        metrics = agent.sleep_loop.force_cycle(agent) or {}

        # RED-TEAM F5, second G2 leg: equal RNG state AFTER the cycle proves the
        # cycle itself consumed the same draws in every arm -- the pre-cycle
        # hash alone cannot say that.
        rng_hash_post = _hash_rng_state()
        after = _snapshot(world_params)
        post_r0 = _battery_readouts(agent, t_r0)
        post_r1 = (_battery_readouts(agent, t_r1) if t_r1 is not None
                   else {"mse": float("nan"), "infonce": float("nan")})

        cons = getattr(agent, "cross_module_consolidator", None)
        trace = list(getattr(cons, "last_step_trace", []) or []) if cons else []
        tsum = _trace_summary(trace, "e2_world")
        traces.append(tsum)

        fired = "post_sleep_z_goal_retention" in metrics
        if fired:
            cycles_fired += 1
        delta = _max_abs_delta(before, after)
        world_delta_total = max(world_delta_total, delta)

        row: Dict[str, Any] = {
            "cycle": float(cyc + 1),
            "r0_mse_pre": pre_r0["mse"], "r0_mse_post": post_r0["mse"],
            "r0_infonce_pre": pre_r0["infonce"], "r0_infonce_post": post_r0["infonce"],
            "r1_mse_pre": pre_r1["mse"], "r1_mse_post": post_r1["mse"],
            "r1_infonce_pre": pre_r1["infonce"], "r1_infonce_post": post_r1["infonce"],
            "world_head_max_abs_delta": delta,
            "sleep_cycle_fired": float(fired),
            "sws_n_writes": float(metrics.get("sws_n_writes", 0.0)),
            "rem_n_rollouts": float(metrics.get("rem_n_rollouts", 0.0)),
            "updates_e2": float(
                metrics.get("cross_module_consolidation_updates_e2", 0.0)),
            "updates_e2_world": float(
                metrics.get("cross_module_consolidation_updates_e2_world", 0.0)),
            "e2_world_key_present": bool(
                "cross_module_consolidation_updates_e2_world" in metrics),
            "step_scale_mean_e2_world": _finite_or_none(metrics.get(
                "cross_module_consolidation_step_scale_mean_e2_world")),
            "step_scale_min_e2_world": _finite_or_none(metrics.get(
                "cross_module_consolidation_step_scale_min_e2_world")),
            "step_scale_max_e2_world": _finite_or_none(metrics.get(
                "cross_module_consolidation_step_scale_max_e2_world")),
            "trace_e2_world": tsum,
            "packets_this_cycle": cycle_packets,
            "packets_cumulative": cumulative_packets,
            "replay_provenance_stats": prov_stats,
            "wf_precision_snapshot": wf_snap,
            "obs_reliability_snapshot": orel_snap,
            "pi_cur_at_sleep_entry": _finite_or_none(wf_snap.get("pi_cur")),
            "evidence_precision_z_at_sleep_entry": _finite_or_none(
                orel_snap.get("evidence_precision_z")),
            "world_buffer_hash": buffer_hash,
            "action_buffer_hash": action_hash,
            "world_head_hash_pre": head_hash,
            "rng_state_hash_pre_cycle": rng_hash,
            "rng_state_hash_post_cycle": rng_hash_post,
        }
        for key, val in metrics.items():
            if key.startswith("provenance_gain_last_"):
                num = _finite_or_none(val)
                if num is not None:
                    row[key] = num
        per_cycle.append(row)

    return {
        "per_cycle": per_cycle,
        "traces": traces,
        "packet_summaries": packet_summaries,
        "cycles_fired": cycles_fired,
        "world_head_max_abs_delta": world_delta_total,
        "finite_ok": finite_ok,
        "early_surprise_frac": early_frac,
        "early_surprise_n": early_n,
    }


# ---------------------------------------------------------------------------
# derived per-cell readouts (preregistration section 8)
# ---------------------------------------------------------------------------
def _relative(pre: Any, post: Any, kind: str) -> float:
    """retention  ret  = post/pre - 1   (0 unchanged, +1 doubled error)
       correction corr = (pre - post)/pre

    ACROSS-RUN convention (a design point the preregistration leaves open): pre
    is the FIRST cycle's pre-sleep readout and post the LAST cycle's post-sleep
    readout, so the scalar the contrasts consume is the cumulative effect of all
    N_CYCLES sleep cycles. The per-cycle values are recorded alongside.
    """
    p = _finite_or_none(pre)
    q = _finite_or_none(post)
    if p is None or q is None or abs(p) <= EPS:
        return float("nan")
    return (q / p - 1.0) if kind == "retention" else ((p - q) / p)


def _cell_derived(per_cycle: List[Dict[str, Any]], identity_mse_r0: float
                  ) -> Dict[str, float]:
    if not per_cycle:
        nan = float("nan")
        return {"retention_r0": nan, "correction_r1": nan, "letgo_r0": nan,
                "skill_r0": nan, "r0_mse_pre_first": nan, "r0_mse_post_last": nan,
                "r1_mse_pre_first": nan, "r1_mse_post_last": nan,
                "r0_mse_delta_sum": nan, "displacement_to_residual_ratio": nan}
    first, last = per_cycle[0], per_cycle[-1]
    ret = _relative(first["r0_mse_pre"], last["r0_mse_post"], "retention")
    corr = _relative(first["r1_mse_pre"], last["r1_mse_post"], "correction")
    pre0 = _finite_or_none(first["r0_mse_pre"])
    skill = (1.0 - pre0 / identity_mse_r0
             if pre0 is not None and _finite_or_none(identity_mse_r0) is not None
             and abs(identity_mse_r0) > EPS else float("nan"))
    disp = max(_finite_vals([c["world_head_max_abs_delta"] for c in per_cycle])
               or [float("nan")])
    ratio = (disp / math.sqrt(pre0)
             if pre0 is not None and pre0 > 0.0 and math.isfinite(disp)
             else float("nan"))
    delta_sum = sum(_finite_vals(
        [float(c["r0_mse_pre"]) - float(c["r0_mse_post"]) for c in per_cycle]))
    return {
        "retention_r0": ret,
        # "letting go": in COND_CONFIDENTLY_WRONG the R0 battery measures the
        # OLD rule, so its retention is the quantity that SHOULD move.
        "letgo_r0": ret,
        "correction_r1": corr,
        "skill_r0": skill,
        "r0_mse_pre_first": first["r0_mse_pre"],
        "r0_mse_post_last": last["r0_mse_post"],
        "r1_mse_pre_first": first["r1_mse_pre"],
        "r1_mse_post_last": last["r1_mse_post"],
        "r0_mse_delta_sum": delta_sum,
        "displacement_to_residual_ratio": ratio,
    }


# ---------------------------------------------------------------------------
# one (arm, condition, seed) cell
# ---------------------------------------------------------------------------
def run_cell(arm: str, condition: str, seed: int, base_cache: Dict[str, Any],
             global_scale: float, n_cycles: int, wake_eps: int, steps: int
             ) -> Dict[str, Any]:
    spec = ARM_SPEC[arm]
    cspec = COND_SPEC[condition]
    print(f"Seed {seed} Arm {arm} Condition {condition}", flush=True)

    env0 = _make_env(seed, invert_action_map=False)
    agent = _make_agent(env0, arm, global_scale=global_scale)
    assert agent.sleep_loop is not None, "use_sleep_loop=True must build sleep_loop"
    # RED-TEAM F5: the cached-head pairing is only valid if this agent's encoder
    # is the one the batteries were captured with and P0 was fitted against.
    _enc = _hash_tensor_list(list(agent.latent_stack.parameters()))
    if _enc != base_cache["encoder_hash"]:
        raise RuntimeError(
            f"ENCODER MISMATCH in {arm}|{condition}|seed{seed}: this agent's "
            f"latent stack hashes {_enc[:16]} but the cached base was built "
            f"with {base_cache['encoder_hash'][:16]}. The frozen batteries and "
            "the P0 head would be in a DIFFERENT latent space from this "
            "agent's P1 z_world stream, so every readout would be "
            "uninterpretable. This means the per-cell RNG reset did not "
            "reproduce the base-build construction order.")
    _load_base(agent, base_cache)
    world_params = _world_head_params(agent)

    p1 = _run_p1(agent, seed, arm, condition, base_cache, n_cycles, wake_eps, steps)
    per_cycle = p1["per_cycle"]

    n_world_buffer = len(getattr(agent, "_world_experience_buffer", []))
    n_action_buffer = len(getattr(agent, "_action_experience_buffer", []))
    n_pairs = float(min(n_world_buffer, n_action_buffer) - 1)
    try:
        action_nonzero_fraction = float(
            agent.e1_action_buffer_stats().get("nonzero_fraction", 0.0))
    except (AttributeError, RuntimeError, ValueError):
        action_nonzero_fraction = float("nan")

    # ---- the trainer's own loss and gradient, probed from output -----------
    # Run in EVERY arm so the global-RNG state is matched; grads are cleared
    # after, and the consolidator zero_grad()s before its own backward, so this
    # probe cannot leak an update. Probed AFTER the cycles so the buffers it
    # reads are the ones the cycles actually drew from.
    world_loss_probe = float("nan")
    world_grad_probe = float("nan")
    try:
        loss = agent.compute_e2_world_loss(batch_size=CMC_BATCH)
        world_loss_probe = float(loss.detach().item())
        for p in world_params:
            p.grad = None
        if loss.requires_grad and world_loss_probe != 0.0:
            loss.backward()
            world_grad_probe = max(
                (float(p.grad.abs().max().item()) if p.grad is not None else 0.0)
                for p in world_params) if world_params else 0.0
        else:
            world_grad_probe = 0.0
    except (RuntimeError, ValueError):
        world_loss_probe = 0.0
        world_grad_probe = 0.0
    finally:
        for p in world_params:
            p.grad = None

    derived = _cell_derived(per_cycle, base_cache["identity_mse_r0"])

    gain_means = _finite_vals([c["trace_e2_world"]["step_scale_mean"]
                               for c in per_cycle])
    gain_step_counts = _finite_vals([c["trace_e2_world"]["n_steps"]
                                     for c in per_cycle])
    n_missing_last = float("nan")
    n_rows_last = float("nan")
    for c in reversed(per_cycle):
        if "provenance_gain_last_n_missing" in c:
            n_missing_last = float(c["provenance_gain_last_n_missing"])
            n_rows_last = float(c.get("provenance_gain_last_n_rows", float("nan")))
            break
    rows_with_provenance = (
        1.0 - n_missing_last / n_rows_last
        if _finite_or_none(n_rows_last) is not None and n_rows_last > 0.0
        and _finite_or_none(n_missing_last) is not None else float("nan"))

    # RED-TEAM F4: the rule's three factors, averaged over the cell's cycles, so
    # a reader can see WHICH factor moved rather than only the product.
    gain_diag: Dict[str, float] = {}
    for _d in ("gain_min", "gain_max", "gain_sd", "k_mean", "m_mean",
               "r_mean", "r_max", "surprise_max", "pi_cur", "n_missing"):
        gain_diag[_d] = _mean([c.get(f"provenance_gain_last_{_d}")
                               for c in per_cycle])

    # RED-TEAM F3: packets-per-buffer-entry is 1.0 BY CONSTRUCTION (SD-PP-3
    # appends one per entry and trims in lockstep), so this is an ALIGNMENT
    # assertion, not a "did anything happen" check.
    packets_per_buffer_entry = float("nan")
    packets_has_prev = float("nan")
    if per_cycle:
        _pc = per_cycle[-1]["packets_cumulative"]
        packets_has_prev = float(_pc.get("n_has_prev", float("nan")))
        _npk = _finite_or_none(_pc.get("n_packets"))
        if _npk is not None and n_world_buffer > 0:
            packets_per_buffer_entry = float(_npk) / float(n_world_buffer)

    params_finite = bool(
        _all_finite(world_params)
        and _all_finite(list(agent.e2.parameters()))
        and _all_finite(list(agent.e1.parameters())))
    readouts_finite = bool(
        per_cycle
        and all(_finite_or_none(c[k]) is not None
                for c in per_cycle
                for k in ("r0_mse_pre", "r0_mse_post")))
    all_cycles_fired = bool(p1["cycles_fired"] == n_cycles)

    # ---- per-cell regime-conditioned gate ----------------------------------
    cell_id = f"{arm}|{condition}|seed{seed}"
    arm_ctx = {
        "id": cell_id, "arm": arm, "condition": condition, "seed": seed,
        "base": cspec["base"],
        "correction_battery": bool(cspec["correction_battery"]),
        "record_metadata": bool(spec["record_metadata"]),
        "gain_mode": spec["gain_mode"],
        "n_cycles": int(n_cycles),
    }
    measured: Dict[str, float] = {
        "world_replay_pairs_populated": n_pairs,
        "world_loss_non_sentinel": world_loss_probe,
        "world_grad_nonzero": world_grad_probe,
        "action_buffer_non_vacuous": action_nonzero_fraction,
        "retention_battery_populated": float(base_cache["n_battery_r0"]),
        "sleep_cycles_fired": float(p1["cycles_fired"]),
    }
    if cspec["correction_battery"]:
        measured["correction_battery_populated"] = float(base_cache["n_battery_r1"])
    if cspec["base"] == "converged":
        measured["base_converged"] = float(base_cache["conv_rel_drop"])
    else:
        measured["base_unconverged"] = float(base_cache["conv_rel_drop"])
    if spec["record_metadata"]:
        measured["provenance_packets_recorded"] = packets_per_buffer_entry
    if spec["gain_mode"] is not None:
        measured["gain_rows_with_provenance"] = rows_with_provenance
    gate = evaluate_arm_gate(cell_id, arm_ctx, PRECONDITIONS, measured)

    cell_ok = bool(gate["gate_green"] and params_finite and readouts_finite
                   and all_cycles_fired)

    pi_cur_print = (per_cycle[0]["pi_cur_at_sleep_entry"]
                    if per_cycle else float("nan"))
    print(
        f"  {arm}|{condition} seed={seed} "
        f"ret={derived['retention_r0']:.6g} corr={derived['correction_r1']:.6g} "
        f"skill={derived['skill_r0']:.6g} "
        f"disp={p1['world_head_max_abs_delta']:.6g} "
        f"gain_mean={_mean(gain_means):.6g} "
        f"pi_cur={pi_cur_print} "
        f"cycles={p1['cycles_fired']}/{n_cycles} n_pairs={n_pairs:.0f} "
        f"act_nonzero={action_nonzero_fraction:.3g} gate={int(gate['gate_green'])}",
        flush=True)
    if spec["gain_mode"] is not None:
        print(
            f"  {arm}|{condition} seed={seed} [gain] "
            f"k_mean={gain_diag['k_mean']:.6g} m_mean={gain_diag['m_mean']:.6g} "
            f"r_mean={gain_diag['r_mean']:.6g} r_max={gain_diag['r_max']:.6g} "
            f"gain_min={gain_diag['gain_min']:.6g} "
            f"gain_max={gain_diag['gain_max']:.6g} "
            f"surprise_max={gain_diag['surprise_max']:.6g} "
            f"step_scale={_mean(gain_means):.6g} "
            f"rows_with_prov={rows_with_provenance:.4g} "
            f"pkt_per_entry={packets_per_buffer_entry:.4g}", flush=True)
    print(f"verdict: {'PASS' if cell_ok else 'FAIL'}", flush=True)

    row: Dict[str, Any] = {
        "arm": arm,
        "condition": condition,
        "seed": seed,
        "base": cspec["base"],
        "gain_mode": spec["gain_mode"],
        "global_scale": float(global_scale),
        "cell_ok": cell_ok,
        "gate": gate,
        "params_finite": params_finite,
        "readouts_finite": readouts_finite,
        "all_cycles_fired": all_cycles_fired,
        "cycles_fired": float(p1["cycles_fired"]),
        "conv_rel_drop": float(base_cache["conv_rel_drop"]),
        "p0_battery_mse_before": base_cache["p0_battery_mse_before"],
        "p0_battery_mse_after": base_cache["p0_battery_mse_after"],
        "p0_n_train_steps": base_cache["p0_n_train_steps"],
        "calib_steps": base_cache.get("calib_steps", float("nan")),
        "calib_v_tot": base_cache.get("calib_v_tot", float("nan")),
        "calib_v_noise": base_cache.get("calib_v_noise", float("nan")),
        "calib_kappa": base_cache.get("calib_kappa", float("nan")),
        "calib_kappa_ready": base_cache.get("calib_kappa_ready", float("nan")),
        "calib_sigma_sq_ema": base_cache.get("calib_sigma_sq_ema", float("nan")),
        "calibration_restored": bool(spec["record_metadata"]),
        "identity_predictor_mse_r0": base_cache["identity_mse_r0"],
        "identity_predictor_mse_r1": base_cache["identity_mse_r1"],
        "n_battery_r0": float(base_cache["n_battery_r0"]),
        "n_battery_r1": float(base_cache["n_battery_r1"]),
        "n_pairs": n_pairs,
        "action_buffer_nonzero_fraction": action_nonzero_fraction,
        "world_loss_probe": world_loss_probe,
        "world_grad_probe": world_grad_probe,
        "world_head_max_abs_delta": float(p1["world_head_max_abs_delta"]),
        "adam_step_bound_8x_lr": float(CMC_STEPS * CMC_LR),
        "gain_mean_over_cycles": _mean(gain_means),
        "gain_step_counts": gain_step_counts,
        "gain_n_missing_last": n_missing_last,
        "gain_n_rows_last": n_rows_last,
        "gain_rows_with_provenance": rows_with_provenance,
        "packets_per_buffer_entry": packets_per_buffer_entry,
        "packets_has_prev": packets_has_prev,
        "gain_diagnostics": gain_diag,
        "encoder_hash": base_cache["encoder_hash"],
        "early_surprise_frac": float(p1["early_surprise_frac"]),
        "early_surprise_n": float(p1["early_surprise_n"]),
        "per_cycle": per_cycle,
        "trace_summaries": p1["traces"],
        "packet_summaries": p1["packet_summaries"],
        "world_buffer_hash_cycle1": (
            per_cycle[0]["world_buffer_hash"] if per_cycle else ""),
        "action_buffer_hash_cycle1": (
            per_cycle[0]["action_buffer_hash"] if per_cycle else ""),
        "rng_state_hash_post_cycle1": (
            per_cycle[0]["rng_state_hash_post_cycle"] if per_cycle else ""),
        "world_head_hash_cycle1": (
            per_cycle[0]["world_head_hash_pre"] if per_cycle else ""),
        "rng_state_hash_cycle1": (
            per_cycle[0]["rng_state_hash_pre_cycle"] if per_cycle else ""),
        "agent": agent,
    }
    row.update(derived)
    return row


# ---------------------------------------------------------------------------
# G1 -- the liveness ladder (SD-PP-4 test 10; preregistration section 9)
# ---------------------------------------------------------------------------
def run_liveness(seed: int, base_cache: Dict[str, Any], n_cycles: int,
                 wake_eps: int, steps: int) -> Dict[str, Any]:
    """Re-measure SD-PP-4 test 10 on the LIVE driver: with everything else held
    fixed, is world-head displacement strictly increasing in the gain scale?

    Each rung re-seeds completely, rebuilds a fresh agent from the SAME cached
    P0 head under ARM_D_GLOBAL flags (provenance_gain_mode="global") at that
    rung's global_scale, runs the identical P1 and forces the cycle(s). Writes
    no manifest. A non-monotone ladder means the gain manipulation is INERT and
    no contrast in this run can be read (matrix R5).

    RED-TEAM F1 -- the ladder is also the ATTAINABILITY CURVE. It records
    retention_r0 per rung, so the manifest answers "what gain would have been
    needed to hold retention at X?" directly, in measured numbers, instead of
    leaving a pre-registered absolute ceiling to be read as reachable when it is
    not. The two smallest rungs (0.02, 0.05) exist for that question alone.
    """
    rungs: List[Dict[str, float]] = []
    cfg_base = {
        "experiment": EXPERIMENT_TYPE, "phase": "G1_liveness",
        "arm": ARM_D_GLOBAL, "condition": COND_CONVERGED_CLEAN,
        "n_cycles": n_cycles, "wake_eps_per_cycle": wake_eps,
        "steps_per_episode": steps,
    }
    for scale in LIVENESS_SCALES:
        cfg = dict(cfg_base)
        cfg["global_scale"] = float(scale)
        with arm_cell(seed, config_slice=cfg, script_path=Path(__file__),
                      extra_ineligible_reasons=[
                          "liveness_probe_no_reuse"]) as cell:
            env0 = _make_env(seed)
            agent = _make_agent(env0, ARM_D_GLOBAL, global_scale=float(scale))
            _load_base(agent, base_cache)
            p1 = _run_p1(agent, seed, f"G1_liveness_s{scale}",
                         COND_CONVERGED_CLEAN, base_cache, n_cycles, wake_eps,
                         steps)
            _der = _cell_derived(p1["per_cycle"], base_cache["identity_mse_r0"])
            rung = {
                "global_scale": float(scale),
                "world_head_max_abs_delta": float(p1["world_head_max_abs_delta"]),
                "step_scale_mean_e2_world": _mean(
                    [t["step_scale_mean"] for t in p1["traces"]]),
                "cycles_fired": float(p1["cycles_fired"]),
                # RED-TEAM F1: the attainability curve.
                "retention_r0": float(_der["retention_r0"]),
                "r0_mse_pre_first": float(_der["r0_mse_pre_first"]),
                "r0_mse_post_last": float(_der["r0_mse_post_last"]),
            }
            cell.stamp(rung)
        rungs.append(rung)
        print(f"  [G1] global_scale={scale:.4g} "
              f"world_head_max_abs_delta={rung['world_head_max_abs_delta']:.6g} "
              f"realised_step_scale={rung['step_scale_mean_e2_world']:.6g} "
              f"retention_r0={rung['retention_r0']:.6g}", flush=True)

    deltas = [r["world_head_max_abs_delta"] for r in rungs]
    strictly_increasing = bool(
        len(deltas) == len(LIVENESS_SCALES)
        and all(_finite_or_none(d) is not None for d in deltas)
        and all(deltas[i + 1] > deltas[i] for i in range(len(deltas) - 1)))
    n_increasing = sum(1 for i in range(len(deltas) - 1)
                       if _finite_or_none(deltas[i]) is not None
                       and _finite_or_none(deltas[i + 1]) is not None
                       and deltas[i + 1] > deltas[i])
    return {
        "seed": seed,
        "scales": list(LIVENESS_SCALES),
        "rungs": rungs,
        "strictly_increasing": strictly_increasing,
        "n_increasing_steps": float(n_increasing),
        "n_steps_required": float(len(LIVENESS_SCALES) - 1),
        # RED-TEAM F1: what retention each gain budget actually buys, measured.
        "attainability": [
            {"global_scale": r["global_scale"],
             "retention_r0": r["retention_r0"],
             "world_head_max_abs_delta": r["world_head_max_abs_delta"]}
            for r in rungs],
        "attainability_note": (
            "retention_r0 = post/pre - 1 on the retention battery at each gain "
            "budget, on the pinned seed's converged base. It answers what a "
            "given budget BUYS. V3-EXQ-1063's landed seed-42 numbers (pre "
            "1.4868e-5, 3-cycle mse_delta_sum -7.142e-5) put ret(B) near 4.8 "
            "rather than the ~1.0 the preregistration's 10%-of-residual ceiling "
            "assumed, which is why that ceiling is now a REPORT (P2b) and not a "
            "leg of P2 -- see RED-TEAM F1."),
    }


def _print_liveness_table(liveness: Dict[str, Any]) -> None:
    print("\n[G1 LIVENESS] displacement vs gain scale "
          f"(seed {liveness['seed']}, ARM_D_GLOBAL flags, cached P0 head)",
          flush=True)
    print("  global_scale | realised_step_scale | world_head_max_abs_delta | "
          "retention_r0", flush=True)
    for rung in liveness["rungs"]:
        print(f"  {rung['global_scale']:>12.4g} | "
              f"{rung['step_scale_mean_e2_world']:>19.6g} | "
              f"{rung['world_head_max_abs_delta']:>24.6g} | "
              f"{rung['retention_r0']:>12.6g}", flush=True)
    print(f"  strictly_increasing={liveness['strictly_increasing']} "
          f"({liveness['n_increasing_steps']:.0f}/"
          f"{liveness['n_steps_required']:.0f} rungs increased)", flush=True)


# ---------------------------------------------------------------------------
# contrasts P1..P7 (preregistration section 11)
# ---------------------------------------------------------------------------
def _effective_margin(deltas: Sequence[float]) -> Tuple[float, float, str]:
    """MARGIN is an ABSOLUTE FLOOR. Where the per-seed SD of the paired deltas
    exceeds it, the SD-scaled margin (1 SD) is reported alongside AND THE
    DECISION USES THE LARGER (preregistration section 11)."""
    sd = _sd(deltas)
    if _finite_or_none(sd) is not None and sd > MARGIN:
        return (float(sd), float(sd), "sd_1")
    return (MARGIN, float(sd) if _finite_or_none(sd) is not None else float("nan"),
            "absolute_0.02")


def _paired(rows: Dict[Tuple[str, str, int], Dict[str, Any]],
            seeds: Sequence[int], condition: str, arm_a: str, arm_b: str,
            key: str) -> List[float]:
    out: List[float] = []
    for seed in seeds:
        ra = rows.get((arm_a, condition, seed))
        rb = rows.get((arm_b, condition, seed))
        if ra is None or rb is None:
            out.append(float("nan"))
            continue
        va, vb = _finite_or_none(ra.get(key)), _finite_or_none(rb.get(key))
        out.append(va - vb if va is not None and vb is not None else float("nan"))
    return out


def _values(rows: Dict[Tuple[str, str, int], Dict[str, Any]],
            seeds: Sequence[int], condition: str, arm: str, key: str
            ) -> List[float]:
    out: List[float] = []
    for seed in seeds:
        r = rows.get((arm, condition, seed))
        v = _finite_or_none(r.get(key)) if r is not None else None
        out.append(v if v is not None else float("nan"))
    return out


def _count(flags: Sequence[bool]) -> int:
    return sum(1 for f in flags if f)


def _contrast_record(name: str, description: str, deltas: List[float],
                     flags: List[bool], seeds: Sequence[int], req: int,
                     margin_used: float, margin_sd: float, margin_source: str,
                     load_bearing: bool, extra: Optional[Dict[str, Any]] = None
                     ) -> Dict[str, Any]:
    lo, hi = _ci95(deltas)
    n_pos = _count([_finite_or_none(d) is not None and d > 0.0 for d in deltas])
    n_neg = _count([_finite_or_none(d) is not None and d < 0.0 for d in deltas])
    rec: Dict[str, Any] = {
        "name": name,
        "description": description,
        "load_bearing": bool(load_bearing),
        "routes_verdict": False,
        "passed": bool(_count(flags) >= req),
        "measured": float(_count(flags)),
        "threshold": float(req),
        "comparator": ">=",
        "per_seed_delta": [float(d) for d in deltas],
        "per_seed_flag": [bool(f) for f in flags],
        "seeds": list(seeds),
        "mean": _mean(deltas),
        "sd": _sd(deltas),
        "ci95_low": lo,
        "ci95_high": hi,
        "ci_low_power": True,
        "ci_note": ("t-based 95% CI at n=3, DESCRIPTIVE ONLY. The pre-registered "
                    "decision rule is the 2-of-3 seed sign count in measured/"
                    "threshold."),
        "margin_used": float(margin_used),
        "margin_absolute": float(MARGIN),
        "margin_sd_1": float(margin_sd),
        "margin_source": margin_source,
        "sign_count_positive": float(n_pos),
        "sign_count_negative": float(n_neg),
    }
    if extra:
        rec.update(extra)
    return rec


# ---------------------------------------------------------------------------
def _label_from_matrix(g1: bool, g3: bool, g4: bool, g5: bool, g6: bool,
                       cond3_readable: bool, p5_non_degenerate: bool,
                       p2: bool, p3: bool, p4: bool, p5: bool, p6: bool
                       ) -> str:
    """Preregistration section 12, made a TOTAL function in a pinned order.
    Gates first (a degenerate channel cannot be read as a result), then the
    cond-3 readability branch (AMENDMENT 3c), then the self-sealing route (R3
    is a finding about P4 that R1/R2 would mask), then R1, R2, R7, then mixed.

    AMENDMENT 3c. R1 and R3 both turn on P4, and P4 is read on
    COND_CONFIDENTLY_WRONG. When G4b/G9/G3b say that condition posed no
    readable contradiction on this head, P4 is not a null -- it is UNASKED, and
    a matrix branch that consumed it would convert a substrate limit into a
    scientific finding. So that case gets its own label.

    BATCH 2 EDIT 2. Inside that branch R7 is tested FIRST, because R7 turns on
    P5, which is read on COND_NOISY_CONTRADICTION and is therefore still a real
    reading when cond 3 is unreadable -- but only when P5 is itself
    non-degenerate. A genuine "evidence precision adds nothing beyond raw
    error" finding must not be relabelled as an unposeable condition just
    because a DIFFERENT condition could not be posed.
    """
    if not (g1 and g6):
        return "instrument_inert_substrate_not_ready"          # R5
    if not (g3 and g4 and g5):
        return "precision_channel_degenerate"                  # R6
    if not cond3_readable:
        if p5_non_degenerate and not p5:
            return "evidence_precision_not_load_bearing"       # R7
        if p2:
            return "confidently_wrong_condition_unposeable_on_this_head"
        return "mixed_inconclusive"
    if p2 and not p4:
        return "self_sealing_failure"                          # R3
    if p2 and p3 and p4 and p6:
        return "provenance_beyond_generic_gain"                # R1
    if p2 and p4 and not p6:
        return "generic_gain_correction_sufficient"            # R2
    if not p5:
        return "evidence_precision_not_load_bearing"           # R7
    return "mixed_inconclusive"


def _print_probe(rows: Dict[Tuple[str, str, int], Dict[str, Any]], seed: int,
                 bases: Dict[Tuple[int, str], Dict[str, Any]]) -> None:
    """RED-TEAM F7. The section-15 FREEZE ARTIFACT: every channel statistic the
    gates are defined on, measured at FULL P0 and FULL calibration on one seed,
    printed so the freeze record can quote it. Writes nothing.

    Read from ARM_B_STORE_ONLY, which carries the producers and whose waking is
    bit-identical to ARM_A's, plus ARM_C_PROVENANCE for the gain diagnostics
    and ARM_D_RESIDUAL for the rival scheduler's realised budget (BATCH 4).
    G8's skill is a property of the BASE (it is the cycle-1 pre-sleep readout
    against the identity predictor, identical across arms), so reading it from
    ARM_B rather than ARM_A is the same number.
    """
    def _b(cond: str) -> Optional[Dict[str, Any]]:
        r = rows.get((ARM_B_STORE_ONLY, cond, seed))
        return r["per_cycle"][0] if (r and r["per_cycle"]) else None

    def _pk(cond: str, key: str) -> float:
        c = _b(cond)
        return (float(c["packets_cumulative"].get(key, float("nan")))
                if c else float("nan"))

    print(f"\n[PROBE] V3-EXQ-1073 channel statistics -- seed {seed}, FULL P0 "
          f"and FULL calibration, 1 cycle. Section-15 freeze artifact.",
          flush=True)
    for (bseed, bbase), bc in sorted(bases.items(), key=lambda kv: kv[0][1]):
        print(f"  base[{bbase}] conv_rel_drop={bc['conv_rel_drop']:.4f} "
              f"mse_after_p0={bc['p0_battery_mse_after']:.6g} "
              f"identity_mse={bc['identity_mse_r0']:.6g} "
              f"calib v_tot={bc['calib_v_tot']:.6g} "
              f"kappa={bc['calib_kappa']:.6g} "
              f"kappa_ready={bc['calib_kappa_ready']:.0f}", flush=True)

    print("  -- per condition (ARM_B_STORE_ONLY, cycle 1) --", flush=True)
    hdr = ("  cond                      ev_var_med   sigma_med   kappa_med"
           "    pi_hist_sd   surprise_med  surpr>10   pe_mean      pi_cur"
           "      mse_pre      skill")
    print(hdr, flush=True)
    for cond in CONDITIONS:
        c = _b(cond)
        r = rows.get((ARM_B_STORE_ONLY, cond, seed))
        pi_cur = (c["pi_cur_at_sleep_entry"] if c else float("nan"))
        print(f"  {cond:<24} "
              f"{_pk(cond, 'evidence_variance_z_median'):>10.4g} "
              f"{_pk(cond, 'sigma_obs_median'):>11.4g} "
              f"{_pk(cond, 'kappa_median'):>11.4g} "
              f"{_pk(cond, 'pi_hist_sd'):>13.4g} "
              f"{_pk(cond, 'surprise_median'):>14.4g} "
              f"{_pk(cond, 'surprise_frac_above_10'):>9.4g} "
              f"{_pk(cond, 'pe_mean'):>10.4g} "
              f"{float(pi_cur) if pi_cur is not None else float('nan'):>11.4g} "
              f"{float(r['r0_mse_pre_first']) if r else float('nan'):>12.4g} "
              f"{float(r['skill_r0']) if r else float('nan'):>10.4g}",
              flush=True)

    # The gate statistics, computed exactly as the full run computes them.
    v1 = _pk(COND_CONVERGED_CLEAN, "evidence_variance_z_median")
    v3 = _pk(COND_CONFIDENTLY_WRONG, "evidence_variance_z_median")
    v4 = _pk(COND_NOISY_CONTRADICTION, "evidence_variance_z_median")
    s1 = _pk(COND_CONVERGED_CLEAN, "sigma_obs_median")
    s3 = _pk(COND_CONFIDENTLY_WRONG, "sigma_obs_median")
    pi1 = _b(COND_CONVERGED_CLEAN)
    pi2 = _b(COND_UNDERFIT)
    r3 = rows.get((ARM_B_STORE_ONLY, COND_CONFIDENTLY_WRONG, seed))
    r1 = rows.get((ARM_B_STORE_ONLY, COND_CONVERGED_CLEAN, seed))
    print("  -- gate statistics --", flush=True)
    print(f"  G3  leg1 var_med(c1)/var_med(c4) = "
          f"{(v1 / v4 if v4 else float('nan')):.6g}  (< {G3_NOISE_RATIO_MAX})",
          flush=True)
    print(f"  G3  leg2 sigma_obs med c1={s1:.6g} c3={s3:.6g} "
          f"floor={SIGMA_FLOOR:.6g}  (must be EQUAL and AT the floor)",
          flush=True)
    print(f"  G3b |log2(var_med(c3)/var_med(c1))| = "
          f"{(abs(math.log2(v3 / v1)) if (v1 > 0 and v3 > 0) else float('nan')):.6g}"
          f"  (<= {G3_SHIFT_LOG2_MAX})", flush=True)
    print(f"  G4  min pi_hist_sd over conditions = "
          f"{min([_pk(c, 'pi_hist_sd') for c in CONDITIONS]):.6g}  (> 0)",
          flush=True)
    print(f"  G4b cond3 early surprise frac = "
          f"{(float(r3['early_surprise_frac']) if r3 else float('nan')):.6g} "
          f"(>= {G4_EARLY_FRAC_FLOOR}); cond1 surprise median = "
          f"{_pk(COND_CONVERGED_CLEAN, 'surprise_median'):.6g} "
          f"(< {G4_CLEAN_MEDIAN_MAX})", flush=True)
    _a = (pi1["pi_cur_at_sleep_entry"] if pi1 else None)
    _bq = (pi2["pi_cur_at_sleep_entry"] if pi2 else None)
    print(f"  G5  pi_cur(c1)/pi_cur(c2) = "
          f"{(_a / _bq if (_a and _bq) else float('nan')):.6g} "
          f"(> {G5_PI_CUR_RATIO})", flush=True)
    print(f"  G8  converged-base skill = "
          f"{(float(r1['skill_r0']) if r1 else float('nan')):.6g}  (> 0)",
          flush=True)
    print(f"  G9  cond3 pe_mean / cond1 residual = "
          f"{(_pk(COND_CONFIDENTLY_WRONG, 'pe_mean') / float(r1['r0_mse_pre_first']) if (r1 and float(r1['r0_mse_pre_first']) > 0) else float('nan')):.6g}"
          f"  (> {G9_PE_RATIO})", flush=True)

    print("  -- ARM_C_PROVENANCE gain rule, per condition --", flush=True)
    print("  cond                        k_mean      m_mean      r_mean"
          "      r_max    gain_min    gain_max  step_scale", flush=True)
    for cond in CONDITIONS:
        r = rows.get((ARM_C_PROVENANCE, cond, seed))
        gd = (r.get("gain_diagnostics", {}) if r else {})
        print(f"  {cond:<24} "
              f"{gd.get('k_mean', float('nan')):>11.4g} "
              f"{gd.get('m_mean', float('nan')):>11.4g} "
              f"{gd.get('r_mean', float('nan')):>11.4g} "
              f"{gd.get('r_max', float('nan')):>10.4g} "
              f"{gd.get('gain_min', float('nan')):>11.4g} "
              f"{gd.get('gain_max', float('nan')):>11.4g} "
              f"{(float(r['gain_mean_over_cycles']) if r else float('nan')):>11.4g}",
              flush=True)

    # BATCH 4: the current-residual scheduler's realised budget, and the ratio
    # P5 must be read against. Dr evaluates no rule, so k/m/r are nan for it by
    # construction -- its gain is clip(gain_max*sqrt(pe_cur/v_ref), min, max).
    print("  -- ARM_D_RESIDUAL (current-residual scheduler), per condition --",
          flush=True)
    print("  cond                      gain_min    gain_max  step_scale"
          "   step_scale_C  dr/C ratio", flush=True)
    for cond in CONDITIONS:
        rd = rows.get((ARM_D_RESIDUAL, cond, seed))
        rc = rows.get((ARM_C_PROVENANCE, cond, seed))
        gdd = (rd.get("gain_diagnostics", {}) if rd else {})
        vd = (float(rd["gain_mean_over_cycles"]) if rd else float("nan"))
        vc = (float(rc["gain_mean_over_cycles"]) if rc else float("nan"))
        ratio = (vd / vc if (math.isfinite(vd) and math.isfinite(vc) and vc > 0.0)
                 else float("nan"))
        print(f"  {cond:<24} "
              f"{gdd.get('gain_min', float('nan')):>9.4g} "
              f"{gdd.get('gain_max', float('nan')):>11.4g} "
              f"{vd:>11.4g} "
              f"{vc:>14.4g} "
              f"{ratio:>11.4g}", flush=True)
    _c4d = rows.get((ARM_D_RESIDUAL, COND_NOISY_CONTRADICTION, seed))
    _c4c = rows.get((ARM_C_PROVENANCE, COND_NOISY_CONTRADICTION, seed))
    if _c4d and _c4c:
        _vd = float(_c4d["gain_mean_over_cycles"])
        _vc = float(_c4c["gain_mean_over_cycles"])
        print(f"  P5 DISCRIMINATION CHECK (cond 4): Dr step_scale {_vd:.4g} vs "
              f"C {_vc:.4g} -> Dr is "
              f"{'ABOVE' if _vd > _vc else 'AT OR BELOW'} C "
              f"({(_vd / _vc if _vc > 0 else float('nan')):.4g}x). P5 "
              f"{'DISCRIMINATES' if _vd > _vc else 'HAS LOST ITS CONTRAST'}: "
              "the noisy channel must drive the residual-only scheduler HARDER "
              "than the evidence-precision-weighted rule, or C ignoring the "
              "noise is not distinguishable from C simply running colder.",
              flush=True)


EVIDENCE_LEVEL_CAP = (
    "CEILING (preregistration section 14, Result 4): organism-level endpoint "
    "absent by construction -- no default-on consumer of e2.world_forward "
    "exists in E3, register B1. The ceiling of this run is mechanistic/local. "
    "Governance may cite the mechanistic pattern but must NOT promote MECH-572 "
    "or register a new claim from it; the organism-level follow-up is a "
    "separate chip on B1."
)


# ---------------------------------------------------------------------------
def main(dry_run: bool = False, liveness_only: bool = False,
         probe_only: bool = False) -> Tuple[str, Optional[str]]:
    # RED-TEAM F7: --probe is the section-15 FREEZE ARTIFACT. It runs at FULL
    # P0 and FULL calibration on one seed -- the only budget at which the
    # channel statistics mean anything -- so it must not inherit the dry-run
    # shortenings.
    seeds = list(SEEDS[:1]) if (dry_run or liveness_only or probe_only) else list(SEEDS)
    p0_steps = 180 if (dry_run or liveness_only) else P0_STEPS
    n_cycles = 1 if (dry_run or liveness_only or probe_only) else N_CYCLES
    wake_eps = 1 if (dry_run or liveness_only) else WAKE_EPS_PER_CYCLE
    steps = 30 if (dry_run or liveness_only) else STEPS_PER_EPISODE
    battery = BATTERY_SIZE   # NEVER shortened -- the DV is read on it
    # ONLY --dry-run shortens the calibration window. --liveness keeps the full
    # CALIB_STEPS on purpose: at 30 steps the converged base never reaches
    # kappa_ready (measured: kappa 1.0 == the not-ready sentinel, and v_tot
    # 2.3e-3 against its true 1.2e-5), so a shortened window would have the G1
    # probe exercise a calibration path the real run never takes.
    calib_steps = 30 if dry_run else CALIB_STEPS
    req = min(SEEDS_REQUIRED, len(seeds))

    arm_contexts = [
        {"id": f"{a}|{c}", "arm": a, "condition": c,
         "base": COND_SPEC[c]["base"],
         "correction_battery": bool(COND_SPEC[c]["correction_battery"]),
         "record_metadata": bool(ARM_SPEC[a]["record_metadata"]),
         "gain_mode": ARM_SPEC[a]["gain_mode"],
         "n_cycles": int(n_cycles)}
        for a in ARMS for c in CONDITIONS]
    # Refuses the run BEFORE compute if any cell carries a structurally
    # unsatisfiable precondition. A pre-registered value that provably fails a
    # gate is a design-time proof -- never lower the threshold to resolve it.
    assert_no_structurally_unsatisfiable_gate(PRECONDITIONS, arm_contexts)

    t0 = time.time()
    rows: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
    arm_results: List[Dict[str, Any]] = []
    agents_seen: List[REEAgent] = []
    agents_by_arm: Dict[str, REEAgent] = {}
    bases: Dict[Tuple[int, str], Dict[str, Any]] = {}
    c_seed_by_seed: Dict[int, float] = {}

    def _base_for(seed: int, condition: str) -> Dict[str, Any]:
        key = (seed, COND_SPEC[condition]["base"])
        if key not in bases:
            bases[key] = _build_base(seed, key[1], p0_steps, battery,
                                     calib_steps)
        return bases[key]

    # ---- the cells --------------------------------------------------------
    def _run_one(arm: str, condition: str, seed: int, scale: float) -> None:
        cfg_slice = {
            "experiment": EXPERIMENT_TYPE,
            "arm": arm,
            "condition": condition,
            "base": COND_SPEC[condition]["base"],
            "invert_action_map": COND_SPEC[condition]["invert_action_map"],
            "obs_noise_sigma": COND_SPEC[condition]["obs_noise"],
            "record_metadata": ARM_SPEC[arm]["record_metadata"],
            "gain_mode": ARM_SPEC[arm]["gain_mode"],
            "global_scale": float(scale),
            "p0_steps": p0_steps,
            "n_cycles": n_cycles,
            "wake_eps_per_cycle": wake_eps,
            "steps_per_episode": steps,
            "battery_size": battery,
            "e2_lr": E2_LR,
            "cmc_steps": CMC_STEPS,
            "cmc_lr": CMC_LR,
            "cmc_batch": CMC_BATCH,
            "gain_min": GAIN_MIN,
            "gain_max": GAIN_MAX,
            "gain_v_ref": GAIN_V_REF,
            "gain_surprise_beta": GAIN_SURPRISE_BETA,
            "gain_reopen_max": GAIN_REOPEN_MAX,
            "gain_noise_gain": GAIN_NOISE_GAIN,
            "grid_size": GRID_SIZE,
            "num_hazards": N_HAZARDS,
            "num_resources": N_RESOURCES,
            "self_dim": SELF_DIM,
            "world_dim": WORLD_DIM,
        }
        base_cache = _base_for(seed, condition)
        with arm_cell(seed, config_slice=cfg_slice, script_path=Path(__file__),
                      extra_ineligible_reasons=[
                          "diagnostic_mechanism_probe_no_reuse",
                          "cell_depends_on_cached_p0_head"]) as cell:
            row = run_cell(arm, condition, seed, base_cache, scale,
                           n_cycles, wake_eps, steps)
            _agent = row.pop("agent")
            # RED-TEAM F8: write_flat_manifest only needs one agent per DISTINCT
            # arm configuration to record the config surface; handing it all 24
            # duplicated the same six configs 4x each.
            if arm not in agents_by_arm:
                agents_by_arm[arm] = _agent
                agents_seen.append(_agent)
            cell.stamp(row)
        rows[(arm, condition, seed)] = row
        arm_results.append(row)

    # ---- F7: --probe, the section-15 freeze artifact -----------------------
    if probe_only:
        _pseed = seeds[0]
        # BATCH 4: ARM_D_RESIDUAL joins the probe. It is no longer a
        # budget-matched control but a scheduler in its own right, so its
        # realised per-condition gain against C's IS one of the freeze numbers.
        for _arm in (ARM_B_STORE_ONLY, ARM_C_PROVENANCE, ARM_D_RESIDUAL):
            for _cond in CONDITIONS:
                _run_one(_arm, _cond, _pseed, 1.0)
        _print_probe(rows, _pseed, bases)
        print(f"\n[{EXPERIMENT_TYPE}] --probe complete; no manifest written.",
              flush=True)
        return "PASS", None

    # ---- G1 liveness, on the first seed's converged base -------------------
    # Reuses the cached P0 head, so the probe costs a few short P1s rather than
    # another 3600-step P0.
    live_base = _base_for(seeds[0], COND_CONVERGED_CLEAN)
    liveness = run_liveness(seeds[0], live_base, n_cycles, wake_eps, steps)
    _print_liveness_table(liveness)
    if liveness_only:
        print(f"\n[{EXPERIMENT_TYPE}] --liveness complete; no manifest written.",
              flush=True)
        return ("PASS" if liveness["strictly_increasing"] else "FAIL"), None

    for seed in seeds:
        for arm in ARMS_PHASE1:
            for condition in CONDITIONS:
                _run_one(arm, condition, seed, 1.0)
        # c_seed: ARM C's realised per-step gain, pooled over ALL conditions and
        # cycles for this seed, weighted by the number of e2_world steps in each
        # cycle. Read from the gain diagnostics, NEVER from any DV.
        num = 0.0
        den = 0.0
        for condition in CONDITIONS:
            r = rows.get((ARM_C_PROVENANCE, condition, seed))
            if r is None:
                continue
            for c in r["per_cycle"]:
                m = _finite_or_none(c["trace_e2_world"]["step_scale_mean"])
                n = _finite_or_none(c["trace_e2_world"]["n_steps"])
                if m is None or n is None or n <= 0.0:
                    continue
                num += m * n
                den += n
        c_seed = (num / den) if den > 0.0 else 1.0
        c_seed_by_seed[seed] = float(c_seed)
        print(f"[{EXPERIMENT_TYPE}] seed={seed} c_seed={c_seed:.6g} "
              f"(pooled over {den:.0f} ARM_C e2_world steps)", flush=True)
        # Only ARM_D_GLOBAL consumes c_seed now (BATCH 4).
        for arm in ARMS_PHASE2:
            for condition in CONDITIONS:
                _run_one(arm, condition, seed, c_seed)

    elapsed = time.time() - t0
    all_rows = [rows[(a, c, s)] for a in ARMS for c in CONDITIONS for s in seeds]
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{ts}_v3"
    aggregate = aggregate_arm_gates([r["gate"] for r in all_rows])
    unready = [f"{r['arm']}|{r['condition']}|seed{r['seed']}" for r in all_rows
               if not r["cell_ok"]]

    run_config = {
        "arms": list(ARMS),
        "arm_spec": {a: dict(ARM_SPEC[a]) for a in ARMS},
        "arm_run_order": {"phase1": list(ARMS_PHASE1),
                          "phase2": list(ARMS_PHASE2)},
        "arm_run_order_note": (
            "Only ARM_D_GLOBAL runs in phase 2, because only it consumes "
            "c_seed (ARM C's realised TOTAL budget). BATCH 4 moved "
            "ARM_D_RESIDUAL into phase 1: the substrate's residual_only mode "
            "now schedules per row from the CURRENT residual "
            "(g_i = clip(gain_max * sqrt(pe_cur_i / v_ref), gain_min, "
            "gain_max)) and ignores global_scale, so it has no matched budget "
            "to wait for. Its realised per-condition budget against C's is "
            "recorded as dr_budget_ratio_<cond>_s<seed>."),
        "conditions": list(CONDITIONS),
        "condition_spec": {c: dict(COND_SPEC[c]) for c in CONDITIONS},
        "seeds": list(seeds),
        "c_seed_by_seed": {str(k): v for k, v in c_seed_by_seed.items()},
        "p0_steps": p0_steps,
        "p0_episode_equiv": P0_EPISODE_EQUIV,
        "p0_trajectory_at": list(P0_TRAJECTORY_AT),
        "n_cycles": n_cycles,
        "wake_eps_per_cycle": wake_eps,
        "steps_per_episode": steps,
        "episodes_per_run": EPISODES_PER_RUN,
        "battery_size": battery,
        "battery_floor": BATTERY_FLOOR,
        "e2_lr": E2_LR,
        "batch_k": BATCH_K,
        "max_grad_norm": MAX_GRAD_NORM,
        "cmc_steps": CMC_STEPS,
        "cmc_lr": CMC_LR,
        "cmc_batch": CMC_BATCH,
        "cmc_schedule": "interleaved",
        "grid_size": GRID_SIZE,
        "num_hazards": N_HAZARDS,
        "num_resources": N_RESOURCES,
        "self_dim": SELF_DIM,
        "world_dim": WORLD_DIM,
        "world_buffer_floor": WORLD_BUFFER_FLOOR,
        "min_conv_rel_drop": MIN_CONV_REL_DROP,
        "max_unconv_rel_drop": MAX_UNCONV_REL_DROP,
        "seeds_required": req,
        "margin_absolute": MARGIN,
        "ret_c_ceiling": RET_C_CEILING,
        "obs_noise_sigma": OBS_NOISE_SIGMA,
        "calib_steps": calib_steps,
        "calibration_note": (
            "AMENDMENT 1. Before any measurement, a CLEAN (non-inverted, "
            "noise-free) waking window of calib_steps on env seed+2 -- a seed no "
            "condition's measurement phase uses -- teaches the SD-PP-1/2 EMAs "
            "this head's residual and this channel's observation noise. The "
            "state is cached with the P0 head and restored IDENTICALLY into "
            "every arm that has producers (ARM_A has none). Without it the "
            "full-budget probe measured pi_hist_min == 1/v_init == 100.0 in "
            "EVERY condition -- the estimator's cold start posing as historical "
            "precision. It cannot leak: it is clean, pre-test, on an unused env "
            "seed, and identical across arms, so it cannot differentiate them."),
        "calibration_state_by_base": {
            f"{bc['base']}_seed{bc['seed']}": {
                "calib_steps": bc.get("calib_steps"),
                "calib_env_seed": bc.get("calibration_state", {}).get(
                    "calib_env_seed"),
                "v_tot": bc.get("calib_v_tot"),
                "v_noise": bc.get("calib_v_noise"),
                "n_obs": bc.get("calib_n_obs"),
                "kappa": bc.get("calib_kappa"),
                "kappa_ready": bc.get("calib_kappa_ready"),
                "sigma_sq_ema": bc.get("calib_sigma_sq_ema"),
                "n_frames": bc.get("calib_n_frames"),
            } for bc in bases.values()},
        "gain_rule_constants": {
            "noise_gain": GAIN_NOISE_GAIN, "v_ref": GAIN_V_REF,
            "surprise_beta": GAIN_SURPRISE_BETA, "reopen_max": GAIN_REOPEN_MAX,
            "gain_min": GAIN_MIN, "gain_max": GAIN_MAX,
        },
        "producer_constants": {
            "obs_ema_alpha": OBS_EMA_ALPHA, "kappa_ema_alpha": KAPPA_EMA_ALPHA,
            "sigma_floor": SIGMA_FLOOR, "pe_ema_alpha": PE_EMA_ALPHA,
            "v_floor": V_FLOOR, "v_init": V_INIT,
            "precision_source": PRECISION_SOURCE,
        },
        "gate_thresholds": {
            "g3_noise_ratio_max": G3_NOISE_RATIO_MAX,
            "g3_shift_log2_max": G3_SHIFT_LOG2_MAX,
            "g4_surprise_high": G4_SURPRISE_HIGH,
            "g4_early_packets": G4_EARLY_PACKETS,
            "g4_early_frac_floor": G4_EARLY_FRAC_FLOOR,
            "g4_clean_median_max": G4_CLEAN_MEDIAN_MAX,
            "g5_pi_cur_ratio": G5_PI_CUR_RATIO,
            "mse_floor": MSE_FLOOR,
            "g9_pe_ratio": G9_PE_RATIO,
            "liveness_scales": list(LIVENESS_SCALES),
        },
        "sd056_rollout_clamp_exempt": SD056_ROLLOUT_CLAMP_EXEMPT,
        "anchor_reachability_exempt": ANCHOR_REACHABILITY_EXEMPT,
    }

    base_manifest: Dict[str, Any] = {
        "schema_version": "v1",
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "sleep_driver_pattern": (
            "manual-cycle-loop (force_cycle() called once per cycle in a "
            "dedicated N_CYCLES wake-sleep-test loop)"
        ),
        "timestamp_utc": ts,
        "started_utc": datetime.utcnow().isoformat() + "Z",
        "arm_results": arm_results,
        "per_arm_gate": aggregate["per_arm_gate"],
        "ethics_preflight": dict(ETHICS_PREFLIGHT),
        "liveness_probe": liveness,
        "arm_pairing_note": (
            "The agent cannot be deep-copied (torch non-leaf tensors), so arms "
            "are paired by RE-RUN FROM A CACHED P0 HEAD: per (seed, base) the "
            "batteries are captured and P0 is run once, agent.e2.state_dict() is "
            "cached, and every arm cell re-seeds identically, builds a fresh "
            "agent with that arm's flags, loads the cached head (strict=True) "
            "and skips P0. G2 asserts the resulting pre-sleep identity FROM "
            "OUTPUT -- world buffer hash, world-head parameter hash and the "
            "torch RNG state entering force_cycle -- rather than trusting the "
            "argument. A consequence, stated rather than hidden: an arm cell's "
            "agent carries no P0 SENSORY history, only the P0 weights. That is "
            "uniform across arms and is the correct waking regime for the "
            "SD-PP-1/2 estimators, which must warm up on P1."),
        "precision_source_note": (
            "use_e2_world_uncertainty is OFF in every arm (it is not set by "
            "_make_agent and defaults False), so SD-PP-2's sd063_or_ema source "
            "falls back to \"ema\" on EVERY packet: precision_source is "
            "\"ema\" throughout and pi_hist is the GLOBAL, state-blind EMA "
            "read, not a per-(z,a) one. G4's surviving leg therefore measures "
            "the EMA's TIME COURSE across the waking window rather than "
            "state-to-state variation in model precision, and no reading here "
            "may be described as per-state. The per-state upgrade is substrate "
            "necessity (f) (RED-TEAM F8)."),
        "letgo_r0_note": (
            "letgo_r0 is a DOCUMENTED ALIAS of retention_r0, not a second "
            "measurement: it is the same number under the name the "
            "preregistration uses for it in COND_CONFIDENTLY_WRONG, where the "
            "R0 battery holds the OLD rule and its retention is the "
            "letting-go reading. Do not sum or compare the two."),
        "scope_note": (
            "NO CLAIM VERDICT. experiment_purpose is diagnostic and "
            "evidence_direction is non_contributory on every route. PASS/FAIL "
            "turns ONLY on validity gates G1..G7; the contrasts P1..P7 route no "
            "verdict and select interpretation.label. " + EVIDENCE_LEVEL_CAP),
    }

    def _write(manifest: Dict[str, Any]) -> Optional[str]:
        if dry_run:
            print(f"[{EXPERIMENT_TYPE}] dry-run complete; not writing manifest.")
            return None
        out_path = write_flat_manifest(
            manifest, dry_run=False, config=manifest.get("config"),
            seeds=list(seeds), script_path=Path(__file__), agent=agents_seen)
        print(f"Result written to: {out_path}")
        return str(out_path)

    # ---- not-ready route ----------------------------------------------------
    if unready:
        diverged = [f"{r['arm']}|{r['condition']}|seed{r['seed']}"
                    for r in all_rows if not r["params_finite"]]
        reason = f"gate unmet in cell(s): {unready}"
        print(f"\n[{EXPERIMENT_TYPE}] -> substrate_not_ready_requeue: {reason}",
              flush=True)
        manifest = dict(base_manifest)
        manifest.update({
            "outcome": "FAIL",
            "result": "FAIL",
            "evidence_direction": "non_contributory",
            "evidence_direction_note": (
                "Diagnostic readiness unmet; NO contrast is read. A pattern read "
                "off an invalid measurement is worth less than no reading. This "
                "is not a MECH-572 finding in either direction. " + reason
                + " " + EVIDENCE_LEVEL_CAP),
            "non_degenerate": False,
            "degeneracy_reason": (
                "substrate_not_ready: " + reason + (
                    " NOTE: cell(s) " + ", ".join(diverged) + " carried a "
                    "NON-FINITE parameter after a sleep cycle -- that is "
                    "numerical DIVERGENCE, not merely an unmet readiness "
                    "precondition." if diverged else "")),
            "interpretation": {
                "label": "substrate_not_ready_requeue",
                "preconditions": aggregate["adjudication_preconditions"],
                "criteria_non_degenerate": {},
            },
            "readout": {"substrate_ready": 0, "overall_pass": 0,
                        "n_unready_cells": float(len(unready))},
            "config": run_config,
            "elapsed_seconds": elapsed,
        })
        return "FAIL", _write(manifest)

    # =======================================================================
    # GATES G1..G9 (preregistration section 9)
    # =======================================================================
    # G1 -- liveness ladder, measured above on the cached head.
    g1 = bool(liveness["strictly_increasing"])

    # G1b -- RED-TEAM F4: is the PROVENANCE CONSUMER live, as opposed to merely
    # the step-scale hook (which G1 tests with a constant)? ARM_C's realised
    # step_scale must actually depart from 1.0 somewhere, or the rule is
    # computing gains that all clip to unity and ARM_C is ARM_B with extra
    # arithmetic. Routes NO verdict; marks every provenance contrast unciteable.
    g1b_per_seed: List[float] = []
    for seed in seeds:
        devs = []
        for condition in CONDITIONS:
            r = rows.get((ARM_C_PROVENANCE, condition, seed))
            if r is None:
                continue
            v = _finite_or_none(r.get("gain_mean_over_cycles"))
            if v is not None:
                devs.append(abs(v - 1.0))
        g1b_per_seed.append(max(devs) if devs else float("nan"))
    g1b_worst = (min(_finite_vals(g1b_per_seed)) if _finite_vals(g1b_per_seed)
                 else float("nan"))
    g1b = bool(_finite_or_none(g1b_worst) is not None and g1b_worst > 1e-3)

    # G2 -- A/B (and all-six-arm) neutrality at cycle 1, BITWISE, tolerance 0.
    # Only cycle 1 is comparable: after the first sleep the arms legitimately
    # differ, so their cycle-2 pre-sleep states differ too. That is the
    # manipulation working, not a gate failure.
    g2_mismatches: List[str] = []
    for condition in CONDITIONS:
        for seed in seeds:
            ref = rows.get((ARMS[0], condition, seed))
            if ref is None:
                continue
            for arm in ARMS[1:]:
                r = rows.get((arm, condition, seed))
                if r is None:
                    continue
                for key in ("world_buffer_hash_cycle1",
                            "action_buffer_hash_cycle1",
                            "world_head_hash_cycle1",
                            "rng_state_hash_cycle1",
                            "rng_state_hash_post_cycle1"):
                    if r[key] != ref[key]:
                        g2_mismatches.append(
                            f"{arm}|{condition}|seed{seed}:{key}")
    g2 = bool(not g2_mismatches)

    # ---- channel readouts, taken from ARM_B (store-only) ------------------
    # ARM_B carries the producers AND is bit-identical to ARM_A in waking, so it
    # is the neutral place to read the channels. Cycle 1, the state every gate
    # is defined against.
    def _b_cycle1(condition: str, seed: int) -> Optional[Dict[str, Any]]:
        r = rows.get((ARM_B_STORE_ONLY, condition, seed))
        if r is None or not r["per_cycle"]:
            return None
        return r["per_cycle"][0]

    def _b_row(condition: str, seed: int) -> Optional[Dict[str, Any]]:
        return rows.get((ARM_B_STORE_ONLY, condition, seed))

    # G3 -- evidence precision varies with NOISE and not with the RULE SHIFT.
    # AMENDMENT 2: read the packet MEDIAN of evidence_variance_z, not the mean
    # of evidence_precision_z. Precision is 1/variance, so its MEAN is dominated
    # by whichever few frames had the smallest variance -- the cold-start ones.
    # The full-budget probe measured the channel separating 32x in variance
    # (3.4e-6 clean vs 1.1e-4 noisy) while the precision MEAN read only 0.66,
    # i.e. the gate was failing for a statistic reason, not a channel reason.
    # Leg 1 is the clean/noisy VARIANCE ratio, which is the same quantity as the
    # noisy/clean PRECISION ratio and takes the SAME UNCHANGED threshold.
    def _var_median(condition: str, seed: int) -> Optional[float]:
        c = _b_cycle1(condition, seed)
        if c is None:
            return None
        return _finite_or_none(
            c["packets_cumulative"].get("evidence_variance_z_median"))

    def _sigma_median(condition: str, seed: int) -> Optional[float]:
        c = _b_cycle1(condition, seed)
        if c is None:
            return None
        return _finite_or_none(
            c["packets_cumulative"].get("sigma_obs_median"))

    g3_noise_ratio: List[float] = []
    g3_shift_log2: List[float] = []
    g3_sigma_dev: List[float] = []
    g3_var_medians: Dict[str, List[float]] = {c: [] for c in CONDITIONS}
    for seed in seeds:
        v1 = _var_median(COND_CONVERGED_CLEAN, seed)
        v3 = _var_median(COND_CONFIDENTLY_WRONG, seed)
        v4 = _var_median(COND_NOISY_CONTRADICTION, seed)
        for cond in CONDITIONS:
            mv = _var_median(cond, seed)
            g3_var_medians[cond].append(
                mv if mv is not None else float("nan"))
        g3_noise_ratio.append(
            v1 / v4 if (v1 is not None and v4 is not None and v4 > 0.0)
            else float("nan"))
        g3_shift_log2.append(
            abs(math.log2(v3 / v1))
            if (v1 is not None and v3 is not None and v1 > 0.0 and v3 > 0.0)
            else float("nan"))
        # RED-TEAM F2, G3's new verdict-routing leg 2. The DIRECT assertion the
        # old rule-shift leg was reaching for: between two CLEAN regimes the
        # SENSORY statistic must be identical, and pinned at the instrument
        # floor. Exact equality is the right comparator because a clean
        # mostly-static field gives median|frame diff| == 0 exactly, so
        # sigma_obs is sigma_floor exactly. Unlike the old leg this cannot be
        # moved by kappa, which is a motion statistic and not a sensory one.
        s1 = _sigma_median(COND_CONVERGED_CLEAN, seed)
        s3 = _sigma_median(COND_CONFIDENTLY_WRONG, seed)
        g3_sigma_dev.append(
            max(abs(s1 - SIGMA_FLOOR), abs(s3 - SIGMA_FLOOR))
            if (s1 is not None and s3 is not None) else float("nan"))
    g3_noise_worst = (max(_finite_vals(g3_noise_ratio))
                      if _finite_vals(g3_noise_ratio) else float("nan"))
    g3_shift_worst = (max(_finite_vals(g3_shift_log2))
                      if _finite_vals(g3_shift_log2) else float("nan"))
    # BATCH 2 EDIT 1. Leg 1 (does evidence precision move with NOISE?) stays a
    # verdict-routing gate. Leg 2 (does it stay PUT under the rule shift?)
    # becomes G3b, routes_verdict False: the dry-run measured |log2| = 3.795
    # against a threshold of 1, and the whole gap is kappa, which SD-PP-1
    # measures on MOTION -- a permuted action map changes the motion statistics,
    # so the two clean conditions differ for a NON-SENSORY reason. That is a
    # scope limit on cond-3 contrasts, not a failure of the evidence channel,
    # whose noise separation is ~1000x and emphatic. kappa is deliberately NOT
    # frozen: its self-measurement under noise in cond 4 is the channel doing
    # its job, and freezing it would break leg 1 to rescue leg 2.
    g3_sigma_worst = (max(_finite_vals(g3_sigma_dev))
                      if _finite_vals(g3_sigma_dev) else float("nan"))
    g3_leg2 = bool(_finite_or_none(g3_sigma_worst) is not None
                   and g3_sigma_worst == 0.0)
    g3 = bool(_finite_or_none(g3_noise_worst) is not None
              and g3_noise_worst < G3_NOISE_RATIO_MAX
              and g3_leg2)
    g3b = bool(_finite_or_none(g3_shift_worst) is not None
               and g3_shift_worst <= G3_SHIFT_LOG2_MAX)

    # G4 -- historical precision varies, and the confidently-wrong case exists.
    pi_sds: List[float] = []
    for condition in CONDITIONS:
        for seed in seeds:
            c = _b_cycle1(condition, seed)
            if c is not None:
                pi_sds.append(_finite_or_none(
                    c["packets_cumulative"]["pi_hist_sd"]) or float("nan"))
    g4_pi_sd_min = (min(_finite_vals(pi_sds)) if _finite_vals(pi_sds)
                    else float("nan"))
    g4_early = _finite_vals(
        [(_b_row(COND_CONFIDENTLY_WRONG, s) or {}).get("early_surprise_frac")
         for s in seeds])
    g4_early_min = min(g4_early) if g4_early else float("nan")
    g4_clean_med = _finite_vals(
        [(_b_cycle1(COND_CONVERGED_CLEAN, s) or {}).get(
            "packets_cumulative", {}).get("surprise_median") for s in seeds])
    g4_clean_max = max(g4_clean_med) if g4_clean_med else float("nan")
    # AMENDMENT 3a. G4 now keeps ONLY the "pi_hist varies" leg as a
    # verdict-routing gate. Its former SURPRISE leg becomes G4b below, because
    # the full-budget probe showed that leg cannot be met on THIS head for a
    # reason that is a substrate fact rather than a channel failure: the head
    # sits near copy-the-input and barely reads the action (MECH-573 / B5), so
    # an action-map inversion is not a CONTRADICTION for it (inverted-rule
    # battery MSE 1.13e-5 vs 1.49e-5 on the original rule; surprise max 1.02).
    # Failing the whole run on it would report a substrate limit as a
    # measurement error, which is exactly the 785 confusion.
    g4 = bool(_finite_or_none(g4_pi_sd_min) is not None and g4_pi_sd_min > 0.0)
    g4b = bool(_finite_or_none(g4_early_min) is not None
               and g4_early_min >= G4_EARLY_FRAC_FLOOR
               and _finite_or_none(g4_clean_max) is not None
               and g4_clean_max < G4_CLEAN_MEDIAN_MAX)

    # G5 -- current model precision varies with the BASE.
    g5_ratios: List[float] = []
    for seed in seeds:
        c1 = _b_cycle1(COND_CONVERGED_CLEAN, seed)
        c2 = _b_cycle1(COND_UNDERFIT, seed)
        a = _finite_or_none(c1["pi_cur_at_sleep_entry"]) if c1 else None
        b = _finite_or_none(c2["pi_cur_at_sleep_entry"]) if c2 else None
        g5_ratios.append(a / b if a is not None and b is not None and b > 0.0
                         else float("nan"))
    g5_worst = (min(_finite_vals(g5_ratios)) if _finite_vals(g5_ratios)
                else float("nan"))
    g5 = bool(_finite_or_none(g5_worst) is not None and g5_worst > G5_PI_CUR_RATIO)

    # G6 -- the consolidator displaces in the BASELINE arm, every condition.
    a_rows = [rows[(ARM_A_BASELINE, c, s)] for c in CONDITIONS for s in seeds]
    g6_min_disp = min(_finite_vals([r["world_head_max_abs_delta"] for r in a_rows])
                      or [float("nan")])
    g6 = bool(_finite_or_none(g6_min_disp) is not None and g6_min_disp > 0.0)

    # G7 -- the readouts have variance and are not saturated.
    a_deltas = [r["r0_mse_delta_sum"] for r in a_rows]
    g7_min_abs_delta = min([abs(d) for d in _finite_vals(a_deltas)]
                           or [float("nan")])
    all_post = _finite_vals([c["r0_mse_post"] for r in all_rows
                             for c in r["per_cycle"]])
    all_post_present = all(
        _finite_or_none(c["r0_mse_post"]) is not None
        and _finite_or_none(c["r0_mse_pre"]) is not None
        for r in all_rows for c in r["per_cycle"])
    g7_min_post = min(all_post) if all_post else float("nan")
    g7 = bool(_finite_or_none(g7_min_abs_delta) is not None
              and g7_min_abs_delta > 0.0 and all_post_present
              and _finite_or_none(g7_min_post) is not None
              and g7_min_post >= MSE_FLOOR)

    # G8 -- MECH-573 readability of the converged base.
    g8_skills = _values(rows, seeds, COND_CONVERGED_CLEAN, ARM_A_BASELINE,
                        "skill_r0")
    g8_n = _count([_finite_or_none(v) is not None and v > 0.0 for v in g8_skills])
    g8 = bool(g8_n >= req)

    # G9 -- PE separation in the confidently-wrong condition.
    g9_flags: List[bool] = []
    g9_ratios: List[float] = []
    for seed in seeds:
        c3 = _b_cycle1(COND_CONFIDENTLY_WRONG, seed)
        c1 = _b_cycle1(COND_CONVERGED_CLEAN, seed)
        pe = _finite_or_none(c3["packets_cumulative"]["pe_mean"]) if c3 else None
        resid = _finite_or_none(c1["r0_mse_pre"]) if c1 else None
        ratio = (pe / resid if pe is not None and resid is not None and resid > 0.0
                 else float("nan"))
        g9_ratios.append(ratio)
        g9_flags.append(_finite_or_none(ratio) is not None and ratio > G9_PE_RATIO)
    g9_n = _count(g9_flags)
    g9 = bool(g9_n >= req)

    gates = [
        {"name": "G1_gain_manipulation_live", "load_bearing": True, "passed": g1,
         "measured": float(liveness["n_increasing_steps"]),
         "threshold": float(liveness["n_steps_required"]), "comparator": ">=",
         "routes_verdict": True,
         "detail": "world_head_max_abs_delta strictly increasing over global_scale"},
        {"name": "G2_arms_bit_identical_pre_sleep", "load_bearing": True,
         "passed": g2, "measured": float(len(g2_mismatches)), "threshold": 0.0,
         "comparator": "<=", "routes_verdict": True,
         "offending_cell": (g2_mismatches[0] if g2_mismatches else ""),
         "detail": ("world buffer + world-head params + torch RNG state entering "
                    "force_cycle, all six arms, cycle 1")},
        {"name": "G1b_provenance_mode_produces_non_unit_scale",
         "load_bearing": False, "passed": g1b, "measured": float(g1b_worst),
         "threshold": 1e-3, "comparator": ">", "routes_verdict": False,
         "per_seed_max_abs_dev_from_unity": [float(v) for v in g1b_per_seed],
         "detail": ("RED-TEAM F4. G1 shows the step-scale HOOK moves under a "
                    "constant; this shows the RULE produces a non-unit scale. "
                    "Routes NO verdict; when unmet it marks P2, P4, P5, P6 and "
                    "P7 non_degenerate=False (provenance consumer inert).")},
        {"name": "G3_evidence_precision_channel_varies", "load_bearing": True,
         "passed": g3, "measured": float(g3_noise_worst),
         "threshold": G3_NOISE_RATIO_MAX, "comparator": "<",
         "routes_verdict": True,
         "shift_log2_worst": float(g3_shift_worst),
         "shift_log2_threshold": G3_SHIFT_LOG2_MAX,
         "statistic": ("packet MEDIAN of evidence_variance_z (AMENDMENT 2); "
                       "var_median(cond1)/var_median(cond4), which is the "
                       "noisy/clean PRECISION ratio, same threshold"),
         "per_seed_noise_ratio": [float(v) for v in g3_noise_ratio],
         "per_seed_var_median_cond1": [
             float(v) for v in g3_var_medians[COND_CONVERGED_CLEAN]],
         "per_seed_var_median_cond4": [
             float(v) for v in g3_var_medians[COND_NOISY_CONTRADICTION]],
         "sigma_obs_max_dev_from_floor": float(g3_sigma_worst),
         "sigma_obs_floor": SIGMA_FLOOR,
         "leg2_sigma_obs_identical_at_floor": bool(g3_leg2),
         "per_seed_sigma_obs_dev": [float(v) for v in g3_sigma_dev],
         "detail": ("TWO verdict-routing legs (RED-TEAM F2): leg 1 = evidence "
                    "precision MOVES with noise; leg 2 = sigma_obs is IDENTICAL "
                    "and at the instrument floor between the two CLEAN regimes "
                    "(cond 1 and cond 3), which is the direct sensory assertion "
                    "the old rule-shift leg could not make. The kappa-driven "
                    "rule-shift leg is now the non-routing G3b.")},
        {"name": "G3b_evidence_precision_does_not_alias_rule_shift",
         "load_bearing": False, "passed": g3b, "measured": float(g3_shift_worst),
         "threshold": G3_SHIFT_LOG2_MAX, "comparator": "<=",
         "routes_verdict": False,
         "per_seed_shift_log2": [float(v) for v in g3_shift_log2],
         "per_seed_var_median_cond1": [
             float(v) for v in g3_var_medians[COND_CONVERGED_CLEAN]],
         "per_seed_var_median_cond3": [
             float(v) for v in g3_var_medians[COND_CONFIDENTLY_WRONG]],
         "detail": ("Routes NO verdict (BATCH 2 EDIT 1). When unmet it marks "
                    "P4, P6 and P7 non_degenerate=False and nothing else -- "
                    "only condition-3 contrasts are affected. See "
                    "gates.degeneracy_causes for the reason text (register "
                    "item B7).")},
        {"name": "G4_historical_precision_channel_varies", "load_bearing": True,
         "passed": g4, "measured": float(g4_pi_sd_min),
         "threshold": 0.0, "comparator": ">",
         "routes_verdict": True,
         "detail": ("within-run SD of pi_hist over packets, worst condition. "
                    "The surprise leg moved to G4b (AMENDMENT 3a).")},
        {"name": "G4b_confidently_wrong_is_a_contradiction", "load_bearing": False,
         "passed": g4b, "measured": float(g4_early_min),
         "threshold": G4_EARLY_FRAC_FLOOR, "comparator": ">=",
         "routes_verdict": False,
         "pi_hist_sd_min": float(g4_pi_sd_min),
         "clean_surprise_median_max": float(g4_clean_max),
         "clean_surprise_median_threshold": G4_CLEAN_MEDIAN_MAX,
         "detail": ("Routes NO verdict (AMENDMENT 3a). With G9 it decides "
                    "whether COND_CONFIDENTLY_WRONG posed a contradiction AT "
                    "ALL on this head; when unmet it marks P4, P7 and P6's "
                    "correction leg non_degenerate=False. The cond-3 cells are "
                    "still RUN and fully RECORDED across every seed -- that "
                    "record IS the instrument measurement that B5 is real.")},
        {"name": "G5_current_precision_channel_varies", "load_bearing": True,
         "passed": g5, "measured": float(g5_worst), "threshold": G5_PI_CUR_RATIO,
         "comparator": ">", "routes_verdict": True,
         "per_seed_ratio": [float(v) for v in g5_ratios]},
        {"name": "G6_consolidator_displaces_baseline", "load_bearing": True,
         "passed": g6, "measured": float(g6_min_disp), "threshold": 0.0,
         "comparator": ">", "routes_verdict": True},
        {"name": "G7_readouts_have_variance", "load_bearing": True, "passed": g7,
         "measured": float(g7_min_abs_delta), "threshold": 0.0, "comparator": ">",
         "routes_verdict": True,
         "min_post_sleep_mse": float(g7_min_post), "mse_floor": MSE_FLOOR},
        {"name": "G8_converged_base_readable", "load_bearing": False, "passed": g8,
         "measured": float(g8_n), "threshold": float(req), "comparator": ">=",
         "routes_verdict": False,
         "per_seed_skill": [float(v) for v in g8_skills],
         "detail": ("MECH-573 readability. Routes NO verdict: when unmet it marks "
                    "the retention contrasts non_degenerate=False and the run "
                    "continues for COND_UNDERFIT and the mechanistic readouts.")},
        {"name": "G9_pe_separation_confidently_wrong", "load_bearing": False,
         "passed": g9, "measured": float(g9_n), "threshold": float(req),
         "comparator": ">=", "routes_verdict": False,
         "per_seed_ratio": [float(v) for v in g9_ratios],
         "detail": ("B5. Routes NO verdict: when unmet COND_CONFIDENTLY_WRONG is "
                    "marked non_degenerate=False.")},
    ]

    validity_pass = all(g["passed"] for g in gates if g["routes_verdict"])

    # ---- RED-TEAM F6: the D-arm budget is matched in TOTAL, not per condition
    # (that is the design), so a P6 split must be read against how far the
    # realised budget in THAT condition departs from C's.
    budget_ratio: Dict[str, float] = {}
    for seed in seeds:
        c_s = _finite_or_none(c_seed_by_seed.get(seed))
        for condition in CONDITIONS:
            r = rows.get((ARM_C_PROVENANCE, condition, seed))
            v = _finite_or_none(r.get("gain_mean_over_cycles")) if r else None
            budget_ratio[f"{condition}_s{seed}"] = (
                c_s / v if (c_s is not None and v is not None and v > 0.0)
                else float("nan"))

    # BATCH 4: ARM_D_RESIDUAL is NOT budget-matched -- it schedules from the
    # CURRENT per-row residual and ignores global_scale -- so the quantity worth
    # recording for it is how its realised per-condition budget compares with
    # C's. P5 (cond 4) must be read against this the same way P6 is read against
    # p6_budget_ratio: it says whether Dr and C were running at comparable gain
    # WHERE THE CONTRAST IS TAKEN.
    dr_budget_ratio: Dict[str, float] = {}
    for seed in seeds:
        for condition in CONDITIONS:
            rc = rows.get((ARM_C_PROVENANCE, condition, seed))
            rd = rows.get((ARM_D_RESIDUAL, condition, seed))
            vc = _finite_or_none(rc.get("gain_mean_over_cycles")) if rc else None
            vd = _finite_or_none(rd.get("gain_mean_over_cycles")) if rd else None
            dr_budget_ratio[f"{condition}_s{seed}"] = (
                vd / vc if (vc is not None and vd is not None and vc > 0.0)
                else float("nan"))

    # =======================================================================
    # CONTRASTS P1..P7 (preregistration section 11). They route NO PASS/FAIL.
    # =======================================================================
    # P1 -- A vs B equivalence. Tolerance ZERO: the readout difference must be
    # exactly 0 AND the G2 hashes must match. This is guaranteed IF the recorder
    # is RNG-free, which is exactly what it tests (red-team 11).
    p1_deltas: List[float] = []
    for condition in CONDITIONS:
        p1_deltas.extend(_paired(rows, seeds, condition, ARM_A_BASELINE,
                                 ARM_B_STORE_ONLY, "retention_r0"))
    p1_worst = max([abs(d) for d in _finite_vals(p1_deltas)] or [float("nan")])
    p1_pass = bool(_finite_or_none(p1_worst) is not None and p1_worst == 0.0 and g2)

    # P2 -- protection on the converged-clean base.
    d_p2 = _paired(rows, seeds, COND_CONVERGED_CLEAN, ARM_C_PROVENANCE,
                   ARM_B_STORE_ONLY, "retention_r0")
    m_p2, sd_p2, src_p2 = _effective_margin(d_p2)
    # RED-TEAM F1. P2 is now the RELATIVE leg alone: ret(C) < ret(B) - margin,
    # i.e. protection MEASURED AGAINST THE BASELINE. The absolute
    # "rise within 10% of residual" leg is not dropped -- it becomes P2b, a
    # non-routing, non-load-bearing REPORT -- because V3-EXQ-1063's landed
    # seed-42 numbers (pre 1.4868e-5, 3-cycle mse_delta_sum -7.142e-5) put
    # ret(B) near 4.8 rather than the ~1.0 the preregistration assumed, and at
    # the realised cond-1 gain (~0.39) that ceiling is UNREACHABLE. A leg that
    # cannot be met at the pre-registered operating point would have silently
    # failed P2 for a scaling reason and sent the matrix to R2/mixed. NO GAIN
    # CONSTANT WAS CHANGED; the liveness attainability curve records what each
    # budget actually buys.
    f_p2 = [_finite_or_none(d) is not None and d < -m_p2 for d in d_p2]
    ret_c_c1 = _values(rows, seeds, COND_CONVERGED_CLEAN, ARM_C_PROVENANCE,
                       "retention_r0")
    ret_b_c1 = _values(rows, seeds, COND_CONVERGED_CLEAN, ARM_B_STORE_ONLY,
                       "retention_r0")
    f_p2b = [_finite_or_none(v) is not None and v < RET_C_CEILING
             for v in ret_c_c1]

    # P3 -- still learns on the underfit base.
    d_p3 = _paired(rows, seeds, COND_UNDERFIT, ARM_C_PROVENANCE,
                   ARM_B_STORE_ONLY, "retention_r0")
    m_p3, sd_p3, src_p3 = _effective_margin(d_p3)
    f_p3 = [_finite_or_none(d) is not None and d <= m_p3 for d in d_p3]

    # P4 -- anti-self-sealing.
    d_p4 = _paired(rows, seeds, COND_CONFIDENTLY_WRONG, ARM_C_PROVENANCE,
                   ARM_B_STORE_ONLY, "correction_r1")
    m_p4, sd_p4, src_p4 = _effective_margin(d_p4)
    corr_c_c3 = _values(rows, seeds, COND_CONFIDENTLY_WRONG, ARM_C_PROVENANCE,
                        "correction_r1")
    f_p4a = [_finite_or_none(d) is not None and d >= -m_p4 for d in d_p4]
    f_p4b = [_finite_or_none(v) is not None and v > 0.0 for v in corr_c_c3]
    f_p4 = [a and b for a, b in zip(f_p4a, f_p4b)]

    # P5 -- evidence precision under noise.
    d_p5 = _paired(rows, seeds, COND_NOISY_CONTRADICTION, ARM_C_PROVENANCE,
                   ARM_D_RESIDUAL, "retention_r0")
    m_p5, sd_p5, src_p5 = _effective_margin(d_p5)
    f_p5 = [_finite_or_none(d) is not None and d < -m_p5 for d in d_p5]

    # P6 -- beyond a generic reduction. The per-seed conjunction IS the test.
    d_p6a = _paired(rows, seeds, COND_CONVERGED_CLEAN, ARM_C_PROVENANCE,
                    ARM_D_GLOBAL, "retention_r0")
    d_p6b = _paired(rows, seeds, COND_CONFIDENTLY_WRONG, ARM_C_PROVENANCE,
                    ARM_D_GLOBAL, "correction_r1")
    m_p6a, sd_p6a, src_p6a = _effective_margin(d_p6a)
    m_p6b, sd_p6b, src_p6b = _effective_margin(d_p6b)
    f_p6 = [(_finite_or_none(a) is not None and a < -m_p6a
             and _finite_or_none(b) is not None and b > m_p6b)
            for a, b in zip(d_p6a, d_p6b)]

    # P7 -- is historical precision load-bearing (intake F1)?
    d_p7 = _paired(rows, seeds, COND_CONFIDENTLY_WRONG, ARM_C_PROVENANCE,
                   ARM_C_NOHIST, "correction_r1")
    m_p7, sd_p7, src_p7 = _effective_margin(d_p7)
    f_p7 = [_finite_or_none(d) is not None and d > m_p7 for d in d_p7]

    contrasts = [
        {"name": "P1_A_vs_B_equivalence", "description":
         ("storage neutrality: recording provenance must change NOTHING. "
          "Tolerance 0 on the readout AND bitwise hash equality (G2)."),
         "load_bearing": False, "routes_verdict": False, "passed": p1_pass,
         "measured": float(p1_worst), "threshold": 0.0, "comparator": "<=",
         "per_seed_delta": [float(d) for d in p1_deltas],
         "hashes_equal": g2, "n_hash_mismatches": float(len(g2_mismatches)),
         "margin_used": 0.0, "margin_absolute": 0.0, "margin_sd_1": 0.0,
         "margin_source": "exact_zero"},
        _contrast_record(
            "P2_provenance_protects_converged",
            ("cond 1: ret(C) < ret(B) - MARGIN -- protection measured AGAINST "
             "THE BASELINE. The absolute 10%-of-residual leg is reported "
             "separately as P2b and routes nothing (RED-TEAM F1)."),
            d_p2, f_p2, seeds, req, m_p2, sd_p2, src_p2, True,
            {"ret_c_per_seed": [float(v) for v in ret_c_c1],
             "ret_b_per_seed": [float(v) for v in ret_b_c1],
             "absolute_ceiling_leg_moved_to": "P2b_mech572_rise_within_10pct_of_residual"}),
        {"name": "P2b_mech572_rise_within_10pct_of_residual",
         "description": (
             "REPORT ONLY (RED-TEAM F1): ret(C) < 0.10 in cond 1, the "
             "preregistration's absolute 'post-sleep error rise at most 10% of "
             "residual' reading. It routes NOTHING and feeds NO label, because "
             "V3-EXQ-1063's landed baseline puts ret(B) near 4.8 rather than "
             "~1.0, so at the realised cond-1 gain this ceiling is not "
             "attainable and its failure would carry no information about "
             "provenance. Read it against liveness.attainability."),
         "load_bearing": False, "routes_verdict": False,
         "passed": bool(_count(f_p2b) >= req),
         "measured": float(_count(f_p2b)), "threshold": float(req),
         "comparator": ">=",
         "per_seed_value": [float(v) for v in ret_c_c1],
         "per_seed_flag": [bool(f) for f in f_p2b],
         "ret_c_ceiling": RET_C_CEILING,
         "ret_b_per_seed": [float(v) for v in ret_b_c1],
         "margin_used": 0.0, "margin_absolute": RET_C_CEILING,
         "margin_sd_1": 0.0, "margin_source": "absolute_ceiling_report_only"},
        _contrast_record(
            "P3_provenance_still_learns_underfit",
            ("cond 2: ret(C) <= ret(B) + MARGIN -- C is not worse than baseline "
             "learning on the underfit base (the anti-freezing check)."),
            d_p3, f_p3, seeds, req, m_p3, sd_p3, src_p3, False),
        _contrast_record(
            "P4_anti_self_sealing_confidently_wrong",
            ("cond 3: corr(C) >= corr(B) - MARGIN AND corr(C) > 0. Failure of "
             "EITHER is self-sealing (matrix R3) regardless of P2."),
            d_p4, f_p4, seeds, req, m_p4, sd_p4, src_p4, True,
            {"sub_flag_not_worse_than_baseline": [bool(f) for f in f_p4a],
             "sub_flag_corrects_at_all": [bool(f) for f in f_p4b],
             "corr_c_per_seed": [float(v) for v in corr_c_c3]}),
        _contrast_record(
            "P5_evidence_precision_ignores_noise",
            ("cond 4: ret(C) < ret(Dr) - MARGIN -- C ignores the un-learnable "
             "noise while a residual-only schedule learns it. Dr uses the "
             "CURRENT RESIDUAL ONLY; C additionally uses evidence precision "
             "(the Kalman weight K), the STORED innovation and historical "
             "surprise (r). So this is the intake's F2 question directly: does "
             "evidence precision add anything beyond raw error? Dr is NOT "
             "budget-matched (BATCH 4) -- read it against "
             "dr_budget_ratio_<cond>_s<seed>."),
            d_p5, f_p5, seeds, req, m_p5, sd_p5, src_p5, False,
            {"dr_budget_ratio_by_condition_seed": dict(dr_budget_ratio),
             "dr_note": (
                 "ARM_D_RESIDUAL schedules g_i = clip(gain_max * "
                 "sqrt(pe_cur_i / v_ref), gain_min, gain_max) from the CURRENT "
                 "per-row MSE of e2.world_forward on the replayed triple, with "
                 "K = 1 and r = 1 and no stored packet term. It ignores "
                 "global_scale and needs no c_seed, so it runs in phase 1.")}),
        _contrast_record(
            "P6_beyond_generic_gain_reduction",
            ("C beats the budget-matched uniform reduction Dg on PROTECTION "
             "(cond 1) AND CORRECTION (cond 3) simultaneously, per seed. Dg is "
             "expected to pass P2 and fail P4; that split is the whole point of "
             "the matched budget."),
            d_p6a, f_p6, seeds, req, m_p6a, sd_p6a, src_p6a, True,
            {"budget_ratio_by_condition_seed": dict(budget_ratio),
             "budget_note": (
                 "RED-TEAM F6. c_seed matches Dg's budget to C's TOTAL across "
                 "all conditions and cycles, deliberately NOT per condition -- "
                 "the rival hypothesis is a single global reduction. So in any "
                 "GIVEN condition Dg runs at c_seed / (C's realised gain in "
                 "that condition) times C's gain; those ratios are listed in "
                 "budget_ratio_by_condition_seed and in the flat readout as "
                 "p6_budget_ratio_<cond>_s<seed>. A P6 split MUST be read "
                 "against them: a ratio far from 1 in cond 1 or cond 3 means "
                 "the two arms were not running at comparable gain WHERE THE "
                 "CONTRAST IS TAKEN, even though their totals match."),
             "cond1_ret_delta_per_seed": [float(d) for d in d_p6a],
             "cond3_corr_delta_per_seed": [float(d) for d in d_p6b],
             "cond1_margin_used": float(m_p6a),
             "cond3_margin_used": float(m_p6b),
             "cond3_margin_sd_1": float(sd_p6b),
             "cond3_margin_source": src_p6b}),
        _contrast_record(
            "P7_historical_precision_load_bearing",
            ("cond 3: corr(C) > corr(C0) + MARGIN. A NULL HERE IS A RESULT -- "
             "historical precision not shown to add information (intake F1)."),
            d_p7, f_p7, seeds, req, m_p7, sd_p7, src_p7, False),
    ]

    p2 = bool(contrasts[1]["passed"])
    p3 = bool(contrasts[2]["passed"])
    p4 = bool(contrasts[3]["passed"])
    p5 = bool(contrasts[4]["passed"])
    p6 = bool(contrasts[5]["passed"])
    p7 = bool(contrasts[6]["passed"])

    # ---- routing ----------------------------------------------------------
    outcome = "PASS" if validity_pass else "FAIL"
    # AMENDMENT 3b + BATCH 2 EDIT 1: COND_CONFIDENTLY_WRONG is readable only if
    # it actually posed a contradiction (G4b), its waking PE separated from the
    # converged residual (G9), AND the evidence channel did not itself shift
    # with the rule (G3b). None of the three routes a verdict; together they
    # decide whether the contrasts BUILT on that condition can be read.
    cond3_readable = bool(g4b and g9 and g3b)
    if validity_pass:
        label = _label_from_matrix(g1, g3, g4, g5, g6, cond3_readable,
                                   bool(g8), p2, p3, p4, p5, p6)
    elif not (g1 and g6):
        label = "instrument_inert_substrate_not_ready"            # matrix R5
    elif not (g3 and g4 and g5):
        label = "precision_channel_degenerate"                    # matrix R6
    elif not g2:
        label = "substrate_not_ready_requeue"
    else:
        label = "provenance_gain_measurement_invalid"             # G7 alone

    criteria_non_degenerate: Dict[str, bool] = {g["name"]: bool(g["passed"])
                                                for g in gates}
    criteria_non_degenerate.update({
        "P1_A_vs_B_equivalence": True,
        # G8 gates every RETENTION contrast read on a CONVERGED base (conds 1/3/4);
        # COND_UNDERFIT (P3) is a separate base and is unaffected.
        "P2_provenance_protects_converged": bool(g8 and g1b),
        "P2b_mech572_rise_within_10pct_of_residual": bool(g8 and g1b),
        "P3_provenance_still_learns_underfit": bool(g1b),
        # G4b + G9 gate every COND_CONFIDENTLY_WRONG contrast (AMENDMENT 3b).
        "P4_anti_self_sealing_confidently_wrong": bool(cond3_readable and g1b),
        "P5_evidence_precision_ignores_noise": bool(g8 and g1b),
        # P6 needs BOTH legs: its protection leg is cond 1 (G8) and its
        # correction leg is cond 3 (G4b/G9).
        "P6_beyond_generic_gain_reduction": bool(g8 and cond3_readable and g1b),
        "P7_historical_precision_load_bearing": bool(cond3_readable and g1b),
    })
    # Why each marked contrast is unciteable -- printed and carried, so a reader
    # never has to reconstruct it from the gate table.
    degeneracy_causes: Dict[str, str] = {}
    for _name, _ok in criteria_non_degenerate.items():
        if _ok or not _name.startswith("P"):
            continue
        _why = []
        if not g1b:
            _why.append(
                "provenance consumer inert: ARM_C's realised step_scale never "
                f"departed from 1.0 by more than {g1b_worst:.4g} in any "
                f"condition (G1b, per seed "
                f"{['%.4g' % v for v in g1b_per_seed]})")
        if _name in ("P2_provenance_protects_converged",
                     "P2b_mech572_rise_within_10pct_of_residual",
                     "P5_evidence_precision_ignores_noise",
                     "P6_beyond_generic_gain_reduction") and not g8:
            _why.append("G8 converged-base skill <= 0 (MECH-573 readability): "
                        f"per-seed skill {['%.4g' % v for v in g8_skills]}")
        if _name in ("P4_anti_self_sealing_confidently_wrong",
                     "P6_beyond_generic_gain_reduction",
                     "P7_historical_precision_load_bearing"):
            if not (g4b and g9):
                _why.append(
                    "COND_CONFIDENTLY_WRONG posed no contradiction on this head "
                    f"(G4b early surprise frac {g4_early_min:.4g} vs "
                    f">= {G4_EARLY_FRAC_FLOOR}; G9 pe/residual "
                    f"{['%.4g' % v for v in g9_ratios]} vs > {G9_PE_RATIO}) -- "
                    "substrate necessity B5, not a null")
            if not g3b:
                _why.append(
                    "SD-PP-1 kappa is measured on motion and a permuted action "
                    "map changes the motion statistics, so evidence precision "
                    "differs between the clean and inverted-rule conditions for "
                    "a non-sensory reason (register item B7); only condition-3 "
                    "contrasts are affected"
                    f" [G3b |log2| {g3_shift_worst:.4g} vs "
                    f"<= {G3_SHIFT_LOG2_MAX}, per seed "
                    f"{['%.4g' % v for v in g3_shift_log2]}]")
        degeneracy_causes[_name] = "; ".join(_why) or "unspecified"

    # ---- printing ---------------------------------------------------------
    print(f"\n[{EXPERIMENT_TYPE}] GATES (G1..G7 route PASS/FAIL):", flush=True)
    _gate_std = {"name", "load_bearing", "passed", "measured", "threshold",
                 "comparator", "routes_verdict", "detail", "offending_cell"}
    for g in gates:
        print(f"  {g['name']}: passed={g['passed']} "
              f"measured={g['measured']:.6g} {g['comparator']} "
              f"{g['threshold']:.6g} routes_verdict={g['routes_verdict']}",
              flush=True)
        # Several gates are CONJUNCTIONS (G3 has a second leg, G4 two more, G7
        # a floor check). Printing only `measured` hides the leg that actually
        # failed, so every extra scalar and per-seed vector is printed too.
        extra = {k: v for k, v in g.items() if k not in _gate_std}
        for key in sorted(extra):
            val = extra[key]
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                print(f"      {key}={float(val):.6g}", flush=True)
            elif isinstance(val, list):
                print(f"      {key}={['%.4g' % float(x) for x in val]}",
                      flush=True)
    print(f"[{EXPERIMENT_TYPE}] CONTRASTS (P1..P7 route NO PASS/FAIL):", flush=True)
    for c in contrasts:
        # P2b reports a per-seed VALUE, not a paired delta (RED-TEAM F1).
        per_seed = c.get("per_seed_delta", c.get("per_seed_value", []))
        print(f"  {c['name']}: passed={c['passed']} "
              f"{int(c['measured'])}/{len(per_seed) or len(seeds)} "
              f"threshold={c['threshold']:.0f} "
              f"margin={c['margin_used']:.6g} ({c['margin_source']}) "
              f"mean={_mean(per_seed):.6g} sd={_sd(per_seed):.6g} "
              f"per_seed={['%.4g' % d for d in per_seed]} "
              f"non_degenerate={criteria_non_degenerate.get(c['name'], True)}",
              flush=True)
    if degeneracy_causes:
        print(f"[{EXPERIMENT_TYPE}] CONTRASTS MARKED NON-DEGENERATE=False "
              f"(unciteable in EITHER direction, and NOT a null):", flush=True)
        for _name in sorted(degeneracy_causes):
            print(f"  {_name}: {degeneracy_causes[_name]}", flush=True)
    else:
        print(f"[{EXPERIMENT_TYPE}] no contrast marked non-degenerate.",
              flush=True)
    print(f"  cond3_readable={cond3_readable} "
          f"(G3b={g3b} G4b={g4b} G9={g9}); "
          f"converged_base_readable={g8}", flush=True)
    print(f"  -> {label} ({outcome}); elapsed={elapsed:.1f}s", flush=True)

    flat: Dict[str, float] = {
        "g1_gain_manipulation_live": int(g1),
        "g2_arms_bit_identical_pre_sleep": int(g2),
        "g3_evidence_precision_channel_varies": int(g3),
        "g4_historical_precision_channel_varies": int(g4),
        "g1b_provenance_mode_produces_non_unit_scale": int(g1b),
        "g1b_worst_abs_dev_from_unity": float(g1b_worst),
        "g3b_evidence_precision_does_not_alias_rule_shift": int(g3b),
        "g3_sigma_obs_max_dev_from_floor": float(g3_sigma_worst),
        "p2b_mech572_rise_within_10pct_of_residual": int(
            _count(f_p2b) >= req),
        "g4b_confidently_wrong_is_a_contradiction": int(g4b),
        "cond3_readable": int(cond3_readable),
        "g5_current_precision_channel_varies": int(g5),
        "g6_consolidator_displaces_baseline": int(g6),
        "g7_readouts_have_variance": int(g7),
        "g8_converged_base_readable": int(g8),
        "g9_pe_separation_confidently_wrong": int(g9),
        "overall_pass": int(validity_pass),
        "p1_a_vs_b_equivalence": int(p1_pass),
        "p2_provenance_protects_converged": int(p2),
        "p3_provenance_still_learns_underfit": int(p3),
        "p4_anti_self_sealing": int(p4),
        "p5_evidence_precision_ignores_noise": int(p5),
        "p6_beyond_generic_gain_reduction": int(p6),
        "p7_historical_precision_load_bearing": int(p7),
        "seeds_required": float(req),
        "n_seeds": float(len(seeds)),
        "liveness_n_increasing_steps": float(liveness["n_increasing_steps"]),
        "n_g2_hash_mismatches": float(len(g2_mismatches)),
        "g3_noise_ratio_worst": float(g3_noise_worst),
        "g3_shift_log2_worst": float(g3_shift_worst),
        "g4_pi_hist_sd_min": float(g4_pi_sd_min),
        "g4_early_surprise_frac_min": float(g4_early_min),
        "g4_clean_surprise_median_max": float(g4_clean_max),
        "g5_pi_cur_ratio_worst": float(g5_worst),
        "g6_min_world_head_delta_baseline": float(g6_min_disp),
        "g7_min_abs_mse_delta_baseline": float(g7_min_abs_delta),
        "g7_min_post_sleep_mse": float(g7_min_post),
        "adam_step_bound_8x_lr": float(CMC_STEPS * CMC_LR),
        "min_identity_predictor_mse_r0": float(
            min(_finite_vals([r["identity_predictor_mse_r0"] for r in all_rows])
                or [float("nan")])),
        "max_world_head_delta_any_arm": float(
            max(_finite_vals([r["world_head_max_abs_delta"] for r in all_rows])
                or [float("nan")])),
        "mean_c_seed": _mean(list(c_seed_by_seed.values())),
        "calib_steps": float(calib_steps),
        "n_contrasts_marked_degenerate": float(len(degeneracy_causes)),
    }
    # RED-TEAM F4: the rule's three factors, per cell, in the FLAT readout so
    # the indexer and a later reader can see WHICH factor moved.
    for _arm in GAIN_ARMS:
        for _cond in CONDITIONS:
            for _seed in seeds:
                _r = rows.get((_arm, _cond, _seed))
                if _r is None:
                    continue
                _tag = f"__{_arm}__{_cond}__s{_seed}"
                _gd = _r.get("gain_diagnostics", {})
                for _short, _key in (("kmean", "k_mean"), ("mmean", "m_mean"),
                                     ("rmean", "r_mean"), ("rmax", "r_max"),
                                     ("gmin", "gain_min"), ("gmax", "gain_max")):
                    _v = _finite_or_none(_gd.get(_key))
                    if _v is not None:
                        flat[f"gain_{_short}{_tag}"] = _v
                _v = _finite_or_none(_r.get("gain_mean_over_cycles"))
                if _v is not None:
                    flat[f"step_scale_mean{_tag}"] = _v
    # RED-TEAM F6: how far the matched-in-TOTAL budget departs per condition.
    for _k, _v in budget_ratio.items():
        _fv = _finite_or_none(_v)
        if _fv is not None:
            flat[f"p6_budget_ratio_{_k}"] = _fv
    # BATCH 4: the same reading for the current-residual scheduler.
    for _k, _v in dr_budget_ratio.items():
        _fv = _finite_or_none(_v)
        if _fv is not None:
            flat[f"dr_budget_ratio_{_k}"] = _fv
    for _bkey, _bc in sorted(bases.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        _tag = f"{_bc['base']}_seed{_bc['seed']}"
        for _f in ("calib_v_tot", "calib_v_noise", "calib_kappa",
                   "calib_sigma_sq_ema"):
            _v = _finite_or_none(_bc.get(_f))
            if _v is not None:
                flat[f"{_f}_{_tag}"] = _v
    for name, deltas in (
            ("p2_cond1_ret_delta", d_p2), ("p3_cond2_ret_delta", d_p3),
            ("p4_cond3_corr_delta", d_p4), ("p5_cond4_ret_delta", d_p5),
            ("p6_cond1_ret_delta", d_p6a), ("p6_cond3_corr_delta", d_p6b),
            ("p7_cond3_corr_delta", d_p7)):
        m, s = _finite_or_none(_mean(deltas)), _finite_or_none(_sd(deltas))
        if m is not None:
            flat[f"{name}_mean"] = m
        if s is not None:
            flat[f"{name}_sd"] = s
    for arm in ARMS:
        for condition in CONDITIONS:
            vals = _values(rows, seeds, condition, arm, "retention_r0")
            m = _finite_or_none(_mean(vals))
            if m is not None:
                flat[f"ret_mean_{arm.lower()}_{condition.lower()}"] = m
    for arm in ARMS:
        vals = _values(rows, seeds, COND_CONFIDENTLY_WRONG, arm, "correction_r1")
        m = _finite_or_none(_mean(vals))
        if m is not None:
            flat[f"corr_mean_{arm.lower()}_cond3"] = m

    note = (
        f"MECHANISM PATTERN (routes no claim verdict). Validity gates G1..G7 "
        f"{'held' if validity_pass else 'FAILED'}: the gain manipulation was "
        f"{'live' if g1 else 'INERT'} (liveness ladder "
        f"{liveness['n_increasing_steps']:.0f}/"
        f"{liveness['n_steps_required']:.0f} rungs increasing), the six arms were "
        f"{'bitwise identical' if g2 else 'NOT identical'} entering cycle 1, and "
        f"the three precision channels "
        f"{'all varied as designed' if (g3 and g4 and g5) else 'did NOT all vary'} "
        f"(evidence cond4/cond1 {g3_noise_worst:.4g} vs < {G3_NOISE_RATIO_MAX}; "
        f"pi_cur cond1/cond2 {g5_worst:.4g} vs > {G5_PI_CUR_RATIO}). "
        f"Contrasts: P2 protection {contrasts[1]['measured']:.0f}/{len(seeds)}, "
        f"P3 underfit-learning {contrasts[2]['measured']:.0f}/{len(seeds)}, "
        f"P4 anti-self-sealing {contrasts[3]['measured']:.0f}/{len(seeds)}, "
        f"P5 noise-ignoring {contrasts[4]['measured']:.0f}/{len(seeds)}, "
        f"P6 beyond-generic {contrasts[5]['measured']:.0f}/{len(seeds)}, "
        f"P7 historical-precision {contrasts[6]['measured']:.0f}/{len(seeds)}, "
        f"each against a {req}-of-{len(seeds)} pre-registered sign count. "
        f"BUDGET SKEW (read P6 against this): c_seed matches Dg to C's TOTAL "
        f"gain, not per condition, and the realised per-condition ratios "
        f"c_seed/C-gain were "
        + ", ".join(f"{k} {v:.3g}" for k, v in sorted(budget_ratio.items())
                    if _finite_or_none(v) is not None)
        + ". ARM_D_RESIDUAL is NOT budget-matched (it schedules from the "
          "CURRENT residual); its realised gain against C's was "
        + ", ".join(f"{k} {v:.3g}" for k, v in sorted(dr_budget_ratio.items())
                    if _finite_or_none(v) is not None)
        + f". Pattern: {label}."
    ) if validity_pass else (
        f"Validity gates failed: "
        f"{[g['name'] for g in gates if g['routes_verdict'] and not g['passed']]}. "
        f"NO contrast reading is emitted -- see gates[] for measured/threshold "
        f"detail."
    )

    manifest = dict(base_manifest)
    manifest.update({
        "outcome": outcome,
        "result": outcome,
        "evidence_direction": "non_contributory",
        "evidence_direction_note": (
            "NON_CONTRIBUTORY BY DESIGN, not by failure. experiment_purpose is "
            "diagnostic and this run routes no verdict on MECH-572: it probes "
            "whether behaviourally grounded epistemic provenance, used ONLY to "
            "regulate sleep-time plasticity gain, improves consolidation beyond "
            "a budget-matched generic reduction, and whether it self-seals. "
        ) + note + " " + EVIDENCE_LEVEL_CAP,
        "non_degenerate": bool(all(
            criteria_non_degenerate[g["name"]] for g in gates
            if g["routes_verdict"])),
        "degeneracy_reason": (
            None if all(criteria_non_degenerate.values())
            else (
                "PARTIAL: "
                + ", ".join(k for k, v in criteria_non_degenerate.items() if not v)
                + " are degenerate and must NOT be cited in either direction. "
                "non_degenerate tracks the VERDICT-ROUTING gates G1..G7 only: a "
                "degenerate contrast does NOT vacate the others or the run "
                "(failure_autopsy_V3-EXQ-785_2026-07-19 sections 2a/8). The "
                "usual cases are G8 (MECH-573 readability) marking the "
                "converged-base RETENTION contrasts unciteable, and G9 (PE "
                "separation) marking the COND_CONFIDENTLY_WRONG contrasts "
                "unciteable, neither of which touches COND_UNDERFIT or the "
                "mechanistic readouts.")),
        "interpretation": {
            "label": label,
            "criteria": gates + contrasts,
            "combination_rule": (
                "PASS iff ALL SEVEN verdict-routing gates pass (plain AND, no "
                "OR/any() branching): G1, G2, G3 (its NOISE leg only), G4 (its "
                "pi_hist-varies leg only), G5, G6, G7. The five gates carrying "
                "routes_verdict=false -- G3b, G4b, G8, G9 and every P -- can "
                "never fail the run; they mark specific contrasts unciteable. "
                "P1..P7 are the pre-registered "
                "CONTRASTS: they carry routes_verdict=false, route NO PASS/FAIL, "
                "and select interpretation.label from the preregistration's "
                "section-12 matrix. A negative P IS A RESULT (P7 explicitly so). "
                "MATRIX READING PINNED (RED-TEAM F8): R3 self_sealing_failure "
                "requires P2 PASS and P4 FAIL -- the preregistration's "
                "'regardless of P2' is read as saying P4's failure is not "
                "excused by P2, NOT that R3 can fire with no protection "
                "demonstrated, because with no protection there is no "
                "self-sealing to claim. R7 is reachable only when R1, R2 and R3 "
                "do not fire, plus (BATCH 2 EDIT 2) as the FIRST test inside "
                "the cond-3-unreadable branch, where P5 is still a real reading "
                "and P4 is not. "
                "THREE DISTINCT FAIL ROUTES: (a) a per-cell readiness "
                "precondition unmet, a cell short of its cycles, or a non-finite "
                "parameter -> label substrate_not_ready_requeue; (b) G1 or G6 "
                "failing -> instrument_inert_substrate_not_ready (matrix R5), "
                "G3/G4/G5 failing -> precision_channel_degenerate (matrix R6), "
                "G2 failing -> substrate_not_ready_requeue; (c) G7 alone failing "
                "-> provenance_gain_measurement_invalid. None emits a pattern "
                "label. G8/G9 route NO verdict and mark only the specific "
                "contrasts they gate non_degenerate=False."),
            "criteria_non_degenerate": criteria_non_degenerate,
            "preconditions": aggregate["adjudication_preconditions"],
        },
        "readout": flat,
        "contrasts": {
            "seeds": list(seeds),
            "seeds_required": float(req),
            "margin_absolute": MARGIN,
            "margin_rule": (
                "MARGIN is an ABSOLUTE FLOOR on the relative quantities (2%). "
                "Where the per-seed SD of a contrast's paired deltas exceeds it, "
                "the 1-SD margin is reported alongside and THE DECISION USES THE "
                "LARGER (preregistration section 11). margin_source on each "
                "record says which was used."),
            "statistics_note": (
                "The pre-registered decision rule is the 2-of-3 seed SIGN COUNT. "
                "The t-based 95% CI at n=3 is carried on every record and is "
                "DESCRIPTIVE ONLY, flagged ci_low_power=true."),
            "records": contrasts,
            "c_seed_by_seed": {str(k): v for k, v in c_seed_by_seed.items()},
            "c_seed_note": (
                "c_seed is ARM C's realised mean per-step gain for that seed, "
                "pooled over ALL conditions and cycles and weighted by the number "
                "of e2_world consolidation steps in each cycle. It is read from "
                "the SD-PP-4 gain diagnostics and NEVER from any DV. The D arms "
                "run AFTER C for a seed for exactly this reason, so budget "
                "equivalence is exact in TOTAL and deliberately not per "
                "condition -- the rival hypothesis is a single global reduction."),
        },
        "gates": {
            "records": gates,
            "routing_note": (
                "EXACTLY SEVEN criteria carry routes_verdict=true: G1, G2, G3 "
                "(noise leg), G4 (pi_hist leg), G5, G6, G7. G3b, G4b, G8 and G9 "
                "route NO verdict: they mark the contrasts they gate "
                "non_degenerate=False rather than letting a compressed, "
                "unseparated or aliased signal read as a null (red-team 14). "
                "G3b and G4b were rescoped from verdict-routing legs of G3 and "
                "G4 before the freeze record, on the probe statistics recorded "
                "in the module docstring."),
            "g2_mismatches": g2_mismatches,
            "cond3_readable": bool(cond3_readable),
            "degeneracy_causes": degeneracy_causes,
            "amendment_3_note": (
                "AMENDMENT 3. G4 keeps ONLY its pi_hist-varies leg as a "
                "verdict-routing gate; its surprise leg is now G4b, which with "
                "G9 decides whether COND_CONFIDENTLY_WRONG posed a "
                "CONTRADICTION at all on this head. The full-budget probe found "
                "it does not: the converged head is near copy-the-input and "
                "barely reads the action (MECH-573 / substrate necessity B5), "
                "so permuting the action map leaves its inverted-rule battery "
                "MSE (1.13e-5) BELOW its original-rule MSE (1.49e-5). P4, P7 "
                "and P6's correction leg are therefore UNASKED rather than "
                "null, and are marked non_degenerate=False instead of failing "
                "the run. The cond-3 cells are still RUN and fully RECORDED "
                "across all three seeds: that record is this run's measurement "
                "that B5 is real, which is a contribution in its own right."),
            "g2_cycle_scope_note": (
                "G2 compares CYCLE 1 only. After the first sleep the arms "
                "legitimately differ, so their cycle-2 pre-sleep states differ "
                "too -- that is the manipulation working, not a gate failure."),
            "g4_early_packets_note": (
                "The rule inversion in COND_CONFIDENTLY_WRONG is present from P1 "
                "step 0, so 'the first 30 real transitions after the shift' are "
                "the first 30 has_prev packets of P1."),
            "channel_readout_source": (
                "G3/G4/G5/G9's channel readouts are taken from ARM_B_STORE_ONLY "
                "at cycle 1: it carries the producers AND its waking is "
                "bit-identical to ARM_A's, so it is the neutral place to read the "
                "channels without a gain having already acted on them."),
        },
        "config": run_config,
        "elapsed_seconds": elapsed,
        "competing_explanations_note": (
            "MECH-572's phenotype has an optimiser explanation that this design "
            "does NOT merely assume: cross_module_consolidation builds a FRESH "
            "Adam per call, so 8 steps move each weight by at most 8*CMC_LR = "
            "0.008 regardless of gradient magnitude, and V3-EXQ-1060's landed "
            "world-head delta 0.007899 is 98.7% of that bound. That is precisely "
            "why the gain acts on the per-step LR as well as on the per-row "
            "weights, and why ARM_D_GLOBAL exists: a uniform lr reduction is the "
            "rival hypothesis, budget-matched to C, and P6 is the contrast that "
            "separates them. adam_step_bound_8x_lr, min_identity_predictor_mse_r0 "
            "and the per-cell displacement_to_residual_ratio are recorded so a "
            "reader can check the perturbation-vs-residual crossover directly."),
        "proxy_limitations_note": (
            "TWO NAMED PROXIES, carried from the substrate spec and not papered "
            "over. (1) SD-PP-1's kappa is measured on real MOTION and applied to "
            "NOISE, so evidence_variance_z OVER-estimates z-noise and the "
            "downstream gain is CONSERVATIVE under noise -- it will under-learn "
            "rather than over-learn a noisy contradiction. (2) SD-PP-2 splits "
            "epistemic from aleatoric by SUBTRACTION (v_epi = v_tot - v_noise), "
            "because SD-063 supplies total predictive spread only; noise_gain=2.0 "
            "is a first-principles constant for a near-identity head (MECH-573), "
            "not a fitted one. Both are registered substrate necessities (f)/(g)."),
    })
    return outcome, _write(manifest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help=("Smoke run: 1 seed, short P0, 1 cycle, 1 short wake "
                              "episode -- but ALL SIX arms and ALL FOUR "
                              "conditions, and the full battery. No manifest."))
    parser.add_argument("--liveness", action="store_true",
                        help=("G1 only: the displacement-vs-gain-scale ladder "
                              "plus the retention attainability curve, at the "
                              "pinned seed. Writes no manifest."))
    parser.add_argument("--probe", action="store_true",
                        help=("Section-15 FREEZE ARTIFACT: ARM_B + ARM_C, one "
                              "seed, all four conditions, FULL P0 and FULL "
                              "calibration, one cycle. Prints every channel "
                              "statistic the gates are defined on. Writes no "
                              "manifest. Takes several minutes."))
    args = parser.parse_args()
    _outcome, _manifest_path = main(dry_run=args.dry_run,
                                    liveness_only=args.liveness,
                                    probe_only=args.probe)
    _clean = str(_outcome).upper() if str(_outcome).upper() in ("PASS", "FAIL") else "FAIL"
    emit_outcome(outcome=_clean, manifest_path=_manifest_path,
                 dry_run=bool(args.dry_run or args.liveness or args.probe))
    sys.exit(0)
