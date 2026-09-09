"""
REPAIRED 2026-09-09 (session wizardly-meninsky-e6c09c, campaign W6-S5b item 1)
=========================================================================
This driver was left UNTRACKED with a DO-NOT-QUEUE banner on 2026-09-04
(WORKSPACE_STATE 2026-09-04T20:11:05Z) carrying three source-verified
blockers, B1-B3. All three are dispositioned below; the 09-04 measurements
that established them are kept verbatim in the repair notes so a reader
can check the fix against the number it answers.

B1 (WAS: both replay-reachability gates unsatisfiable -- min>=1 over 75
    eval-episode firings of mech285_n_draws while agent.reset() empties the
    anchor pool every episode; measured per-episode draws [0,50,0,50,0,0,0,0]
    at a ~0.45 boundary-events/episode rate, P(all 75) ~ 8.6e-34, and NOT a
    warmup-budget artifact -- the 200-episode warmup showed the same 2-of-8
    rate).
B2 (WAS: channel (c) inert -- draws_per_cycle only reaches the waking agent
    through phase_manager.py:521-524, which computes mean_anchor from the
    routed draws ONLY when use_mech272_routing_consumer is True and otherwise
    pins mean_anchor = 1.0, so run_sleep_cycle was byte-identical across
    blocks).
    FIX FOR B1+B2 TOGETHER -- THE LEVER MOVES. The mech285 DRAW COUNT is the
    wrong operationalisation of "suppressed replay" on this substrate, for
    two reasons verified at source on 2026-09-09: (i) its only consumer, the
    GAP-8 routing consumer, is SIGN-INVERTED for this design -- with the
    consumer ON, zero draws yields mean_anchor = 1.0 (a FULL-strength schema
    write) while fifty draws yields the SWS row weight 0.6, so "suppressing
    replay" would STRENGTHEN the offline write (phase_manager.py:518-525,
    routing_gate.py table); and (ii) even when populated, the anchor pool is
    only intermittently non-empty (B1), so the contrast would be ~25%-engaged
    at best. The channel is therefore re-specified onto the OFFLINE PASSES
    THE DRAWS FEED: the SD-017 SWS schema-installation pass
    (agent.run_sws_schema_pass -> E1.ContextMemory writes) and the REM
    attribution pass. EVAL_HYPERVIGILANT sets agent.config.sws_enabled =
    agent.config.rem_enabled = False (and draws_per_cycle = 0); BASELINE and
    REVERSION hold both True. run_sleep_cycle reads those flags live
    (agent.py run_sleep_cycle: "If both are False this is a no-op"), and
    SleepLoopManager._run_cycle honours require_sleep_passes_enabled by
    returning None at every HV boundary -- the recorded suppression
    signature. Nothing is rebuilt; the lever is runtime-revertible.
    THE PATH TO THE DV IS OPEN AND MEASURED (Step 2.5a probe, seed 11,
    8 x 30-step warmup; REPRO, no external file needed: build_config() +
    _train_warmup() as below, then snapshot (cm.memory, cm.read(q),
    e1.generate_prior(q)) for a fixed random query q of width
    cm.latent_dim around each agent.force_sleep_cycle_at_eval_boundary()
    call, toggling agent.config.sws_enabled/rem_enabled between them): a forced
    eval-boundary cycle with the passes enabled reports sws_n_writes = 5
    (= sws_consolidation_steps, DETERMINISTIC given a populated
    _world_experience_buffer, which survives agent.reset(): 237 before and
    after) and moves E1.ContextMemory.memory by max|delta| 0.0568, its
    read() by 0.0170 and e1.generate_prior() by 0.0042 on a fixed query;
    with the passes disabled the same call returns None and every one of
    those deltas is exactly 0.0; re-enabling restores writes (5) and
    movement (0.0575). generate_prior() is what _e1_tick feeds to
    hippocampal.propose_trajectories(e1_prior=...) in _agent_tick, whose
    candidates E3 selects from -- so offline schema installation reaches
    the committed action through candidate generation. The MAGNITUDE of
    channel (c)'s behavioural contribution is NOT separately estimated (the
    claim is about the JOINT three-channel regime; no channel is separately
    load-bearing), which is unchanged from 981/981a-as-drafted. The
    replay-reachability gates now read sws_n_writes (min >= 1 per firing in
    BASELINE/REVERSION; max <= 0 and None-return on every HV boundary), which
    the substrate can deliver; mech285_n_draws is still RECORDED per firing
    as a diagnostic (use_event_segmenter stays True) but gates nothing.
    Side effect stated, not hidden: disabling the SWS pass also skips
    enter_sws_mode()'s MECH-120 SHY normalisation and the serotonin
    sleep-phase hooks for the HV block. That is part of what "suppressed
    offline replay" means here and is recorded per block as
    offline_passes_enabled. NOTE: sws_consolidation_steps = 0 is NOT a usable
    lever -- run_sws_schema_pass divides by it (agent.py ~12360,
    ZeroDivisionError, found by the probe).
    C1-null routing (B2's second half): with channel (c) wired for real,
    the else-branch "weakens" stays -- it is reachable ONLY after Gate A and
    Gate B certify every channel engaged, so a null there is a measured null,
    not an unreached leg.
B3 (WAS: MIN_BIN_COVERAGE_STEPS = 150 per seed per bin unmet on all 3 seeds
    by 981's own EVAL_BASELINE HIGH counts 139/38/52, with HAZARD_HIGH_MIN
    unchanged so HIGH occupancy is unchanged).
    FIX -- THE FLOOR IS RE-DERIVED FROM WHAT IT PROTECTS, not lowered to fit.
    The coverage gate exists so the positive control (HIGH-vs-SAFE avoidance
    LIFT, pooled across seeds, against POSITIVE_CONTROL_MARGIN = 0.05) is
    estimable. A binary-rate SE is <= 0.5/sqrt(n); the +0.05 bar needs a
    pooled SE of about 0.04, i.e. n >= 150 POOLED per positive-control band
    (981 measured 229 pooled HIGH and 412 pooled SAFE under the OLD band; the
    new SAFE band d >= 7 is strictly larger). The per-seed floor is a
    NON-VACUITY minimum, not a power requirement: MIN_BIN_COVERAGE_STEPS_PER_
    SEED = 30 so a single tick moves a per-seed rate by <= 3.3% (981's
    smallest HIGH cell was 38). So: per-seed floor 30 on every (block, bin);
    pooled floor 150 on the two positive-control bands. Both are pre-registered
    constants derived from the bar, and both are checked against 981's
    measured counts above (min margins 1.27x and 1.53x) rather than assumed.
    C1/C2 read the AMBIGUOUS band (n ~ 2700 per seed) and are unaffected.
    DRY-RUN: the floors scale with the eval-tick budget (3 x 15 = 45 ticks
    per block vs 25 x 120 = 3000), and under --dry-run Stage B now RUNS
    REGARDLESS of Gate A as a code-path positive control (red-team F3: Stage
    B, Gate B, C1/C2 and every verdict branch had never executed once). A
    forced smoke run is stamped gate_a_forced_for_smoke and cannot read as
    anything but substrate_not_ready_requeue.

STATUS OF THE AUTOPSY'S THREE FIXES: fix 2 (C1 bar inside the DV range) and
fix 3 (two-stage gate-first design) are implemented and verified. Fix 1
(use_event_segmenter=True) is kept -- it is correct as far as it goes -- but
is no longer what channel (c) rests on (see B1/B2 above). The reversed
positive control was ruled NOT a driver bug (section 2 below); the band
re-derivation works -- the dry-run reads positive_control_margin = +0.44 on
lift against a +0.05 floor, versus 981's -0.4307 on raw rates.
=========================================================================
V3-EXQ-981a -- MECH-027 (hypervigilance signature): control-plane channel
forcing, targeted probe, with within-run reversion check.

SUPERSEDES V3-EXQ-981 (manifest
v3_exq_981_mech027_control_plane_pathological_modes_20260903T053044Z_v3.json,
confirmed autopsy failure_autopsy_V3-EXQ-981_2026-09-03.md). 981 self-routed
substrate_not_ready_requeue with 4 of 11 preconditions unmet; that self-route
was adjudicated CORRECT. MECH-027 has never been tested and takes no weight
from 981.

SLEEP DRIVER: manual-multi (force_cycle() every SLEEP_EVERY_N_EPISODES during
warmup; agent.force_sleep_cycle_at_eval_boundary() at every eval episode
boundary in all four blocks -- MECH-027 Build 2 convention, 2026-09-02. In
EVAL_HYPERVIGILANT the offline passes are DISABLED (sws_enabled = rem_enabled
= False, 2026-09-09 repair), so the same boundary call is a recorded no-op
that returns None -- the suppression signature channel (c) is gated on; the
paired EVAL_HV_AB_PAIRED block keeps them ON).

RED-TEAM OF THE 2026-09-09 REPAIR (fable): CONTESTED, six findings, all FIXED
-- see RED_TEAM_REPAIR_VERDICT below and the queue entry note. The design now
has FOUR eval blocks: EVAL_BASELINE, EVAL_HYPERVIGILANT (a+b+c forced) and
EVAL_REVERSION on the same agent, plus EVAL_HV_AB_PAIRED (a+b forced, offline
passes ON) on a deep copy of the post-Stage-A agent and env started from the
same RNG state as EVAL_HYPERVIGILANT -- so lift(HV) - lift(HV_AB) is the paired
behavioural contribution of suppressing replay (red-team F4). C1/C2 read the
AMBIGUOUS-band LIFT (F1). Verdict routings added for an order confound (F3),
an inert channel (c) (F4) and a range-limited channel (a) (F5).

RED-TEAM (981a): see custom_information.red_team_disposition_note and the
queue entry note. (981's own red-team, fable 2026-09-02: CONTESTED, 1
BLOCKING F1 fixed; its F7 -- "the SAFE hazard bin may be structurally empty
for a central hazard placement" -- was ACKNOWLEDGED, NOT MITIGATED, and is
the finding this successor is built on.)

=====================================================================
WHAT CHANGED FROM 981, AND WHY (the four coupled fixes)
=====================================================================

(1) REPLAY ANCHOR POOL -- hippocampal.use_event_segmenter = True.
    981's mech285_n_draws read 0 on all 75 BASELINE firings per block: the
    sampler drew 50 times per firing and got None each time because
    AnchorSet.all_with_dual_trace() was always empty. The only live
    anchor-install path is consume_boundary_events -> write_anchor, which
    runs only when sense() emits BoundaryEvents, which requires
    HippocampalConfig.use_event_segmenter -- default False
    (ree_core/utils/config.py:2761) and never set by 981. The V3-EXQ-909
    three-flag recipe (use_anchor_sets + use_mech285_sampler +
    use_mech272_routing) is NECESSARY BUT NOT SUFFICIENT; this is the
    missing fourth flag. It is a config correction, not a build. It must be
    set BEFORE REEAgent construction (HippocampalModule instantiates the
    segmenter in __init__, module.py:273).

(2) THE REVERSED POSITIVE CONTROL IS *NOT* AN INVERTED HAZARD-BAND
    ASSIGNMENT. Ruled out explicitly at source before any redesign, per the
    governance-20260903 red-team amendment. Three checks, all negative:
      (a) DIRECTION. _hazard_bin maps value < SAFE_MAX -> SAFE and
          value >= HIGH_MIN -> HIGH. The value is
          hazard_field[agent_cell] / hazard_field.max() and
          hazard_field[i,j] = 1/(1 + d*decay) with d = Manhattan distance to
          the hazard (causal_grid_world.py:4602), max = 1.0 attained at the
          hazard's own cell. So the value is strictly DECREASING in
          distance: high value = close = HIGH band. Correct, not inverted.
      (b) AXES. _hazard_cells reads np.argwhere(env.grid == hazard), whose
          (i, j) is the same (x, y) the env indexes movement and the field
          with (hazard_field[new_x, new_y], step()'s new_x = agent_x + dx,
          ACTIONS dx applied to agent_x). No transpose.
      (c) SIGN OF THE AWAY VECTOR. away = (ax - hx, ay - hy) and the chosen
          action maximises dx*away_dx + dy*away_dy -- i.e. moves AWAY. Correct.
    THE ACTUAL CAUSE, read off 981's own manifest, is a band-population and
    boundary-geometry confound, and it is fixed by (2a) and (2b) below:
      - BAND POPULATIONS ARE EXTREME. Per-seed BASELINE (block_bins):
        SAFE n = 123 / 132 / 157, AMBIGUOUS n = 2738 / 2830 / 2747,
        HIGH n = 139 / 38 / 52 of ~3000 steps. AMBIGUOUS carried 91-94%; the
        two tails carried 1.3-5.2%.
      - THE OLD "SAFE" BAND WAS THE FAR CORNER, NOT A LOW-SIGNAL STATE.
        SAFE required value < 0.15, i.e. d > 11.33. On a 10x10 non-toroidal
        grid the maximum d from a CENTRAL hazard is 10, so SAFE is EMPTY for
        a central placement and, for any other, is exactly the wall-adjacent
        far corner -- a region where the single "avoidant" action is a
        wall-blocked no-op. 981's SAFE-band avoidant rates (0.293 / 0.871 /
        0.885) are boundary geometry, not avoidance behaviour, which is why
        HIGH - SAFE came out at -0.4307.
    (2a) BANDS RE-DERIVED SO ALL THREE ARE POPULATED FOR ANY HAZARD
         PLACEMENT. HAZARD_SAFE_MAX 0.15 -> 0.25 (value < 0.25 <=> d >= 7)
         and HAZARD_HIGH_MIN stays 0.50 (value >= 0.50 <=> d <= 2). Both
         cutpoints are exact analytic consequences of value = 1/(1 + 0.5*d)
         with decay 0.5, computed from the env's geometry BEFORE the run --
         not from any scored statistic. Coverage floors: per-seed 30
         (non-vacuity) + pooled 150 on the positive-control bands (power),
         re-derived 2026-09-09 -- see B3 at the top of this docstring.
    (2b) THE DV IS AVAILABILITY-NORMALISED AND WALL-AWARE.
         _avoidant_actions returns the SET of actions that STRICTLY INCREASE
         the Manhattan distance to the nearest hazard AFTER the env's own
         clamping -- so a wall-bump or a stay is correctly NOT avoidant, and
         a tick with no distance-increasing action available is UNSCORABLE
         and excluded from its band's denominator (counted as
         n_unscorable). Per band the run records rate = avoidant / n,
         chance = mean(|avoidant set| / action_dim), and
         lift = rate - chance.
         The POSITIVE CONTROL compares two DIFFERENT bands, whose chance
         rates differ by geometry, so it reads LIFT. C1/C2 compare the SAME
         band across arms, where availability is matched by construction, so
         they read the raw AMBIGUOUS-band RATE (bounded [0,1], which is what
         makes its headroom check well defined).
    (2c) THE PAIRING IS ESTABLISHED *FIRST*, MECHANICALLY. The run is now
         TWO-STAGE. Stage A runs warmup + EVAL_BASELINE for every seed and
         then evaluates GATE A (positive control, band coverage, replay
         reachability, and both dv_headroom checks). EVAL_HYPERVIGILANT and
         EVAL_REVERSION are NOT RUN AT ALL unless Gate A is met -- so a
         non-hazard-sensitive pairing costs one block, not three, and can
         never be built on.

(3) C1'S THRESHOLD NOW LIES INSIDE THE DV'S RANGE, AND IS GATED.
    981 set C1 at 2 x 0.5771 = 1.1542 on a DV bounded in [0,1] -- unattainable
    by any value once the pooled baseline exceeded 0.5, which it did on all
    three seeds. FALSE_ALARM_ELEVATION_MULTIPLIER (2.0, multiplicative) is
    REPLACED by FALSE_ALARM_ELEVATION_ABS = 0.10, an ABSOLUTE elevation on a
    [0,1] DV, and is gated by a dv_headroom precondition measured on this
    run's OWN baseline arm (statistic "ceiling_headroom", dv_bounds (0,1),
    margin 2.0 -- so the bar must sit at most half the achievable range
    away). On 981's numbers the headroom would have been
    1 - 0.6949 = 0.3051 against a required 0.20: feasible. The non-overlap
    condition ("HV above baseline's own across-seed range") is DEMOTED to a
    recorded diagnostic (c1_nonoverlap_also_holds) so that C1 has exactly
    one pre-registered constant for the headroom gate to guard.

(4) THE PRECISION-MARGIN / COMMIT-TEMPERATURE FLOORS ARE RE-DERIVED AGAINST
    MEASURED HEADROOM, NOT LEFT AT A FIXED 0.01.
    981 measured a BASELINE pooled precision_margin_norm of 0.99980, so the
    arithmetic ceiling on any elevation was 0.000195 against a 0.01 floor --
    a 51x shortfall the manipulation could not close (it extracted 0.000194,
    99.4% of everything available). commit_temperature_reduced_under_hv is
    ALGEBRAICALLY THE SAME QUANTITY (T_eff = 1.0 + alpha*(1 - margin), alpha
    = 1.0), was equally unreachable, and is DROPPED as an independent
    readiness check -- it is retained as a diagnostic metric only.
    The margin saturates because precision_margin_norm =
    clamp(1 - commit_variance/effective_threshold, 0, 1) and
    effective_threshold comes from E3Config.commitment_threshold (default
    0.40) while the trained running_variance sits ~5000x below it. So MECH-027
    Build 1's "graded" consumer is not graded in a trained substrate at all:
    T_eff was pinned within 2e-4 of base_temperature, i.e. channel (a) had no
    behavioural consumer. This run therefore CALIBRATES the gate:
    immediately after warmup, and identically in ALL THREE BLOCKS (so it is
    never part of the arm contrast), it sets
      agent.e3.config.commitment_threshold =
          COMMIT_THRESHOLD_VARIANCE_MULTIPLE * running_variance_post_warmup
    with COMMIT_THRESHOLD_VARIANCE_MULTIPLE = 4.0, which places the baseline
    margin near 1 - 1/4 = 0.75 and leaves ~0.25 of ceiling headroom. The
    calibration is a pre-registered RULE, not a tuned number, and it is
    VERIFIED by a dv_headroom precondition in Gate A
    (dv_name precision_margin_norm, ceiling_headroom against dv_bounds
    (0,1), criterion_threshold PRECISION_MARGIN_HV_ELEVATION_FLOOR = 0.05).
    If the calibration fails to open headroom the run self-routes
    substrate_not_ready_requeue after Stage A, having spent one block.

DV-SYMMETRY INVARIANCE (per arm, mandatory declaration). The DV is a
per-tick BINARY membership test (was the committed action in the
distance-increasing set?) aggregated to a rate, so its symmetry group is
(i) permutations of ticks within a band and (ii) any relabelling of actions
that preserves the (dx, dy) map. The manipulation is invariant under
NEITHER: EVAL_HYPERVIGILANT changes WHICH action is committed on a given
tick (via the precision-scaled commit temperature at the committed-selection
layer, plus a shortened CEM scoring window and a suppressed replay channel),
which is exactly a change of membership, not a relabelling and not a
reordering. It is a discrete SELECTION-level manipulation, so it is also not
a broadcast additive constant on a scored quantity (the V3-EXQ-604c class);
the constant-offset hazard cannot apply because nothing here is read through
an argmax over a shifted score vector supplied by the driver. The same
statement holds unchanged for EVAL_BASELINE and EVAL_REVERSION, which differ
from EVAL_HYPERVIGILANT only in those channel settings.

Claims: MECH-027 ("Pathological modes reflect mis-tuned control-plane
regimes")

EXPERIMENT_PURPOSE = "evidence" (direct falsifier test of MECH-027, scoped to
one of its five named pathological labels -- see SCOPE below).

WHY THIS SCOPE (dispatch_mode: targeted_probe, per source proposal
EXP-0761/EVB-1396). A prior substrate-readiness investigation established:
  - gain/precision (ARC-016, E3TrajectorySelector.current_precision /
    _running_variance) -- IMPLEMENTED, WIRED, non-degenerate (confirmed by
    V3-EXQ-876/876a's own precision separation, 4-5 orders of magnitude).
  - prediction horizon (HippocampalConfig.horizon structural default, and the
    SD-MECH267-HORIZON-DEPTH runtime CEM elite-scoring-window mechanism,
    mode_horizon_scale + operating_mode) -- IMPLEMENTED, WIRED, non-degenerate.
  - replay/hippocampal-injection suppression (mech285 sampler,
    use_mech285_sampler / mech285_draws_per_cycle) -- IMPLEMENTED, WIRED,
    confirmed reachable (V3-EXQ-909: draws_per_cycle=50 on every seed once
    use_anchor_sets + use_mech285_sampler + use_mech272_routing are all set).
  - "learning eligibility" (needed for the MANIA signature) has NO confirmed
    substrate hook, and "hippocampal gating" (dissociation/psychosis-like
    labels) was out of scope for this investigation.
So THIS experiment tests ONLY the HYPERVIGILANCE signature, which the claim's
own text defines as needing exactly the three channels above: "elevated gain
+ shortened horizon + suppressed replay ... reproduces ... excessive
false-alarm / over-reactive responding ... AND the behavior reverts when the
channel is returned to its normal range." This is a scoped single-signature
probe (claims.yaml MECH-027 what_would_answer), not a forced 5x5 sweep across
all five pathological labels -- the other four are out of scope here.

NON-DEGENERACY PRECONDITION (claim's own text, MANDATORY): each of the three
channels must show independently measurable AND independently perturbable,
NON-DEGENERATE variance across the pathological/non-pathological range under
test. A run where any channel is flat, hardcoded, or torn down at read time
self-routes substrate_not_ready_requeue, never a verdict. MECH-025 (sharing
MECH-027's ARC-016/ARC-005 dependency) had three consecutive false-negative
instrument defects of exactly this kind (hardcoded E3 precision; missing
self-attribution channel; cached/torn-down commitment-state field; frozen
running_variance during eval) before V3-EXQ-876 finally got a fair test
(failure_autopsy_mech025-cluster-876-671b_2026-08-03). This script keeps
running_variance LIVE during eval (agent.update_residue() -> E3.post_action_
update() -> update_running_variance() every tick, exactly 876/876a's fix) and
additionally forces the channel via a per-tick multiplicative scale (see
"HOW EACH CHANNEL IS FORCED" below) so it stays perturbed AND non-degenerate
(real per-tick variance) rather than a frozen hardcoded constant.

DESIGN -- ONE agent per seed, THREE sequential eval regimes on the SAME
trained substrate (not three separately-trained agents). This is the design
the claim's own framing calls for: "mis-tuned regimes of the SAME
control-plane machinery," not a different learned substrate per condition.
  1. WARMUP (shared): standard E1/E2/E3 training under baseline channel
     settings (with periodic forced sleep cycles, mech285 replay engaged at
     the baseline draw count) -- builds ONE functional agent per seed.
  2. EVAL_BASELINE block: normal-range gain/precision, horizon, replay.
  3. EVAL_HYPERVIGILANT block (SAME agent, channels forced): elevated
     precision + shortened horizon + suppressed replay, simultaneously.
  4. EVAL_REVERSION block (SAME agent, channels reverted to baseline):
     confirms the claim's own required reversion half of the confirming
     signature -- "the behavior reverts when the channel is returned to its
     normal range."

HOW EACH CHANNEL IS FORCED (all three are RUNTIME-revertible on the live
agent -- verified against current source before writing this script, none
requires rebuilding the agent's network):

  (a) Gain/precision -- E3TrajectorySelector._running_variance (plain
      instance attribute; current_precision = 1/(running_variance+1e-6),
      ree_core/predictors/e3_selector.py:771-773). Each EVAL_HYPERVIGILANT
      tick, AFTER agent.update_residue() has driven its normal live EMA
      update (post_action_update -> update_running_variance, agent.py:10489,
      e3_selector.py:4151), the driver additionally multiplies
      _running_variance by HV_PRECISION_SCALE (<1, shrinks variance ->
      elevates precision). This keeps the channel LIVE (still moves with
      genuine per-tick prediction error) while forcing it out of its normal
      operating range -- the "directly perturb precision_init / inject a
      scaling factor" option the design brief names. EVAL_BASELINE and
      EVAL_REVERSION apply scale=1.0 (no forcing; natural ARC-016 dynamics).

      GRADED DOWNSTREAM CONSUMER (MECH-027 Build 1, 2026-09-02,
      chip-20260902-mech027-precision-replay-eval-substrate): a red-team
      review of an earlier draft of this script found channel (a) as
      described above BLOCKED -- the only pre-existing consumer of
      _running_variance was the binary ARC-016 commit gate
      (committed = commit_variance < effective_threshold), which SATURATES
      in a trained substrate (running_variance empirically ~125x below
      threshold), so once committed, further shrinking variance had NO
      further observable effect anywhere downstream: the (a) forcing above
      could move current_precision as a NUMBER while leaving E3's actual
      selection behavior untouched. FIX: this driver additionally sets
      E3Config.use_precision_scaled_commit_temperature=True (build_config()
      below), which gives current_precision/running_variance a GRADED
      consumer: the committed argmin becomes
      multinomial(softmax(-q/T_eff)) with T_eff = base_temperature +
      PRECISION_SCALED_COMMIT_ENTROPY_ALPHA * (1 - precision_margin_norm),
      where precision_margin_norm = clamp(1 - commit_variance/
      effective_threshold, 0, 1). A maximally-confident tick (precision
      forced up, precision_margin_norm -> 1) commits COLD (T_eff -> base,
      hard/rigid argmin -- the hypervigilance direction); a barely-committed
      tick commits HOTTER (softer, more exploratory). q is restricted to an
      F-eligibility envelope (PRECISION_SCALED_COMMIT_HARM_FLOOR *
      raw_score_range of the best raw score) so a hot commit-T can never
      softmax-promote a clearly-harmful candidate. This is
      READ, not additionally forced, by this driver -- per-tick
      precision_margin_norm and precision_scaled_commit_temperature_eff are
      captured from agent.e3.last_score_diagnostics at every real E3
      selection (see _agent_tick) and pooled into the new readiness checks
      below, so a run where the graded consumer never actually engaged
      (e.g. the standalone committed branch not reached, or baseline
      already precision-saturated so HV cannot push it further) self-routes
      substrate_not_ready_requeue rather than a false hypervigilance
      verdict.

  (b) Prediction horizon -- SD-MECH267-HORIZON-DEPTH
      (HippocampalModule._compute_mode_horizon_scale /
      config.hippocampal.mode_horizon_scale, module.py:1836-1860,
      1990-2016). CORRECTED FROM THE ORIGINAL BRIEF: agent.py's real
      _e3_tick() call site (agent.py:5951-6153) does NOT pass operating_mode
      to hippocampal.propose_trajectories() at all (verified by direct grep
      of the call site) -- so agent.generate_trajectories() cannot be used
      for this channel. This driver instead calls
      agent.hippocampal.propose_trajectories(..., operating_mode=...)
      DIRECTLY (see _agent_tick below), mirroring _e3_tick's essential shape
      (candidate caching between e3_ticks, theta-independent since this
      probe does not use goal/ghost-probe machinery) but with a synthetic
      operating_mode dict this driver controls per-tick. mode_horizon_scale
      scales the CEM ELITE-SELECTION SCORING WINDOW, NOT the physical
      rollout length (config.horizon / terrain_prior's output width stays a
      fixed structural network dimension, per the module's own docstring at
      1904-1910) -- so this is genuinely runtime-revertible on the live
      agent with no network rebuild, unlike HippocampalConfig.horizon
      itself.

  (c) Replay/hippocampal-injection suppression -- the SD-017 OFFLINE
      PASSES (2026-09-09 repair; see B1/B2 at the top of this docstring).
      The lever is agent.config.sws_enabled / agent.config.rem_enabled,
      read LIVE by agent.run_sleep_cycle() at every cycle, plus
      agent.sleep_loop.draws_per_cycle for the mech285 sampler that feeds
      them. EVAL_HYPERVIGILANT sets both flags False and draws to 0 (full
      suppression: no SWS schema installation into E1.ContextMemory, no REM
      attribution pass, no replay draws); EVAL_BASELINE and EVAL_REVERSION
      hold both True and draws at MECH285_BASELINE_DRAWS. The three
      construction-time gates that make agent.sleep_replay_sampler non-None
      (use_anchor_sets, use_mech285_sampler, use_mech272_routing) plus the
      fourth (hippocampal.use_event_segmenter) are set on EVERY agent so the
      machinery is built once and only its runtime flags differ per block.
      WHY NOT draws_per_cycle ALONE (the 981/981a-draft lever): its sole
      consumer (GAP-8 routing consumer, phase_manager.py:518-525) is
      sign-inverted for suppression (0 draws -> full-strength write) and the
      anchor pool it draws from is only intermittently populated (B1). The
      offline passes are the thing the draws were meant to modulate, and
      they have a measured route to the DV: SWS writes E1.ContextMemory,
      which E1.generate_prior() reads, which _agent_tick feeds into
      hippocampal.propose_trajectories(e1_prior=...), whose candidates E3
      selects from.

      SLEEP CYCLES ACTUALLY FIRE DURING EVAL (MECH-027 Build 2, 2026-09-02,
      unchanged). This driver calls agent.force_sleep_cycle_at_eval_boundary()
      once at the end of EVERY eval episode, in all three blocks (see
      _run_eval_block) -- the formalized V3-EXQ-909 flush-then-force_cycle
      sequence, which does NOT call agent.reset() and is reachable at an
      eval boundary without corrupting the wake/sleep state machine. In
      BASELINE/REVERSION each call returns a metrics dict carrying
      sws_n_writes (deterministically = sws_consolidation_steps once the
      world-experience buffer is populated), rem_n_rollouts and
      mech285_n_draws; in HV the call returns None (SleepLoopManager
      ._run_cycle honours require_sleep_passes_enabled). This driver records
      all of it per firing so the readiness gates confirm REAL engagement
      (sws_n_writes >= 1 on every BASELINE/REVERSION firing; None on every HV
      boundary), not merely that a flag was set.

DV -- false-alarm / over-reactive responding (claim's own phrase for the
hypervigilance signature). Operationalised as the rate of AVOIDANT action --
where "avoidant" is now the SET of grid moves that STRICTLY INCREASE the
Manhattan distance to the nearest hazard cell after the env's own boundary
clamping (see _avoidant_actions; 981 used the single argmax move, which
scored wall-blocked no-ops as avoidance in the far corner) -- taken while the
CURRENT hazard signal (hazard_field_view's centre cell, index 12 of the 5x5
proximity window, world_state channel, ARC-024 proxy-gradient) is in the
AMBIGUOUS band: a low BUT NONZERO reading, not a genuine imminent-contact
reading. This is exactly "excessive/over-reactive responding to non-imminent
signal," the claim's own operationalisation target, as distinct from a
HIGH-band reading (where avoidant response is normatively appropriate, not a
false alarm) or a SAFE reading (low signal; little to react to).

A tick on which NO action increases the distance (the agent is pinned in a
corner of the grid, or is standing on the hazard) is UNSCORABLE: it is
excluded from the band's denominator and counted in n_unscorable. Per band
the run records rate = avoidant/n, chance = mean over scorable ticks of
|avoidant set| / action_dim, and lift = rate - chance. Availability differs
systematically BETWEEN bands (geometry), so any CROSS-BAND reading -- the
positive control -- uses LIFT; C1/C2 are within-band cross-arm comparisons
where availability is matched by construction, so they use the raw RATE.

THRESHOLD CALIBRATION (pre-registered BEFORE the real run, not derived from
this run's own statistics -- see PRE-REGISTERED THRESHOLDS below). env is
CausalGridWorldV2(size=10, num_hazards=1), non-toroidal, hazard_field_decay
default 0.5. hazard_field_view's centre-cell value is
hazard_field[agent_pos] / hazard_field.max() = hazard_field[agent_pos] / 1.0
(the field's global max is always exactly 1.0, attained at the hazard's own
cell, since hazard_field[h] = 1/(1+0*decay) = 1.0 there) = 1/(1 + d*0.5)
where d is the Manhattan distance to the nearest hazard. Inverting:

    value >= 0.50  <=>  d <= 2      -> HIGH
    value <  0.25  <=>  d >= 7      -> SAFE
    otherwise        (3 <= d <= 6)  -> AMBIGUOUS

Both cutpoints are exact analytic consequences of the field's own formula
and the grid's geometry -- computed from the env definition, never from a
scored outcome. They were chosen because the 10x10 non-toroidal grid has a wall border
(walkable interior 8x8, 64 free cells; red-team 2026-09-09 correction of an
earlier "max d 10-18" derivation), so 981's SAFE cutpoint (value < 0.15 <=>
d > 11.33) is empty for a central placement and, otherwise, selects only the
wall-adjacent far corner. d >= 7 and d <= 2 are both populated for every
placement observed (probe: SAFE 9-24, AMBIGUOUS 31-42, HIGH 9-13 of 64 cells). This driver's own
--dry-run smoke prints the observed per-band step counts and the min/max
centre-cell value so the cutpoints can be re-checked against the empirical
distribution before the real run.

NON-TOROIDAL BY DESIGN, AND DELIBERATELY NOT "FIXED" BY WRAPPING. A torus
would remove the boundary confound directly, but CausalGridWorld's
_build_fields computes the hazard field with a NON-toroidal Manhattan
distance (causal_grid_world.py:4602) while step() wraps movement when
toroidal=True. On a torus the agent's true distance and the signal it
receives would therefore disagree, which is a worse confound than the one
being removed. Fix (2a)/(2b) above stay inside the non-toroidal regime,
where field and movement are consistent.

GOV-REUSE-1: checked upstream before authoring -- 0/973 manifests tag
claim_ids containing MECH-027 (no prior MECH-027 run of any kind exists to
reuse or supersede).

SLEEP: use_sleep_loop/sws_enabled/rem_enabled are on (required to build
agent.sleep_loop so draws_per_cycle exists to force/revert -- channel (c)).
During warmup, sleep cycles are driven manually via
agent.sleep_loop.force_cycle() every SLEEP_EVERY_N_EPISODES episodes.
During eval, agent.force_sleep_cycle_at_eval_boundary() is called once at
the end of every eval episode in all three blocks (see channel (c) above
and MECH-027 Build 2) -- this is a deliberate CHANGE from the earlier
"no sleep firing during eval" draft, made because a replay lever has no
observable effect unless a sleep cycle actually consults it. In
EVAL_HYPERVIGILANT the passes are disabled, so that call is a recorded
None-return (offline_passes_enabled = False for the block). The waking
behavioural DV (avoidant-action rate) is still measured only from waking
env steps; the forced sleep cycles run at episode boundaries, between DV
observations, not during them.

PHASED TRAINING: none. This is a direct-perturbation behavioural probe, not
a representation-learning experiment -- no NEW head is trained on z_world/
z_harm/encoder output. The warmup phase trains only the standard E1
prediction loss, E2 world-forward loss, and E3 harm-eval head (the same
always-present substrate training every prior CausalGridWorldV2 script in
this repo trains), so the P0/P1/P2 phased-training discipline (freeze
encoder, train head on .detach()ed latents) does not apply; warmup episode
count (WARMUP_EPISODES) exists solely to reach a functional world model /
harm evaluator before eval, verified via the world_forward_r2 diagnostic.

PRE-REGISTERED THRESHOLDS (all defined as module-level constants below,
fixed before the real run; the --dry-run smoke calibration check above is
what confirmed these values give non-trivial bin coverage on THIS env
config, not what derived them from a scored run's own outcome):
  HAZARD_SAFE_MAX   = 0.25   (hazard_field_view centre < this -> SAFE, d >= 7)
  HAZARD_HIGH_MIN   = 0.50   (>= this -> HIGH, d <= 2; between -> AMBIGUOUS)
  FALSE_ALARM_ELEVATION_ABS = 0.10
      C1 (LOAD-BEARING): EVAL_HYPERVIGILANT's pooled (across-seed mean)
      ambiguous-band avoidant RATE must exceed EVAL_BASELINE's pooled
      ambiguous-band avoidant rate by at least this ABSOLUTE amount, on a DV
      bounded in [0, 1]. This replaces 981's multiplicative 2.0x bar, which
      set an unattainable 1.1542 target on a bounded DV once the baseline
      exceeded 0.5 (autopsy section 4). Feasibility is not assumed: the
      c1_elevation_headroom dv_headroom precondition measures this run's own
      baseline ceiling headroom (1 - max across-seed baseline rate) and
      requires it to be at least HEADROOM_MARGIN_C1 x this bar. The
      non-overlap condition is recorded as the diagnostic
      c1_nonoverlap_also_holds and is NOT part of C1.
  HEADROOM_MARGIN_C1 = 2.0
      The dv_headroom margin for C1: the bar must sit at most half of the
      measured achievable range away, not merely be touchable.
  COMMIT_THRESHOLD_VARIANCE_MULTIPLE = 4.0
      Post-warmup calibration of E3Config.commitment_threshold to
      4 x the substrate's own running_variance, applied identically in all
      three blocks. See "WHAT CHANGED FROM 981" fix (4). Verified, not
      assumed, by the precision_margin_headroom dv_headroom precondition.
  REVERSION_RECOVERY_FLOOR = 0.5
      C2 (LOAD-BEARING): at least 50% of the BASELINE->HYPERVIGILANT
      elevation must revert in the EVAL_REVERSION block. Computed from the
      POOLED rates, not averaged per-seed ratios:
      pooled_recovered_fraction = (mean_hv_rate - mean_rev_rate) /
      (mean_hv_rate - mean_base_rate), must be >= 0.5. (Corrected 2026-09-02,
      red-team F6: a per-seed recovered_fraction is an unbounded ratio that
      can sign-flip on a near-zero or negative per-seed elevation, letting
      one degenerate seed decide C2 while C1 passes cleanly on pooled means
      -- the per-seed-averaged mean_recovered_fraction is still recorded as
      a diagnostic, but is no longer load-bearing.) This is the reversion
      half of the claim's own confirming signature -- required, not
      optional.
  PRECISION_RATIO_FLOOR = 5.0 (P0 readiness, not a claim criterion): pooled
      EVAL_HYPERVIGILANT precision mean / EVAL_BASELINE precision mean.
  HORIZON_RATIO_CEIL = 0.5 (P0 readiness): structural
      effective_horizon under MODE_HV / effective_horizon under MODE_BASE.
  POSITIVE_CONTROL_MARGIN = 0.05 (P0 readiness): EVAL_BASELINE's own
      avoidant rate must be higher in the HIGH band than the SAFE band by at
      least this much -- confirms the DV/env pairing is hazard-sensitive at
      all before trusting anything built on top of it.
  MIN_BIN_COVERAGE_STEPS_PER_SEED = 30 / MIN_BIN_COVERAGE_STEPS_POOLED = 150
      (P0 readiness; see the 2026-09-09 B3 repair note and the Gate A entry
      below for the derivation from the positive-control bar).

  MECH-027 Build 1/2 readiness thresholds (P0, added 2026-09-02 -- see
  "GRADED DOWNSTREAM CONSUMER" / "SLEEP CYCLES ACTUALLY FIRE DURING EVAL"
  above; these gate the two new mechanisms actually engaging, distinct from
  the raw-manipulation checks above which only confirm the forcing itself):
  PRECISION_SCALED_COMMIT_ENGAGED_FRACTION_FLOOR = 0.05 (P0 readiness):
      fraction of real E3 selections (across HV+BASELINE, all seeds) where
      last_score_diagnostics["precision_scaled_commit_active"] was True --
      confirms the standalone committed branch (the only branch this lever
      is wired at) engaged on a MEANINGFUL share of selections, not
      silently bypassed by an inactive Factor-A shortlist / loop-
      segregation path on all but a negligible few ticks. (Corrected
      2026-09-02, red-team F3: an earlier raw count>=1 floor passed on
      negligible engagement, far too rare to plausibly drive a 2x DV
      elevation.)
  PRECISION_MARGIN_HV_ELEVATION_FLOOR = 0.05 (P0 readiness): pooled mean
      precision_margin_norm under EVAL_HYPERVIGILANT minus pooled mean under
      EVAL_BASELINE, must be positive and clear this floor -- confirms the
      forced precision perturbation genuinely moved the graded-consumer
      input, not just the raw _running_variance number (guards against a
      baseline that is already saturated near precision_margin_norm~1,
      leaving no room for HV to differ).
  (COMMIT_TEMPERATURE_HV_REDUCTION_FLOOR is DELETED in 981a. T_eff =
      base_temperature + alpha * (1 - precision_margin_norm) with alpha =
      1.0, so a commit-temperature reduction check is algebraically the same
      quantity as the precision-margin elevation check wearing a different
      name -- it gave false reassurance of independent confirmation and was
      equally unreachable (autopsy section 4 and learning 5).
      commit_temperature_eff_hv_reduction is still RECORDED, as a
      diagnostic metric only.)
  POSITIVE_CONTROL_MARGIN = 0.05 (P0 readiness, Gate A): EVAL_BASELINE's own
      avoidance LIFT (rate minus availability-matched chance) must be higher
      in the HIGH band than in the SAFE band by at least this much. 981 read
      raw RATES here and measured -0.4307; lift is what makes the two bands
      comparable at all, since their chance rates differ by geometry.
  MIN_BIN_COVERAGE_STEPS_PER_SEED = 30 (P0 readiness, Gate A and Gate B):
      every (block, hazard-bin) cell, per seed, must have at least this many
      SCORABLE observed steps -- a non-vacuity floor (one tick moves a rate
      by <= 3.3%). 981's floor of 5 admitted a 38-step HIGH bin; the 981a
      draft's 150 was unmet on all three seeds by 981's own measured HIGH
      counts 139/38/52 (B3). Scaled by the eval-tick budget under --dry-run.
  MIN_BIN_COVERAGE_STEPS_POOLED = 150 (P0 readiness, Gate A): the HIGH and
      SAFE bands, POOLED across seeds in EVAL_BASELINE, must each have at
      least this many scorable steps. Derived from the positive-control bar:
      a binary-rate SE is <= 0.5/sqrt(n), and a +0.05 lift margin needs a
      pooled SE of about 0.04, i.e. n >= 150. 981 measured 229 pooled HIGH.
      Scaled under --dry-run.
  SLEEP_CYCLE_FIRE_FLOOR = 1 (P0 readiness): every seed, in EVAL_BASELINE
      and EVAL_REVERSION, must show at least this many non-None
      agent.force_sleep_cycle_at_eval_boundary() returns -- confirms the
      eval-boundary interleave actually fired (use_sleep_loop reachable),
      not silently no-op'd. EVAL_HYPERVIGILANT is gated the other way (below).
  SWS_WRITES_BASELINE_FLOOR = 1 (P0 readiness): min sws_n_writes MEASURED
      (from the force_sleep_cycle_at_eval_boundary() return) across every
      EVAL_BASELINE firing (Gate A) and every EVAL_REVERSION firing (Gate B)
      -- the offline replay channel is engaged for real on every firing, not
      nominally unsuppressed. Deterministic given a populated buffer.
  SWS_WRITES_HV_CEIL = 0 (P0 readiness, Gate B): max sws_n_writes across
      EVAL_HYPERVIGILANT boundaries (a None return counts as 0) must be this
      or below, AND hv_offline_passes_suppressed_every_boundary requires the
      None-return fraction over HV boundaries to be 1.0 -- suppression was
      exercised at every boundary a cycle would otherwise have fired.
  mech285_n_draws is RECORDED per firing as a diagnostic in every block but
      gates nothing (see B1/B2).

Overall verdict: PASS iff readiness (all P0 preconditions met) AND C1 AND C2.
FAIL with label naming which of the three failed. Any unmet P0 precondition
routes the whole run to substrate_not_ready_requeue (evidence_direction
non_contributory), never a claim-pressure verdict, per the claim's own
non-degeneracy precondition.
"""

from __future__ import annotations

import argparse
import copy
import random
import time
import types
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from experiment_protocol import emit_outcome  # noqa: E402
from experiments._lib.arm_fingerprint import (  # noqa: E402
    compute_arm_fingerprint,
    reset_all_rng,
)
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator  # noqa: E402
from experiments._metrics import (  # noqa: E402
    check_degeneracy,
    dv_headroom_check,
    dv_headroom_observation_check,
    p0_readiness_gate,
    P0NotReady,
)
from experiments.pack_writer import write_flat_manifest  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402


EXPERIMENT_TYPE = "v3_exq_981a_mech027_control_plane_pathological_modes"
QUEUE_ID = "V3-EXQ-981a"
SUPERSEDES = "V3-EXQ-981"
EXPERIMENT_PURPOSE = "evidence"
CLAIM_IDS = ["MECH-027"]
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"

# Seed 44 excluded (documented reef-config early-death instability, CLAUDE.md).
SEEDS = [11, 23, 37]

# ---- Env -------------------------------------------------------------------
ENV_SIZE = 10
NUM_HAZARDS = 1
NUM_RESOURCES = 2
HAZARD_HARM = 0.05
HAZARD_FIELD_DECAY = 0.5

# ---- Agent dims --------------------------------------------------------------
WORLD_DIM = 32
SELF_DIM = 32

# ---- Training ----------------------------------------------------------------
WARMUP_EPISODES = 200
STEPS_PER_EPISODE = 120
NAV_BIAS = 0.25          # mild bias toward the hazard during warmup so the
                         # agent experiences enough near-hazard states to
                         # learn a meaningful harm-avoidance signal
SLEEP_EVERY_N_EPISODES = 25   # forced sleep-cycle cadence during warmup

# ---- Eval ----------------------------------------------------------------
EVAL_EPISODES_PER_BLOCK = 25
EVAL_STEPS_PER_EPISODE = 120

# ---- Channel-forcing constants (see module docstring "HOW EACH CHANNEL IS
# FORCED") ----
HV_PRECISION_SCALE = 0.05    # multiplicative shrink of running_variance/tick
BASE_PRECISION_SCALE = 1.0
HV_HORIZON_FRAC = 0.2        # CEM elite-scoring-window fraction under MODE_HV
MODE_BASE = "MODE_BASE"
MODE_HV = "MODE_HV"
MECH285_BASELINE_DRAWS = 50
MECH285_HV_DRAWS = 0

# ---- MECH-027 Build 1: graded precision-scaled commit temperature (see
# module docstring "GRADED DOWNSTREAM CONSUMER") -- pre-registered explicitly
# rather than left to E3Config's own defaults, since these are now load-
# bearing for channel (a)'s readiness gates ----
PRECISION_SCALED_COMMIT_ENTROPY_ALPHA = 1.0
PRECISION_SCALED_COMMIT_HARM_FLOOR = 0.25

# ---- Hazard-signal bins (pre-registered; see module docstring "THRESHOLD
# CALIBRATION") ----
HAZARD_SAFE_MAX = 0.25   # value < this -> SAFE  (d >= 7 at decay 0.5)
HAZARD_HIGH_MIN = 0.50   # value >= this -> HIGH  (d <= 2 at decay 0.5)

# ---- Pre-registered criteria thresholds (see module docstring) ----
# 981a: ABSOLUTE, on a DV bounded in [0,1]; 981's multiplicative 2.0x bar was
# unattainable by construction once the baseline exceeded 0.5 (autopsy sec 4).
FALSE_ALARM_ELEVATION_ABS = 0.10
HEADROOM_MARGIN_C1 = 2.0
REVERSION_RECOVERY_FLOOR = 0.5
PRECISION_RATIO_FLOOR = 5.0
HORIZON_RATIO_CEIL = 0.5
POSITIVE_CONTROL_MARGIN = 0.05
# 2026-09-09 B3 repair: per-seed NON-VACUITY floor + pooled POWER floor on the
# positive-control bands, derived from POSITIVE_CONTROL_MARGIN (docstring B3).
MIN_BIN_COVERAGE_STEPS_PER_SEED = 30
MIN_BIN_COVERAGE_STEPS_POOLED = 150
# Post-warmup calibration of the ARC-016 variance-space commit threshold, so
# precision_margin_norm has real dynamic range instead of saturating at
# ~0.9998. Applied identically in every block -- never part of the contrast.
COMMIT_THRESHOLD_VARIANCE_MULTIPLE = 4.0

# ---- MECH-027 Build 1/2 readiness thresholds (see module docstring
# "MECH-027 Build 1/2 readiness thresholds") ----
# PRECISION_SCALED_COMMIT_ENGAGED_FRACTION_FLOOR replaces an earlier raw-
# count floor (red-team F3, verified 2026-09-02): a count>=1 floor passes on
# negligible engagement, which cannot plausibly drive a 2x DV elevation.
PRECISION_SCALED_COMMIT_ENGAGED_FRACTION_FLOOR = 0.05
PRECISION_MARGIN_HV_ELEVATION_FLOOR = 0.05
# COMMIT_TEMPERATURE_HV_REDUCTION_FLOOR deleted in 981a -- algebraically the
# same quantity as the precision-margin elevation (autopsy sec 4/learning 5).
SLEEP_CYCLE_FIRE_FLOOR = 1
# 2026-09-09 B1/B2 repair: channel (c) is gated on the SWS schema-pass write
# count, not on mech285 draws (which are recorded as a diagnostic only).
SWS_WRITES_HV_CEIL = 0
SWS_WRITES_BASELINE_FLOOR = 1
HV_OFFLINE_PASSES_SUPPRESSED_FRACTION_FLOOR = 1.0
# Offline-pass step counts pinned explicitly (REEConfig defaults) so the
# manifest records the exact write count a BASELINE firing must report.
SWS_CONSOLIDATION_STEPS = 5
REM_ATTRIBUTION_STEPS = 10

BLOCK_BASELINE = "EVAL_BASELINE"
BLOCK_HYPERVIGILANT = "EVAL_HYPERVIGILANT"
BLOCK_REVERSION = "EVAL_REVERSION"
# Red-team F4 (fable, 2026-09-09): PAIRED channel-(c) contrast block. Run on a
# deep copy of the post-Stage-A agent + env from the SAME RNG state as
# EVAL_HYPERVIGILANT, with channels (a)+(b) forced identically but the offline
# passes left ON -- so lift(HV) - lift(HV_AB) is the behavioural contribution
# of "suppressed replay" at the same point in the agent's history.
BLOCK_HV_AB = "EVAL_HV_AB_PAIRED"
BLOCKS = (BLOCK_BASELINE, BLOCK_HYPERVIGILANT, BLOCK_REVERSION)
BLOCKS_ALL = BLOCKS + (BLOCK_HV_AB,)
# Red-team dispositions (F2/F3/F4/F5), pre-registered:
MIN_E3_SELECTIONS_PER_BLOCK = 30     # F2: decisions, not ticks (E3 cadence ~1/10 tick); non-vacuity
ORDER_DRIFT_FLOOR = 0.05             # F3: |BASELINE 2nd-half - 1st-half AMBIGUOUS lift| >= half the bar
CHANNEL_C_CONTRAST_FLOOR = 0.02      # F4: |lift(HV) - lift(HV_AB)| below this = channel (c) inert
# F5: channel (a)'s only selection consumer can move a membership rate by at most
# the BASELINE fraction of precision-scaled committed picks that were NOT the
# envelope argmin; a "weakens" null under a smaller range is range-limited.
CHANNEL_A_RANGE_FLOOR = FALSE_ALARM_ELEVATION_ABS
# Depth cap for the pre-deepcopy graph-attached-tensor walk (ported from the
# V3-EXQ-1003 driver, measured failure path depth 8).
_DETACH_MAX_DEPTH = 14

HAZARD_BIN_SAFE = "SAFE"
HAZARD_BIN_AMBIGUOUS = "AMBIGUOUS"
HAZARD_BIN_HIGH = "HIGH"


def build_config(env: CausalGridWorldV2) -> REEConfig:
    """ONE config shared by every arm/block -- only the runtime channel
    values (precision scale, operating_mode, config.sws_enabled /
    config.rem_enabled / sleep_loop.draws_per_cycle) differ between blocks;
    nothing about the agent's structure changes.
    """
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=SELF_DIM,
        world_dim=WORLD_DIM,
        reafference_action_dim=env.action_dim,
        use_event_classifier=True,
        # Sleep/replay channel (c) -- see module docstring. Manual-cycle-loop
        # driver (matches V3-EXQ-909's SLEEP DRIVER convention): K set very
        # high so the automatic K-episode cadence never fires; every firing
        # in this script is an explicit force_cycle() call.
        use_sleep_loop=True,
        sleep_loop_episodes_K=1_000_000,
        sws_enabled=True,
        rem_enabled=True,
        sws_consolidation_steps=SWS_CONSOLIDATION_STEPS,
        rem_attribution_steps=REM_ATTRIBUTION_STEPS,
        use_mech285_sampler=True,
        mech285_draws_per_cycle=MECH285_BASELINE_DRAWS,
        use_mech272_routing=True,
        use_anchor_sets=True,
        # Precision channel (a) graded consumer -- MECH-027 Build 1 (see
        # module docstring "GRADED DOWNSTREAM CONSUMER"). Without this, the
        # HV_PRECISION_SCALE forcing below only moves current_precision as a
        # number; the binary ARC-016 commit gate saturates in a trained
        # substrate and channel (a) has no further observable effect on
        # selection. use_harm_variance_commit stays at its default (False,
        # world-variance commit mode) -- required for precision_margin_norm
        # to be computed at all (see e3_selector.py select()).
        use_precision_scaled_commit_temperature=True,
        precision_scaled_commit_entropy_alpha=PRECISION_SCALED_COMMIT_ENTROPY_ALPHA,
        precision_scaled_commit_harm_floor=PRECISION_SCALED_COMMIT_HARM_FLOOR,
    )
    # Horizon channel (b) -- not reachable through from_dims(); set directly
    # per SD-MECH267-HORIZON-DEPTH (ree_core/utils/config.py HippocampalConfig).
    cfg.hippocampal.mode_conditioning_enabled = True
    cfg.hippocampal.mode_horizon_scale = {MODE_BASE: 1.0, MODE_HV: HV_HORIZON_FRAC}
    # 981a FIX (1): the missing FOURTH replay flag. Without it sense() emits no
    # BoundaryEvents, consume_boundary_events -> write_anchor never runs, and
    # AnchorSet.all_with_dual_trace() is permanently empty -- so the mech285
    # sampler draws draws_per_cycle times and gets None every time
    # (V3-EXQ-981: mech285_n_draws == 0 on all 75 BASELINE firings per block).
    # Default False at ree_core/utils/config.py:2761. MUST be set before
    # REEAgent construction: HippocampalModule builds the segmenter in
    # __init__ (ree_core/hippocampal/module.py:273).
    cfg.hippocampal.use_event_segmenter = True
    return cfg


def config_slice() -> Dict[str, Any]:
    """Exactly what each cell's computation reads -- no acceptance thresholds."""
    return {
        "env": "CausalGridWorldV2",
        "env_size": ENV_SIZE,
        "num_hazards": NUM_HAZARDS,
        "num_resources": NUM_RESOURCES,
        "hazard_harm": HAZARD_HARM,
        "hazard_field_decay": HAZARD_FIELD_DECAY,
        "world_dim": WORLD_DIM,
        "self_dim": SELF_DIM,
        "warmup_episodes": WARMUP_EPISODES,
        "steps_per_episode": STEPS_PER_EPISODE,
        "eval_episodes_per_block": EVAL_EPISODES_PER_BLOCK,
        "hv_precision_scale": HV_PRECISION_SCALE,
        "hv_horizon_frac": HV_HORIZON_FRAC,
        "mech285_baseline_draws": MECH285_BASELINE_DRAWS,
        "mech285_hv_draws": MECH285_HV_DRAWS,
        "sws_consolidation_steps": SWS_CONSOLIDATION_STEPS,
        "rem_attribution_steps": REM_ATTRIBUTION_STEPS,
        "hv_offline_passes_enabled": False,
        "base_offline_passes_enabled": True,
        "blocks": list(BLOCKS_ALL),
        "c1_c2_statistic": "ambiguous_band_lift",
        "hazard_safe_max": HAZARD_SAFE_MAX,
        "hazard_high_min": HAZARD_HIGH_MIN,
        "use_event_segmenter": True,
        "commit_threshold_variance_multiple": COMMIT_THRESHOLD_VARIANCE_MULTIPLE,
        "use_precision_scaled_commit_temperature": True,
        "precision_scaled_commit_entropy_alpha": PRECISION_SCALED_COMMIT_ENTROPY_ALPHA,
        "precision_scaled_commit_harm_floor": PRECISION_SCALED_COMMIT_HARM_FLOOR,
    }


def _random_onehot(action_dim: int, device) -> torch.Tensor:
    v = torch.zeros(1, action_dim, device=device)
    v[0, random.randint(0, action_dim - 1)] = 1.0
    return v


def _hazard_cells(env: CausalGridWorldV2) -> List[Tuple[int, int]]:
    hz = np.argwhere(env.grid == env.ENTITY_TYPES["hazard"])
    return [(int(x), int(y)) for x, y in hz]


def _clamped_step(env: CausalGridWorldV2, ax: int, ay: int, dx: int, dy: int) -> Tuple[int, int]:
    """Where the agent would ACTUALLY end up. Non-toroidal: a move into a wall
    or off the grid is a no-op, so the position is unchanged (mirrors
    causal_grid_world.step()'s own guard). This is what makes a wall-bump
    correctly NOT count as avoidance -- 981's argmax definition scored exactly
    those no-ops as avoidance in the far corner, which is where its reversed
    positive control came from."""
    nx, ny = ax + dx, ay + dy
    if env.toroidal:
        return nx % env.size, ny % env.size
    if not (0 <= nx < env.size and 0 <= ny < env.size):
        return ax, ay
    if env.grid[nx, ny] == env.ENTITY_TYPES["wall"]:
        return ax, ay
    return nx, ny


def _hazard_dist(ax: int, ay: int, hazard_cells: List[Tuple[int, int]]) -> int:
    return min(abs(ax - hx) + abs(ay - hy) for hx, hy in hazard_cells)


def _avoidant_actions(
    env: CausalGridWorldV2, hazard_cells: List[Tuple[int, int]]
) -> Tuple[set, int]:
    """The SET of movement actions that STRICTLY INCREASE the Manhattan
    distance to the nearest hazard, evaluated at the position the env would
    actually move the agent to. Returns (set_of_action_indices, action_dim).

    Read live from env._action_map (not cached) so it stays correct if a
    future config permutes the action map. An EMPTY set means no action can
    increase the distance from here (a corner, or standing on the hazard) --
    the caller treats that tick as UNSCORABLE rather than as a failure to
    avoid.

    981 used the single argmax of the dot product with the away-vector. That
    definition (i) picked one arbitrary member when two moves tied, (ii)
    returned an arbitrary action when the agent stood on the hazard
    (away-vector zero -> every score 0 -> first key wins), and (iii) counted
    wall-blocked no-ops as avoidance. All three are removed here.
    """
    action_dim = int(env.action_dim)
    if not hazard_cells:
        return set(), action_dim
    ax, ay = int(env.agent_x), int(env.agent_y)
    d0 = _hazard_dist(ax, ay, hazard_cells)
    out = set()
    for a, (dx, dy) in env._action_map.items():
        nx, ny = _clamped_step(env, ax, ay, dx, dy)
        if (nx, ny) == (ax, ay):
            continue                      # no-op / wall-bump / stay
        if _hazard_dist(nx, ny, hazard_cells) > d0:
            out.add(int(a))
    return out, action_dim


def _scaled_floor(floor: int, scale: float) -> int:
    """Scale a coverage floor by the eval-tick budget (1.0 at the real run;
    (3*15)/(25*120) under --dry-run) so the gate arithmetic is exercised in the
    smoke. Never below 1."""
    import math
    return max(1, int(math.ceil(float(floor) * float(scale))))


def _hazard_bin(value: float) -> str:
    if value < HAZARD_SAFE_MAX:
        return HAZARD_BIN_SAFE
    if value >= HAZARD_HIGH_MIN:
        return HAZARD_BIN_HIGH
    return HAZARD_BIN_AMBIGUOUS


class _TickState:
    """Multi-rate-clock bookkeeping this driver owns itself, since it calls
    hippocampal.propose_trajectories() directly (bypassing
    agent.generate_trajectories()/_e3_tick(), which never forward
    operating_mode -- see module docstring point (b))."""

    def __init__(self) -> None:
        self.last_action: Optional[torch.Tensor] = None
        self.z_self_prev: Optional[torch.Tensor] = None
        self.action_prev: Optional[torch.Tensor] = None
        self.last_precision: float = 0.0


def _agent_tick(
    agent: REEAgent,
    env: CausalGridWorldV2,
    obs_dict: Dict[str, Any],
    state: _TickState,
    operating_mode: Optional[Dict[str, float]],
    precision_scale: float,
    world_dim: int,
    device,
    train: bool,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict[str, float]]]:
    """One environment tick. Returns (chosen action [1,action_dim], z_world
    [1,world_dim] detached, precision_scaled_commit_diagnostics) -- z_world
    is returned so callers that need it (e.g. the warmup world-forward
    buffer) never have to call agent.sense() a second time for the same
    tick. precision_scaled_commit_diagnostics is a FRESH snapshot of
    {"precision_margin_norm", "precision_scaled_commit_active",
    "precision_scaled_commit_temperature_eff"} taken from
    agent.e3.last_score_diagnostics -- but ONLY on a tick where E3 actually
    ran select() this call (ticks["e3_tick"] True AND candidates non-empty);
    None otherwise. Callers must not treat a None as "zero" -- it means "no
    fresh selection this tick", the same sample-size-integrity discipline as
    state.last_precision/state.last_action (both intentionally re-read across
    skipped ticks; the diagnostics snapshot intentionally is NOT, so pooled
    precision_margin_norm/T_eff stats reflect actual E3 firings only).

    Mirrors REEAgent._e3_tick's essential shape but calls
    hippocampal.propose_trajectories() directly so operating_mode (channel
    (b), the horizon lever) can be injected -- see module docstring. Channel
    (a) (precision) is forced AFTER agent.update_residue() drives its normal
    live update, so the channel stays non-degenerate (real per-tick
    movement) rather than frozen.
    """
    obs_body = obs_dict["body_state"]
    obs_world = obs_dict["world_state"]
    ctx = torch.no_grad() if not train else _nullcontext()
    precision_diag: Optional[Dict[str, float]] = None
    with ctx:
        latent = agent.sense(obs_body, obs_world)
        if state.z_self_prev is not None and state.action_prev is not None:
            agent.record_transition(state.z_self_prev, state.action_prev, latent.z_self.detach())
        ticks = agent.clock.advance()
        e1_prior = (
            agent._e1_tick(latent)
            if ticks.get("e1_tick", False)
            else torch.zeros(1, world_dim, device=device)
        )
        if ticks.get("e3_tick", False):
            candidates = agent.hippocampal.propose_trajectories(
                latent.z_world,
                latent.z_self,
                e1_prior=e1_prior,
                operating_mode=operating_mode,
            )
            if candidates:
                result = agent.e3.select(candidates, temperature=1.0)
                action = result.selected_action.detach()
                state.last_action = action
                state.last_precision = float(result.precision)
                diag = agent.e3.last_score_diagnostics
                precision_diag = {
                    "precision_margin_norm": float(diag.get("precision_margin_norm", -1.0)),
                    "precision_scaled_commit_active": bool(
                        diag.get("precision_scaled_commit_active", False)
                    ),
                    "precision_scaled_commit_temperature_eff": float(
                        diag.get("precision_scaled_commit_temperature_eff", -1.0)
                    ),
                }
                # Red-team F5: was the graded pick the envelope argmin? The
                # non-argmin fraction in BASELINE bounds what going COLD in HV
                # can change through channel (a). last_selected_idx and
                # last_scores are written unconditionally by select().
                try:
                    _ls = agent.e3.last_scores
                    _li = agent.e3.last_selected_idx
                    precision_diag["precision_scaled_pick_is_argmin"] = bool(
                        _ls is not None and _li is not None
                        and int(_li) == int(_ls.argmin().item()))
                except Exception:
                    precision_diag["precision_scaled_pick_is_argmin"] = True
        action = state.last_action
        if action is None:
            action = _random_onehot(env.action_dim, device)
            state.last_action = action

        drive_level = REEAgent.compute_drive_level(obs_body)
        benefit_exposure = max(0.0, float(obs_dict.get("benefit_exposure", 0.0)))
        agent.update_z_goal(benefit_exposure=benefit_exposure, drive_level=drive_level)

    state.z_self_prev = latent.z_self.detach()
    state.action_prev = action.detach()
    return action, latent.z_world.detach(), precision_diag


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


def _apply_precision_scale(agent: REEAgent, scale: float) -> None:
    """Channel (a): force _running_variance AFTER its normal live update, so
    the channel is perturbed but never frozen. scale=1.0 is a no-op."""
    if scale == 1.0:
        return
    rv = float(agent.e3._running_variance)
    agent.e3._running_variance = max(1e-9, rv * scale)


def _train_warmup(
    agent: REEAgent,
    env: CausalGridWorldV2,
    optimizer: optim.Optimizer,
    wf_optimizer: optim.Optimizer,
    harm_eval_optimizer: optim.Optimizer,
    num_episodes: int,
    steps_per_episode: int,
    world_dim: int,
    device,
) -> Dict[str, Any]:
    agent.train()
    state = _TickState()
    harm_buf_pos: List[torch.Tensor] = []
    harm_buf_neg: List[torch.Tensor] = []
    wf_buf: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    total_harm = 0
    sleep_fires = 0

    for ep in range(num_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()
        state = _TickState()
        z_world_prev: Optional[torch.Tensor] = None

        for _ in range(steps_per_episode):
            action, z_world_curr_pre, _precision_diag = _agent_tick(
                agent, env, obs_dict, state,
                operating_mode={MODE_BASE: 1.0},
                precision_scale=BASE_PRECISION_SCALE,
                world_dim=world_dim, device=device, train=True,
            )

            # nav_bias: with probability NAV_BIAS, override toward the
            # nearest hazard so training sees enough near-hazard states.
            if random.random() < NAV_BIAS:
                hz = _hazard_cells(env)
                if hz:
                    ax, ay = int(env.agent_x), int(env.agent_y)
                    hx, hy = min(hz, key=lambda h: abs(h[0] - ax) + abs(h[1] - ay))
                    dx, dy = hx - ax, hy - ay
                    best_a, best_score = None, -1e18
                    for a, (adx, ady) in env._action_map.items():
                        score = adx * dx + ady * dy
                        if score > best_score:
                            best_score = score
                            best_a = a
                    if best_a is not None:
                        action = _random_onehot(env.action_dim, device) * 0.0
                        action[0, best_a] = 1.0
                        state.last_action = action
                        state.action_prev = action.detach()

            flat_obs, harm_signal, done, info, obs_dict = env.step(action)
            agent.update_residue(
                harm_signal=float(harm_signal), world_delta=None,
                hypothesis_tag=False, owned=True,
            )

            theta_z = agent.theta_buffer.summary()
            if z_world_prev is not None:
                wf_buf.append((z_world_prev.cpu(), state.action_prev.cpu(), z_world_curr_pre.cpu()))
                if len(wf_buf) > 2000:
                    wf_buf = wf_buf[-2000:]
            if harm_signal < 0:
                total_harm += 1
                harm_buf_pos.append(theta_z.detach())
                if len(harm_buf_pos) > 1000:
                    harm_buf_pos = harm_buf_pos[-1000:]
            else:
                harm_buf_neg.append(theta_z.detach())
                if len(harm_buf_neg) > 1000:
                    harm_buf_neg = harm_buf_neg[-1000:]

            e1_loss = agent.compute_prediction_loss()
            if e1_loss.requires_grad:
                optimizer.zero_grad()
                e1_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.e1.parameters(), 1.0)
                optimizer.step()

            if len(wf_buf) >= 16:
                k = min(32, len(wf_buf))
                idxs = torch.randperm(len(wf_buf))[:k].tolist()
                zw_b = torch.cat([wf_buf[i][0] for i in idxs]).to(device)
                a_b = torch.cat([wf_buf[i][1] for i in idxs]).to(device)
                zw1_b = torch.cat([wf_buf[i][2] for i in idxs]).to(device)
                wf_loss = F.mse_loss(agent.e2.world_forward(zw_b, a_b), zw1_b)
                if wf_loss.requires_grad:
                    wf_optimizer.zero_grad()
                    wf_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        list(agent.e2.world_transition.parameters())
                        + list(agent.e2.world_action_encoder.parameters()), 1.0,
                    )
                    wf_optimizer.step()

            if len(harm_buf_pos) >= 4 and len(harm_buf_neg) >= 4:
                k_p = min(16, len(harm_buf_pos))
                k_n = min(16, len(harm_buf_neg))
                pi = torch.randperm(len(harm_buf_pos))[:k_p].tolist()
                ni = torch.randperm(len(harm_buf_neg))[:k_n].tolist()
                zw_b = torch.cat([harm_buf_pos[i] for i in pi] + [harm_buf_neg[i] for i in ni], dim=0)
                target = torch.cat([
                    torch.ones(k_p, 1, device=device), torch.zeros(k_n, 1, device=device),
                ], dim=0)
                pred = agent.e3.harm_eval(zw_b)
                harm_loss = F.mse_loss(pred, target)
                if harm_loss.requires_grad:
                    harm_eval_optimizer.zero_grad()
                    harm_loss.backward()
                    torch.nn.utils.clip_grad_norm_(agent.e3.harm_eval_head.parameters(), 0.5)
                    harm_eval_optimizer.step()

            z_world_prev = z_world_curr_pre
            if done:
                break

        if (ep + 1) % 50 == 0 or ep == num_episodes - 1:
            print(f"  [train] ep {ep+1}/{num_episodes}  harm={total_harm}  sleep_fires={sleep_fires}", flush=True)

        if (ep + 1) % SLEEP_EVERY_N_EPISODES == 0:
            try:
                agent._flush_exploration_episode()
            except AttributeError:
                pass
            agent.sleep_loop.force_cycle(agent)
            sleep_fires += 1

    return {"total_harm": total_harm, "wf_buf": wf_buf, "sleep_fires": sleep_fires}


def _compute_world_forward_r2(agent: REEAgent, wf_buf: List, n_test: int = 200) -> float:
    if len(wf_buf) < n_test:
        return 0.0
    idxs = list(range(len(wf_buf) - n_test, len(wf_buf)))
    with torch.no_grad():
        zw = torch.cat([wf_buf[i][0] for i in idxs])
        a = torch.cat([wf_buf[i][1] for i in idxs])
        zw1 = torch.cat([wf_buf[i][2] for i in idxs])
        pred = agent.e2.world_forward(zw, a)
        ss_res = ((zw1 - pred) ** 2).sum()
        ss_tot = ((zw1 - zw1.mean(dim=0, keepdim=True)) ** 2).sum()
    return float((1 - ss_res / (ss_tot + 1e-8)).item())


def _detach_agent_buffers(agent: REEAgent) -> Tuple[int, Dict[int, Any]]:
    """Detach graph-attached tensors anywhere in the agent's object graph so
    copy.deepcopy can take the post-Stage-A snapshot for the paired channel-(c)
    block (red-team F4). Ported verbatim in mechanism from the V3-EXQ-1003
    driver (2026-09-04): walks containers, nn.Module._modules/_buffers and any
    object's __dict__/__slots__, id-keyed seen set, depth cap; returns
    (n_detached, deepcopy_memo) with every reachable python MODULE object memo'd
    to itself (deepcopy cannot pickle a module). Detaching is safe here: the
    snapshot is taken after warmup, and no backward pass is ever taken through
    eval-block state. nn.Parameters are returned untouched."""
    n = 0
    seen: set = set()
    memo: Dict[int, Any] = {}

    def _fix(obj: Any, depth: int) -> Any:
        nonlocal n
        if depth > _DETACH_MAX_DEPTH:
            return obj
        if isinstance(obj, torch.Tensor):
            if isinstance(obj, torch.nn.Parameter):
                return obj
            if obj.grad_fn is not None:
                n += 1
                return obj.detach()
            return obj
        if obj is None or isinstance(obj, (str, bytes, bytearray, int, float,
                                           bool, complex, np.ndarray)):
            return obj
        if isinstance(obj, types.ModuleType):
            memo[id(obj)] = obj
            return obj
        if isinstance(obj, (type, types.FunctionType, types.MethodType,
                            types.BuiltinFunctionType)):
            return obj
        oid = id(obj)
        if oid in seen:
            return obj
        seen.add(oid)
        if isinstance(obj, dict):
            for k in list(obj.keys()):
                obj[k] = _fix(obj[k], depth + 1)
            return obj
        if isinstance(obj, deque):
            items = [_fix(v, depth + 1) for v in list(obj)]
            obj.clear()
            obj.extend(items)
            return obj
        if isinstance(obj, list):
            for i, v in enumerate(obj):
                obj[i] = _fix(v, depth + 1)
            return obj
        if isinstance(obj, tuple):
            return tuple(_fix(v, depth + 1) for v in obj)
        if isinstance(obj, (set, frozenset)):
            return obj
        d = getattr(obj, "__dict__", None)
        if isinstance(d, dict):
            for k in list(d.keys()):
                if k in ("_parameters", "_non_persistent_buffers_set"):
                    continue
                d[k] = _fix(d[k], depth + 1)
        for slot in getattr(type(obj), "__slots__", ()) or ():
            if isinstance(slot, str) and hasattr(obj, slot):
                try:
                    setattr(obj, slot, _fix(getattr(obj, slot), depth + 1))
                except Exception:
                    pass
        return obj

    _fix(agent, 0)
    return n, memo


def _set_offline_passes(agent: REEAgent, enabled: bool) -> None:
    """Channel (c) lever (2026-09-09): enable/disable the SD-017 SWS schema
    pass and REM attribution pass on the LIVE agent. agent.run_sleep_cycle()
    reads config.sws_enabled / config.rem_enabled at call time, so this is
    runtime-revertible with no rebuild. NOT sws_consolidation_steps = 0,
    which divides by zero in run_sws_schema_pass."""
    agent.config.sws_enabled = bool(enabled)
    agent.config.rem_enabled = bool(enabled)


def _run_eval_block(
    agent: REEAgent,
    env: CausalGridWorldV2,
    block: str,
    num_episodes: int,
    steps_per_episode: int,
    world_dim: int,
    device,
    zg: ZGoalStreamAccumulator,
) -> Dict[str, Any]:
    """Run one eval regime block on the (already-trained) agent. Applies the
    block's channel settings (precision scale, operating_mode, offline-pass
    flags + replay draw count) at block ENTRY -- this is what makes EVAL_REVERSION a genuine
    within-run reversion rather than a fresh agent.

    MECH-027 Build 1/2 (2026-09-02): also captures, per real E3 selection,
    the precision-scaled-commit-temperature diagnostics
    (precision_margin_norm / precision_scaled_commit_temperature_eff /
    precision_scaled_commit_active) from agent.e3.last_score_diagnostics via
    _agent_tick's third return value; and fires a real sleep cycle at every
    eval episode boundary via agent.force_sleep_cycle_at_eval_boundary(),
    recording each firing's MEASURED mech285_n_draws -- see module docstring
    "GRADED DOWNSTREAM CONSUMER" and "SLEEP CYCLES ACTUALLY FIRE DURING
    EVAL"."""
    if block == BLOCK_BASELINE:
        operating_mode = {MODE_BASE: 1.0}
        precision_scale = BASE_PRECISION_SCALE
        agent.sleep_loop.draws_per_cycle = MECH285_BASELINE_DRAWS
        offline_passes_enabled = True
    elif block == BLOCK_HYPERVIGILANT:
        operating_mode = {MODE_HV: 1.0}
        precision_scale = HV_PRECISION_SCALE
        agent.sleep_loop.draws_per_cycle = MECH285_HV_DRAWS
        offline_passes_enabled = False
    elif block == BLOCK_REVERSION:
        operating_mode = {MODE_BASE: 1.0}
        precision_scale = BASE_PRECISION_SCALE
        agent.sleep_loop.draws_per_cycle = MECH285_BASELINE_DRAWS
        offline_passes_enabled = True
    elif block == BLOCK_HV_AB:
        # F4 paired block: (a)+(b) forced exactly as EVAL_HYPERVIGILANT, (c) NOT
        # suppressed. Runs on the deep-copied post-Stage-A agent (run_seed_stage_b).
        operating_mode = {MODE_HV: 1.0}
        precision_scale = HV_PRECISION_SCALE
        agent.sleep_loop.draws_per_cycle = MECH285_BASELINE_DRAWS
        offline_passes_enabled = True
    else:
        raise ValueError(f"unknown block {block!r}")
    # Channel (c) lever (2026-09-09 repair, docstring B1/B2): the SD-017
    # offline passes themselves. run_sleep_cycle reads these flags live.
    _set_offline_passes(agent, offline_passes_enabled)

    agent.eval()
    state = _TickState()
    step_rows: List[Dict[str, Any]] = []
    precision_samples: List[float] = []
    effective_horizon_samples: List[float] = []
    precision_margin_samples: List[float] = []
    commit_temp_eff_samples: List[float] = []
    precision_scaled_commit_engaged_count = 0
    e3_selection_count = 0
    pick_active_count = 0
    pick_nonargmin_count = 0
    sleep_fires = 0
    sleep_boundary_calls = 0
    sleep_cycle_none_returns = 0
    sleep_mech285_draws: List[int] = []
    sleep_sws_writes: List[int] = []
    sleep_rem_rollouts: List[int] = []
    fatal = 0

    for ep in range(num_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()
        state = _TickState()

        for step_idx in range(steps_per_episode):
            hazard_cells = _hazard_cells(env)
            hv = obs_dict.get("hazard_field_view", None)
            hazard_value = float(hv[12]) if hv is not None else 0.0
            hbin = _hazard_bin(hazard_value)
            avoidant_set, action_dim_now = _avoidant_actions(env, hazard_cells)

            try:
                action, _zw, precision_diag = _agent_tick(
                    agent, env, obs_dict, state,
                    operating_mode=operating_mode,
                    precision_scale=precision_scale,
                    world_dim=world_dim, device=device, train=False,
                )
            except Exception:
                fatal += 1
                action = _random_onehot(env.action_dim, device)
                state.last_action = action
                precision_diag = None

            if precision_diag is not None:
                e3_selection_count += 1
                # precision_margin_norm is valid whenever the world-variance
                # commit gate ran with effective_threshold>0 -- independent
                # of whether the precision-scaled-commit BRANCH specifically
                # engaged (see module docstring). Pool it whenever it is a
                # real reading (not the -1.0 "not computed" sentinel).
                margin = precision_diag["precision_margin_norm"]
                if margin >= 0.0:
                    precision_margin_samples.append(margin)
                # precision_scaled_commit_temperature_eff, in contrast, is
                # ONLY set when the branch actually fired (envelope admitted
                # >=2 candidates); pooling it unconditionally would mix real
                # T_eff readings with the -1.0 "branch did not engage"
                # sentinel and silently bias the pooled mean (caught in
                # this script's own smoke test: an unfiltered pool put the
                # BASELINE mean below 1.0 while HV's mean, which happened to
                # engage on every tick in the tiny dry run, read ~1.0 --
                # backwards from the predicted direction).
                if precision_diag["precision_scaled_commit_active"]:
                    precision_scaled_commit_engaged_count += 1
                    commit_temp_eff_samples.append(
                        precision_diag["precision_scaled_commit_temperature_eff"]
                    )
                    pick_active_count += 1
                    if not precision_diag.get("precision_scaled_pick_is_argmin", True):
                        pick_nonargmin_count += 1

            chosen_idx = int(action.argmax(dim=-1).item())
            # A tick with NO distance-increasing action is UNSCORABLE (see
            # _avoidant_actions): it is excluded from the band denominator
            # rather than recorded as a failure to avoid.
            scorable = bool(avoidant_set)
            took_avoidant = bool(scorable and chosen_idx in avoidant_set)

            step_rows.append({
                "ep": ep,
                "hazard_value": hazard_value,
                "hazard_bin": hbin,
                "scorable": scorable,
                "took_avoidant": took_avoidant,
                "n_avoidant_available": len(avoidant_set),
                "action_dim": action_dim_now,
            })

            # Sample the CEM scoring window only on ticks where a proposal ran
            # (red-team CLEAR-nit: a stale value otherwise leaks across the
            # block boundary on the first held tick).
            if precision_diag is not None:
                eh = getattr(agent.hippocampal, "_last_effective_horizon", None)
                if eh is not None:
                    effective_horizon_samples.append(float(eh))

            flat_obs, harm_signal, done, info, obs_dict = env.step(action)
            agent.update_residue(
                harm_signal=float(harm_signal), world_delta=None,
                hypothesis_tag=False, owned=True,
            )
            _apply_precision_scale(agent, precision_scale)
            precision_samples.append(float(agent.e3.current_precision))

            if done:
                break

        # MECH-027 Build 2 -- fire a real sleep cycle at this eval episode
        # boundary (BEFORE the next iteration's agent.reset(), which would
        # otherwise flush an already-empty exploration buffer). See module
        # docstring "SLEEP CYCLES ACTUALLY FIRE DURING EVAL". This is what
        # makes the block's draws_per_cycle setting (above) an exercised
        # lever rather than a dead parameter: mech285_n_draws in the
        # returned metrics is the MEASURED replay count for this firing.
        sleep_boundary_calls += 1
        sleep_metrics = agent.force_sleep_cycle_at_eval_boundary()
        if sleep_metrics is not None:
            sleep_fires += 1
            sleep_mech285_draws.append(int(sleep_metrics.get("mech285_n_draws", -1)))
            sleep_sws_writes.append(int(sleep_metrics.get("sws_n_writes", 0)))
            sleep_rem_rollouts.append(int(sleep_metrics.get("rem_n_rollouts", 0)))
        else:
            # Passes disabled for this block (HV): the substrate reports the
            # suppressed cycle as a None return. Recorded as 0 writes so the
            # HV ceiling gate reads a number, plus the None count itself.
            sleep_cycle_none_returns += 1
            sleep_sws_writes.append(0)
            sleep_rem_rollouts.append(0)

        print(f"  [eval] block={block} ep {ep+1}/{num_episodes} steps_logged={len(step_rows)}", flush=True)

    zg.observe(agent)

    bins: Dict[str, Dict[str, Any]] = {
        k: {"n": 0, "avoidant": 0, "n_unscorable": 0, "avail_frac_sum": 0.0}
        for k in (HAZARD_BIN_SAFE, HAZARD_BIN_AMBIGUOUS, HAZARD_BIN_HIGH)
    }
    for row in step_rows:
        b = bins[row["hazard_bin"]]
        if not row["scorable"]:
            b["n_unscorable"] += 1
            continue
        b["n"] += 1
        b["avail_frac_sum"] += (
            row["n_avoidant_available"] / float(max(1, row["action_dim"]))
        )
        if row["took_avoidant"]:
            b["avoidant"] += 1

    # Red-team F3: per-episode AMBIGUOUS-band series (lift), so a monotone
    # drift across the block order is visible and testable post hoc.
    per_episode_ambiguous_lift: List[float] = []
    per_episode_ambiguous_n: List[int] = []
    for ep_i in range(num_episodes):
        n_ep = av_ep = 0
        avail_ep = 0.0
        for row in step_rows:
            if row["ep"] != ep_i or row["hazard_bin"] != HAZARD_BIN_AMBIGUOUS or not row["scorable"]:
                continue
            n_ep += 1
            avail_ep += row["n_avoidant_available"] / float(max(1, row["action_dim"]))
            if row["took_avoidant"]:
                av_ep += 1
        per_episode_ambiguous_n.append(n_ep)
        per_episode_ambiguous_lift.append(
            (av_ep / n_ep - avail_ep / n_ep) if n_ep > 0 else float("nan"))

    rates = {
        k: (v["avoidant"] / v["n"] if v["n"] > 0 else 0.0) for k, v in bins.items()
    }
    # Availability-matched chance rate. The number of distance-increasing
    # actions varies systematically BETWEEN bands (geometry), so any
    # cross-band reading must be taken on the lift, not the raw rate.
    chance = {
        k: (v["avail_frac_sum"] / v["n"] if v["n"] > 0 else 0.0)
        for k, v in bins.items()
    }
    lifts = {k: rates[k] - chance[k] for k in rates}

    return {
        "block": block,
        "n_steps": len(step_rows),
        "bins": bins,
        "rates": rates,
        "chance": chance,
        "lifts": lifts,
        "precision_mean": float(np.mean(precision_samples)) if precision_samples else 0.0,
        "precision_samples": precision_samples,
        "effective_horizon_mean": (
            float(np.mean(effective_horizon_samples)) if effective_horizon_samples else None
        ),
        "draws_per_cycle": int(agent.sleep_loop.draws_per_cycle),
        "offline_passes_enabled": bool(offline_passes_enabled),
        "fatal_errors": fatal,
        "n_ticks": len(step_rows),
        "per_episode_ambiguous_lift": per_episode_ambiguous_lift,
        "per_episode_ambiguous_n": per_episode_ambiguous_n,
        # F2: an over-avoidant agent pinned in a corner is UNSCORABLE; count it.
        "unscorable_in_safe": int(bins[HAZARD_BIN_SAFE]["n_unscorable"]),
        "unscorable_total": int(sum(v["n_unscorable"] for v in bins.values())),
        # F5: channel (a) selection range.
        "precision_scaled_pick_active_count": int(pick_active_count),
        "precision_scaled_pick_nonargmin_count": int(pick_nonargmin_count),
        "precision_scaled_pick_nonargmin_frac": (
            pick_nonargmin_count / float(pick_active_count) if pick_active_count else -1.0),
        # MECH-027 Build 1 -- graded precision-scaled commit temperature.
        "precision_margin_norm_mean": (
            float(np.mean(precision_margin_samples)) if precision_margin_samples else -1.0
        ),
        "precision_margin_norm_samples": precision_margin_samples,
        "commit_temperature_eff_mean": (
            float(np.mean(commit_temp_eff_samples)) if commit_temp_eff_samples else -1.0
        ),
        "commit_temperature_eff_samples": commit_temp_eff_samples,
        "precision_scaled_commit_engaged_count": precision_scaled_commit_engaged_count,
        "e3_selection_count": e3_selection_count,
        # MECH-027 Build 2 -- eval-boundary sleep-cycle interleave.
        "sleep_fires": sleep_fires,
        "sleep_mech285_draws": sleep_mech285_draws,
        "sleep_mech285_draws_max": (
            max(sleep_mech285_draws) if sleep_mech285_draws else -1
        ),
        "sleep_mech285_draws_min": (
            min(sleep_mech285_draws) if sleep_mech285_draws else -1
        ),
        # 2026-09-09 repair: channel (c) engagement is read off the SWS
        # schema-pass write count (deterministic when engaged), per firing.
        "sleep_boundary_calls": sleep_boundary_calls,
        "sleep_cycle_none_returns": sleep_cycle_none_returns,
        "sleep_sws_writes": sleep_sws_writes,
        "sleep_sws_writes_min": (min(sleep_sws_writes) if sleep_sws_writes else -1),
        "sleep_sws_writes_max": (max(sleep_sws_writes) if sleep_sws_writes else -1),
        "sleep_rem_rollouts": sleep_rem_rollouts,
        "sleep_rem_rollouts_mean": (
            float(np.mean(sleep_rem_rollouts)) if sleep_rem_rollouts else -1.0
        ),
    }



def _calibrate_commit_threshold(agent: REEAgent) -> Dict[str, float]:
    """981a FIX (4): give precision_margin_norm real dynamic range.

    precision_margin_norm = clamp(1 - commit_variance/effective_threshold, 0, 1)
    and effective_threshold derives from E3Config.commitment_threshold (default
    0.40), while a trained substrate's running_variance sits ~5000x below it --
    so 981 measured a BASELINE margin of 0.99980 and the graded commit
    temperature T_eff = 1.0 + alpha*(1 - margin) was pinned within 2e-4 of
    base_temperature. Channel (a) therefore had no behavioural consumer at all.

    This sets commitment_threshold = COMMIT_THRESHOLD_VARIANCE_MULTIPLE x the
    substrate's OWN post-warmup running_variance, which places the baseline
    margin near 1 - 1/multiple and leaves the rest of [0,1] as headroom. It is a
    pre-registered RULE, not a tuned number, it is applied identically in all
    three blocks (so it is never part of the arm contrast), and it is VERIFIED
    rather than assumed by the precision_margin_headroom dv_headroom
    precondition in Gate A.
    """
    rv = float(agent.e3._running_variance)
    old = float(agent.e3.config.commitment_threshold)
    new = max(1e-9, COMMIT_THRESHOLD_VARIANCE_MULTIPLE * rv)
    agent.e3.config.commitment_threshold = new
    print(f"  [calib] commitment_threshold {old:.6g} -> {new:.6g}"
          f"  (running_variance={rv:.6g}, x{COMMIT_THRESHOLD_VARIANCE_MULTIPLE})",
          flush=True)
    return {"running_variance_post_warmup": rv,
            "commitment_threshold_before": old,
            "commitment_threshold_after": new}


def run_seed_stage_a(seed: int, dry_run: bool) -> Dict[str, Any]:
    """Warmup + commit-threshold calibration + EVAL_BASELINE only.

    Stage A exists so the DV/env pairing is established BEFORE anything is built
    on it (autopsy section 6 fix 3): Gate A is evaluated on Stage A's output and
    EVAL_HYPERVIGILANT / EVAL_REVERSION never run if it is unmet.
    """
    device = torch.device("cpu")
    reset_all_rng(seed)

    print(f"\nSeed {seed} Condition control_plane_hypervigilance_probe", flush=True)

    env = CausalGridWorldV2(
        seed=seed, size=ENV_SIZE, num_hazards=NUM_HAZARDS, num_resources=NUM_RESOURCES,
        hazard_harm=HAZARD_HARM, hazard_field_decay=HAZARD_FIELD_DECAY,
    )
    cfg = build_config(env)
    agent = REEAgent(cfg).to(device)

    optimizer = optim.Adam(list(agent.e1.parameters()), lr=1e-3)
    wf_optimizer = optim.Adam(
        list(agent.e2.world_transition.parameters()) + list(agent.e2.world_action_encoder.parameters()),
        lr=1e-3,
    )
    harm_eval_optimizer = optim.Adam(list(agent.e3.harm_eval_head.parameters()), lr=1e-4)

    warmup_eps = 5 if dry_run else WARMUP_EPISODES
    warmup_steps = 15 if dry_run else STEPS_PER_EPISODE
    eval_eps = 3 if dry_run else EVAL_EPISODES_PER_BLOCK
    eval_steps = 15 if dry_run else EVAL_STEPS_PER_EPISODE

    train_out = _train_warmup(
        agent, env, optimizer, wf_optimizer, harm_eval_optimizer,
        warmup_eps, warmup_steps, WORLD_DIM, device,
    )
    world_forward_r2 = _compute_world_forward_r2(agent, train_out["wf_buf"])
    print(f"  world_forward_r2={world_forward_r2:.4f}  sleep_fires={train_out['sleep_fires']}"
          f"  draws_per_cycle(post-warmup)={agent.sleep_loop.draws_per_cycle}", flush=True)

    calib = _calibrate_commit_threshold(agent)

    zg = ZGoalStreamAccumulator()
    block_results: Dict[str, Dict[str, Any]] = {}
    block_results[BLOCK_BASELINE] = _run_eval_block(
        agent, env, BLOCK_BASELINE, eval_eps, eval_steps, WORLD_DIM, device, zg,
    )
    # Red-team F4: snapshot the post-Stage-A agent + env for the paired
    # channel-(c) block. Taken BEFORE Stage B touches the original.
    n_detached, _memo = _detach_agent_buffers(agent)
    paired_agent = copy.deepcopy(agent, _memo)
    paired_env = copy.deepcopy(env)
    print(f"  [pair] seed={seed} post-Stage-A snapshot: detached={n_detached} "
          f"graph tensors, shared_modules={len(_memo)}", flush=True)
    return {
        "paired_agent": paired_agent,
        "paired_env": paired_env,
        "paired_n_detached": int(n_detached),
        "seed": seed,
        "env": env,
        "agent": agent,
        "zg": zg,
        "device": device,
        "eval_eps": eval_eps,
        "eval_steps": eval_steps,
        "world_forward_r2": world_forward_r2,
        "sleep_fires_warmup": train_out["sleep_fires"],
        "calibration": calib,
        "block_results": block_results,
    }


def run_seed_stage_b(state: Dict[str, Any]) -> None:
    """EVAL_HYPERVIGILANT then EVAL_REVERSION on the SAME trained agent.

    reset_all_rng(seed) is re-applied at entry so each seed's whole computation
    stays a pure function of (substrate, config, seed) despite Stage A of the
    other seeds having run in between -- which is what keeps the per-cell
    arm_fingerprint honest.
    """
    reset_all_rng(state["seed"])
    for block in (BLOCK_HYPERVIGILANT, BLOCK_REVERSION):
        state["block_results"][block] = _run_eval_block(
            state["agent"], state["env"], block, state["eval_eps"],
            state["eval_steps"], WORLD_DIM, state["device"], state["zg"],
        )
    # Red-team F4: the PAIRED block -- same post-Stage-A agent state (deep copy),
    # same env state (deep copy), same RNG state (reset_all_rng(seed), exactly
    # as EVAL_HYPERVIGILANT above started), channels (a)+(b) forced identically,
    # offline passes ON. Its AMBIGUOUS lift minus EVAL_HYPERVIGILANT's is the
    # behavioural contribution of suppressing (c), paired within seed.
    reset_all_rng(state["seed"])
    state["block_results"][BLOCK_HV_AB] = _run_eval_block(
        state["paired_agent"], state["paired_env"], BLOCK_HV_AB, state["eval_eps"],
        state["eval_steps"], WORLD_DIM, state["device"], state["zg"],
    )


def _arm_row(state: Dict[str, Any], blocks: Tuple[str, ...]) -> Dict[str, Any]:
    br = state["block_results"]
    row: Dict[str, Any] = {
        "seed": state["seed"],
        "world_forward_r2": state["world_forward_r2"],
        "sleep_fires_warmup": state["sleep_fires_warmup"],
        "commit_threshold_calibration": state["calibration"],
        "blocks_run": list(blocks),
        "block_rates": {b: br[b]["rates"] for b in blocks},
        "block_chance": {b: br[b]["chance"] for b in blocks},
        "block_lifts": {b: br[b]["lifts"] for b in blocks},
        "block_bins": {b: br[b]["bins"] for b in blocks},
        "block_precision_mean": {b: br[b]["precision_mean"] for b in blocks},
        "block_effective_horizon_mean": {b: br[b]["effective_horizon_mean"] for b in blocks},
        "block_draws_per_cycle": {b: br[b]["draws_per_cycle"] for b in blocks},
        "block_fatal_errors": {b: br[b]["fatal_errors"] for b in blocks},
        "block_precision_margin_norm_mean": {b: br[b]["precision_margin_norm_mean"] for b in blocks},
        "block_commit_temperature_eff_mean": {b: br[b]["commit_temperature_eff_mean"] for b in blocks},
        "block_precision_scaled_commit_engaged_count": {
            b: br[b]["precision_scaled_commit_engaged_count"] for b in blocks},
        "block_e3_selection_count": {b: br[b]["e3_selection_count"] for b in blocks},
        "block_sleep_fires": {b: br[b]["sleep_fires"] for b in blocks},
        "block_sleep_mech285_draws_max": {b: br[b]["sleep_mech285_draws_max"] for b in blocks},
        "block_sleep_mech285_draws_min": {b: br[b]["sleep_mech285_draws_min"] for b in blocks},
        "block_offline_passes_enabled": {b: br[b]["offline_passes_enabled"] for b in blocks},
        "block_sleep_boundary_calls": {b: br[b]["sleep_boundary_calls"] for b in blocks},
        "block_sleep_cycle_none_returns": {b: br[b]["sleep_cycle_none_returns"] for b in blocks},
        "block_sleep_sws_writes": {b: br[b]["sleep_sws_writes"] for b in blocks},
        "block_sleep_sws_writes_min": {b: br[b]["sleep_sws_writes_min"] for b in blocks},
        "block_sleep_sws_writes_max": {b: br[b]["sleep_sws_writes_max"] for b in blocks},
        "block_sleep_rem_rollouts_mean": {b: br[b]["sleep_rem_rollouts_mean"] for b in blocks},
        "block_sleep_mech285_draws": {b: br[b]["sleep_mech285_draws"] for b in blocks},
        "block_n_ticks": {b: br[b]["n_ticks"] for b in blocks},
        "block_per_episode_ambiguous_lift": {b: br[b]["per_episode_ambiguous_lift"] for b in blocks},
        "block_per_episode_ambiguous_n": {b: br[b]["per_episode_ambiguous_n"] for b in blocks},
        "block_unscorable_in_safe": {b: br[b]["unscorable_in_safe"] for b in blocks},
        "block_unscorable_total": {b: br[b]["unscorable_total"] for b in blocks},
        "block_precision_scaled_pick_nonargmin_frac": {
            b: br[b]["precision_scaled_pick_nonargmin_frac"] for b in blocks},
        "block_precision_scaled_pick_active_count": {
            b: br[b]["precision_scaled_pick_active_count"] for b in blocks},
        "paired_n_detached": state.get("paired_n_detached"),
    }
    # Red-team F1: C1/C2 read the availability-normalised LIFT (rate minus the
    # per-band chance rate), not the raw rate -- within-band availability is
    # position-dependent and the manipulation moves the agent's position
    # (smoke: AMBIGUOUS chance 0.329 BASE vs 0.378 HV, half the bar). Raw rates
    # stay recorded.
    row["base_ambiguous_rate"] = br[BLOCK_BASELINE]["rates"][HAZARD_BIN_AMBIGUOUS]
    row["base_ambiguous_lift"] = br[BLOCK_BASELINE]["lifts"][HAZARD_BIN_AMBIGUOUS]
    row["base_ambiguous_chance"] = br[BLOCK_BASELINE]["chance"][HAZARD_BIN_AMBIGUOUS]
    if BLOCK_HYPERVIGILANT in br:
        row["hv_ambiguous_rate"] = br[BLOCK_HYPERVIGILANT]["rates"][HAZARD_BIN_AMBIGUOUS]
        row["reversion_ambiguous_rate"] = br[BLOCK_REVERSION]["rates"][HAZARD_BIN_AMBIGUOUS]
        row["hv_ambiguous_lift"] = br[BLOCK_HYPERVIGILANT]["lifts"][HAZARD_BIN_AMBIGUOUS]
        row["reversion_ambiguous_lift"] = br[BLOCK_REVERSION]["lifts"][HAZARD_BIN_AMBIGUOUS]
        row["hv_ambiguous_chance"] = br[BLOCK_HYPERVIGILANT]["chance"][HAZARD_BIN_AMBIGUOUS]
        row["reversion_ambiguous_chance"] = br[BLOCK_REVERSION]["chance"][HAZARD_BIN_AMBIGUOUS]
        row["ambiguous_chance_shift_hv_minus_base"] = (
            row["hv_ambiguous_chance"] - row["base_ambiguous_chance"])
        elev = row["hv_ambiguous_lift"] - row["base_ambiguous_lift"]
        row["recovered_fraction"] = (
            (row["hv_ambiguous_lift"] - row["reversion_ambiguous_lift"]) / elev
            if abs(elev) > 1e-9 else 0.0
        )
        if BLOCK_HV_AB in br:
            row["hv_ab_ambiguous_lift"] = br[BLOCK_HV_AB]["lifts"][HAZARD_BIN_AMBIGUOUS]
            row["hv_ab_ambiguous_rate"] = br[BLOCK_HV_AB]["rates"][HAZARD_BIN_AMBIGUOUS]
            # F4: behavioural contribution of suppressing channel (c), paired.
            row["channel_c_contrast_ambiguous_lift"] = (
                row["hv_ambiguous_lift"] - row["hv_ab_ambiguous_lift"])
        # F3: BASELINE drift = second-half minus first-half mean AMBIGUOUS lift.
        _series = [x for x in br[BLOCK_BASELINE]["per_episode_ambiguous_lift"] if x == x]
        if len(_series) >= 2:
            _h = len(_series) // 2
            row["baseline_ambiguous_lift_drift"] = float(
                np.mean(_series[_h:]) - np.mean(_series[:_h]))
        else:
            row["baseline_ambiguous_lift_drift"] = float("nan")
    row["arm_fingerprint"] = compute_arm_fingerprint(
        config_slice=config_slice(), seed=state["seed"], script_path=Path(__file__),
        rng_fully_reset=True, config_slice_declared=True,
    )
    return row


def _fmt_preconditions(preconditions: List[Dict[str, Any]]) -> str:
    out = []
    for p in preconditions:
        try:
            m = f"{float(p.get('measured')):.6g}"
            t = f"{float(p.get('threshold')):.6g}"
        except (TypeError, ValueError):
            m, t = str(p.get("measured")), str(p.get("threshold"))
        out.append(f"- {p.get('name')}: measured={m} threshold={t} met={p.get('met')}")
    return "\n".join(out)


def _base_manifest(status: str, evidence_direction: str, label: str,
                   dry_run: bool, metrics: Dict[str, Any], criteria: List[Dict[str, Any]],
                   combination_rule: str, arm_rows: List[Dict[str, Any]],
                   non_degenerate: bool, degeneracy_reason: Any,
                   preconditions: List[Dict[str, Any]],
                   criteria_non_degenerate: Dict[str, bool],
                   summary_markdown: str, red_team_note: str) -> Dict[str, Any]:
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    return {
        "run_id": f"{EXPERIMENT_TYPE}_{ts}_v3",
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "supersedes": SUPERSEDES,
        "architecture_epoch": ARCHITECTURE_EPOCH,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "outcome": status,
        "timestamp_utc": ts,
        "evidence_direction": evidence_direction,
        "dry_run": dry_run,
        "metrics": metrics,
        "criteria": criteria,
        "combination_rule": combination_rule,
        "arm_results": arm_rows,
        "non_degenerate": non_degenerate,
        "degeneracy_reason": degeneracy_reason,
        "sleep_driver_pattern": (
            "manual-multi (force_cycle() every SLEEP_EVERY_N_EPISODES during warmup; "
            "force_sleep_cycle_at_eval_boundary() at every eval episode boundary; "
            "EVAL_HYPERVIGILANT disables the SWS/REM passes so its boundary calls "
            "are recorded None returns)"
        ),
        "interpretation": {
            "label": label,
            "preconditions": preconditions,
            "criteria_non_degenerate": criteria_non_degenerate,
        },
        "custom_information": {
            "scope_note": (
                "Scoped single-signature probe (hypervigilance only) per "
                "MECH-027's what_would_answer and the source proposal "
                "EXP-0761/EVB-1396 dispatch_mode=targeted_probe. The other four "
                "named pathological labels are out of scope."
            ),
            "gov_reuse_1_note": (
                "GOV-REUSE-1 re-checked at 981a authoring time (2026-09-04): the "
                "decisive readout is the AMBIGUOUS-band availability-normalised "
                "avoidance rate under a forced-precision/short-horizon/"
                "suppressed-replay regime. The only manifest that has ever "
                "tagged MECH-027 is V3-EXQ-981's, and it recorded that readout "
                "under the OLD single-argmax DV, the OLD band cutpoints, an "
                "EMPTY replay anchor pool and an UNCALIBRATED commit threshold "
                "-- four differences each of which changes the quantity itself. "
                "Not recoverable by reanalysis; the run is required."
            ),
            "inverted_band_ruling": (
                "governance-20260903 red-team amendment 3, discharged: 981's "
                "reversed positive control (-0.4307 on 3/3 seeds) is NOT an "
                "inverted hazard-band assignment. _hazard_bin's direction, the "
                "(x,y) axis convention shared by np.argwhere(env.grid) / "
                "hazard_field / step(), and the sign of the away-vector all "
                "verify correct at source. The cause is band-population "
                "imbalance plus a boundary-geometry confound in the old SAFE "
                "band (d > 11.33 = the wall-adjacent far corner, where the "
                "single-argmax avoidant action is a wall-blocked no-op): "
                "per-seed BASELINE bins were SAFE 123/132/157, AMBIGUOUS "
                "2738/2830/2747, HIGH 139/38/52. See the module docstring's "
                "'WHAT CHANGED FROM 981' fix (2)."
            ),
            "channel_forcing_note": (
                "All three channels are forced/reverted on the SAME trained "
                "agent per seed. The ARC-016 commit threshold is calibrated once "
                "after warmup and held identical in all three blocks."
            ),
            "red_team_disposition_note": red_team_note,
        },
        "ethics_preflight": {
            "involves_negative_valence": True,
            "involves_suffering_like_state": False,
            "involves_self_model": False,
            "involves_inescapability_or_helplessness": False,
            "involves_offline_replay_over_harm": True,
            "involves_social_mind_or_language": False,
            "involves_human_data_or_clinical_context": False,
            "decision": "allow",
        },
        "summary_markdown": summary_markdown,
    }


RED_TEAM_REPAIR_VERDICT = (
    "CONTESTED (fable, 2026-09-09; 6 findings, none BLOCKING; all six FIXED, none "
    "dismissed). F1 C1/C2 read raw rate while within-band availability shifts "
    "with the manipulation (smoke chance 0.329 BASE vs 0.378 HV) -> C1/C2 now "
    "read the AMBIGUOUS-band LIFT, chance shift recorded. F2 HV can empty the C1 "
    "band / hide in unscorable corners, ticks are not decisions -> Gate B floors "
    "the pooled AMBIGUOUS n in HV/REVERSION/HV_AB (150 scaled) and the per-block "
    "E3 selection count (30 scaled); unscorable_in_safe recorded. F3 no time "
    "control, an upward drift lands in the mixed cell -> per-episode AMBIGUOUS "
    "lift series recorded per block; |BASELINE half-drift| >= 0.05 routes the "
    "C1-and-not-C2 cell to non_contributory. F4 channel (c) contrast inert by "
    "construction (waking ContextMemory writes off, HV inherits BASELINE's "
    "memory) -> a PAIRED block EVAL_HV_AB_PAIRED on a deep copy of the "
    "post-Stage-A agent+env from the same RNG state, (a)+(b) forced, passes ON; "
    "PASS with |lift(HV)-lift(HV_AB)| < 0.02 routes to 'mixed' (channels a+b "
    "only, c inert). F5 channel (a)'s only consumer spans ~12% temperature -> "
    "BASELINE non-argmin fraction of graded picks recorded; a weakens null under "
    "a fraction < 0.10 routes to non_contributory (range-limited); engagement "
    "gate now per block. F6 positive control was a mean of per-seed lifts while "
    "the 150 floor was derived for pooled counts -> computed on pooled counts, "
    "mean-of-seeds recorded. CLEAR-nits: effective-horizon sampled only on "
    "proposal ticks; docstring geometry corrected (walled 8x8 interior)."
)

RED_TEAM_NOTE = (
    "981a red-team (see queue entry note for verdict + model). Carried "
    "forward from 981's own red-team (fable, 2026-09-02): F1 fixed "
    "(no_fatal_action_selection_errors); F3 fixed (engagement FRACTION gate); "
    "F6 fixed (C2 on pooled rates); F8 fixed. F2 verified and dismissed at "
    "source (agent.e2_harm_s is None because use_e2_harm_s_forward is never "
    "set, so SleepLoopManager's offline_gradient_pass writeback cannot reach "
    "the DV). F4 (fixed BASELINE->HV->REVERSION block order is an order "
    "confound) ACKNOWLEDGED, NOT MITIGATED -- it is the direct consequence of "
    "MECH-027 Build 2's mandate that replay suppression be an exercised lever; "
    "C2's reversion requirement is the guard, and the C1-and-not-C2 cell "
    "already reads as partial. F5 (REVERSION restores the scale but not the "
    "compounded _running_variance) ACKNOWLEDGED, NOT MITIGATED BY DESIGN -- "
    "reversion is deliberately natural recovery, matching the claim's own "
    "'returned to its normal range'. F7 (SAFE bin structurally empty for a "
    "central hazard placement) is now FIXED, not acknowledged: it was the root "
    "of the reversed positive control, and 981a re-derives the cutpoints "
    "(HAZARD_SAFE_MAX 0.15 -> 0.25) so every band is populated for every "
    "hazard placement, re-derives the coverage floors (per-seed 30 non-vacuity, "
    "pooled 150 power, from the positive-control bar; the draft's flat 150 per "
    "seed was unmet by 981's own HIGH counts -- docstring B3), and replaces "
    "the single-argmax DV with an availability-normalised, wall-aware set "
    "membership test. 2026-09-09 REPAIR (campaign W6-S5b item 1): B1/B2 -- "
    "channel (c) re-specified from the mech285 draw count (sign-inverted "
    "consumer, intermittently-populated pool) onto the SD-017 offline passes "
    "themselves (sws_enabled/rem_enabled per block), with the replay gates "
    "read off sws_n_writes; B3 -- coverage floors re-derived; dry-run Stage B "
    "positive control added. Repair red-team: " + RED_TEAM_REPAIR_VERDICT
)


def run_experiment(seeds: List[int], dry_run: bool) -> Dict[str, Any]:
    t0 = time.perf_counter()
    # (config_slice() is read per-cell in _arm_row)

    # ================= STAGE A: warmup + EVAL_BASELINE only =================
    states: Dict[int, Dict[str, Any]] = {}
    for seed in seeds:
        states[seed] = run_seed_stage_a(seed, dry_run)

    def _bl(seed: str, key: str, bin_: str) -> float:
        return states[seed]["block_results"][BLOCK_BASELINE][key][bin_]

    base_rates = [_bl(s, "rates", HAZARD_BIN_AMBIGUOUS) for s in seeds]
    base_rate_max = float(np.max(base_rates))
    base_lifts = [_bl(s, "lifts", HAZARD_BIN_AMBIGUOUS) for s in seeds]
    base_chances = [_bl(s, "chance", HAZARD_BIN_AMBIGUOUS) for s in seeds]

    # positive control on LIFT (cross-band -> availability must be normalised),
    # computed on POOLED COUNTS across seeds (red-team F6: the 150 pooled floor
    # was derived for a pooled-count SE; a mean of per-seed lifts has the SE of
    # its smallest cell). Per-seed lifts are recorded as diagnostics.
    def _pooled_lift(bin_: str) -> float:
        n = sum(states[s]["block_results"][BLOCK_BASELINE]["bins"][bin_]["n"] for s in seeds)
        av = sum(states[s]["block_results"][BLOCK_BASELINE]["bins"][bin_]["avoidant"] for s in seeds)
        af = sum(states[s]["block_results"][BLOCK_BASELINE]["bins"][bin_]["avail_frac_sum"] for s in seeds)
        return (av / n - af / n) if n > 0 else 0.0
    high_lifts = [_bl(s, "lifts", HAZARD_BIN_HIGH) for s in seeds]
    safe_lifts = [_bl(s, "lifts", HAZARD_BIN_SAFE) for s in seeds]
    positive_control_margin = float(_pooled_lift(HAZARD_BIN_HIGH) - _pooled_lift(HAZARD_BIN_SAFE))
    positive_control_margin_mean_of_seeds = float(np.mean(high_lifts) - np.mean(safe_lifts))
    # the same reading on RAW rates, recorded so 981's number stays comparable
    positive_control_margin_raw = float(
        np.mean([_bl(s, "rates", HAZARD_BIN_HIGH) for s in seeds])
        - np.mean([_bl(s, "rates", HAZARD_BIN_SAFE) for s in seeds])
    )

    min_bin_coverage_a = min(
        states[s]["block_results"][BLOCK_BASELINE]["bins"][hb]["n"]
        for s in seeds
        for hb in (HAZARD_BIN_SAFE, HAZARD_BIN_AMBIGUOUS, HAZARD_BIN_HIGH)
    )
    # Pooled (across-seed) coverage of the two POSITIVE-CONTROL bands.
    pooled_pc_coverage = {
        hb: sum(states[s]["block_results"][BLOCK_BASELINE]["bins"][hb]["n"] for s in seeds)
        for hb in (HAZARD_BIN_SAFE, HAZARD_BIN_HIGH)
    }
    min_pooled_pc_coverage = min(pooled_pc_coverage.values())
    # Coverage floors scale with the eval-tick budget so --dry-run exercises
    # the gate arithmetic (docstring B3); at the real budget the scale is 1.0.
    eval_tick_scale = (
        states[seeds[0]]["eval_eps"] * states[seeds[0]]["eval_steps"]
        / float(EVAL_EPISODES_PER_BLOCK * EVAL_STEPS_PER_EPISODE)
    )
    floor_per_seed = _scaled_floor(MIN_BIN_COVERAGE_STEPS_PER_SEED, eval_tick_scale)
    floor_pooled = _scaled_floor(MIN_BIN_COVERAGE_STEPS_POOLED, eval_tick_scale)
    base_margins = [
        states[s]["block_results"][BLOCK_BASELINE]["precision_margin_norm_mean"] for s in seeds
    ]
    base_replay_min = float(min(
        states[s]["block_results"][BLOCK_BASELINE]["sleep_mech285_draws_min"] for s in seeds
    ))
    base_sws_writes_min = float(min(
        states[s]["block_results"][BLOCK_BASELINE]["sleep_sws_writes_min"] for s in seeds
    ))
    fatal_a = sum(states[s]["block_results"][BLOCK_BASELINE]["fatal_errors"] for s in seeds)

    gate_a_checks = [
        {
            "name": "positive_control_hazard_sensitivity",
            "measured": positive_control_margin,
            "threshold": POSITIVE_CONTROL_MARGIN,
            "direction": "lower",
            "control": (
                "EVAL_BASELINE avoidance LIFT (rate minus availability-matched "
                "chance) in the HIGH hazard band minus the SAFE band, on POOLED "
                "COUNTS across seeds (the statistic the 150 pooled floor was "
                "derived for -- red-team F6; the mean-of-seeds reading is "
                "recorded as positive_control_margin_mean_of_seeds). LIFT, not raw rate: the number of "
                "distance-increasing actions available differs systematically "
                "between bands by grid geometry, and comparing raw rates across "
                "bands is what produced V3-EXQ-981's -0.4307 reversal. The DV/env "
                "pairing must be hazard-sensitive before ANY hypervigilance block "
                "is run -- this gate is evaluated at the end of Stage A and "
                "EVAL_HYPERVIGILANT / EVAL_REVERSION do not run if it is unmet."
            ),
        },
        {
            "name": "hazard_bin_sample_coverage",
            "measured": float(min_bin_coverage_a),
            "threshold": float(floor_per_seed),
            "direction": "lower",
            "control": (
                "the smallest per-seed EVAL_BASELINE (hazard-bin) count of "
                "SCORABLE steps against the per-seed NON-VACUITY floor "
                f"(MIN_BIN_COVERAGE_STEPS_PER_SEED = {MIN_BIN_COVERAGE_STEPS_PER_SEED}, "
                f"scaled x{eval_tick_scale:.4g} by the eval-tick budget). 981's "
                "floor of 5 admitted a 38-step HIGH bin; the 981a draft's 150 "
                "per seed was unmet on all three seeds by 981's own measured "
                "HIGH counts 139/38/52 (docstring B3)."
            ),
        },
        {
            "name": "positive_control_band_pooled_coverage",
            "measured": float(min_pooled_pc_coverage),
            "threshold": float(floor_pooled),
            "direction": "lower",
            "control": (
                "the smaller of the HIGH and SAFE band scorable-step counts "
                "POOLED across seeds in EVAL_BASELINE, against the POWER floor "
                f"(MIN_BIN_COVERAGE_STEPS_POOLED = {MIN_BIN_COVERAGE_STEPS_POOLED}, "
                f"scaled x{eval_tick_scale:.4g}). Derived from the positive-control "
                "bar: a binary-rate SE is <= 0.5/sqrt(n) and a +0.05 lift margin "
                "needs pooled SE ~0.04, i.e. n >= 150. This is the statistic the "
                "positive control is actually read on (pooled lift)."
            ),
        },
        {
            "name": "replay_channel_baseline_reachable",
            "measured": base_sws_writes_min,
            "threshold": float(SWS_WRITES_BASELINE_FLOOR),
            "direction": "lower",
            "control": (
                "MEASURED sws_n_writes (from force_sleep_cycle_at_eval_"
                "boundary()'s own return) across every EVAL_BASELINE firing on "
                "every seed -- the SD-017 SWS schema pass wrote E1.ContextMemory "
                "on every firing, so the offline replay channel that "
                "EVAL_HYPERVIGILANT will suppress is engaged for real. "
                "Deterministically = sws_consolidation_steps once the "
                "world-experience buffer is populated (2026-09-09 probe: 5/5). "
                "Replaces the 981a-draft mech285_n_draws min>=1 gate, which the "
                "per-episode anchor-pool reset made unsatisfiable (docstring B1); "
                "mech285_n_draws is still recorded as sleep_mech285_draws_base_min."
            ),
        },
        {
            "name": "no_fatal_action_selection_errors_stage_a",
            "measured": float(fatal_a),
            "threshold": 0.5,
            "direction": "upper",
            "control": (
                "total _agent_tick exceptions in Stage A -- each substitutes a "
                "RANDOM action into the DV stream, injecting near-chance noise "
                "into the baseline the whole run is calibrated against."
            ),
        },
        dv_headroom_check(
            "c1_elevation_headroom",
            dv_name="ambiguous_band_avoidant_rate",
            criterion_threshold=FALSE_ALARM_ELEVATION_ABS,
            # MEAN-MATCHED ceiling (2026-09-09; validate_experiments
            # dv_headroom-statistic-mismatch lint, the V3-EXQ-972a shape): C1
            # compares POOLED MEANS of the AMBIGUOUS-band LIFT (red-team F1).
            # The largest lift a block can reach is 1 - chance, so the room
            # above the mean baseline lift is (1 - chance) - lift = 1 - rate,
            # i.e. 1 - mean(base_rates). The max-based figure is still recorded
            # as c1_elevation_headroom_available_max_based in stage_a_metrics.
            achievable=float(1.0 - np.mean(base_rates)),
            margin=HEADROOM_MARGIN_C1,
            control=(
                "C1 registers an ABSOLUTE elevation of "
                f"{FALSE_ALARM_ELEVATION_ABS} of the AMBIGUOUS-band avoidance "
                "LIFT (rate minus availability-matched chance; red-team F1), "
                "read on POOLED MEANS. The room above the mean baseline lift is "
                "(1 - chance) - lift = 1 - mean(EVAL_BASELINE rate), the same "
                "statistic C1 aggregates with, and must be at least "
                f"{HEADROOM_MARGIN_C1}x the bar. V3-EXQ-981 set 2 x 0.5771 = "
                "1.1542 on this same bounded DV -- outside its range at any "
                "outcome. Falsified from the run's own data at emit time by "
                "dv_headroom_observation_check (headroom_ceiling_exceeded_by_"
                "observation)."
            ),
        ),
        dv_headroom_check(
            "precision_margin_headroom",
            dv_name="precision_margin_norm",
            criterion_threshold=PRECISION_MARGIN_HV_ELEVATION_FLOOR,
            # MEAN-MATCHED for the same reason: the Gate B check it certifies
            # (precision_margin_norm_elevated_under_hv) reads pooled means.
            achievable=float(1.0 - np.mean(base_margins)),
            margin=1.0,
            control=(
                "Verifies the post-warmup commitment_threshold calibration "
                "actually opened range in the graded consumer's input. "
                "V3-EXQ-981's uncalibrated baseline margin was 0.99980, leaving "
                "0.000195 of ceiling headroom against a 0.01 floor -- a 51x "
                "shortfall no manipulation could close, and the reason "
                "commit_temperature_eff was pinned within 2e-4 of "
                "base_temperature."
            ),
        ),
    ]

    ready_a = True
    preconditions: List[Dict[str, Any]] = []
    try:
        preconditions = p0_readiness_gate(gate_a_checks)
    except P0NotReady as e:
        preconditions = e.preconditions
        ready_a = False

    stage_a_metrics = {
        "positive_control_margin": positive_control_margin,
        "positive_control_margin_mean_of_seeds": positive_control_margin_mean_of_seeds,
        "positive_control_margin_raw_rates": positive_control_margin_raw,
        "mean_base_ambiguous_lift": float(np.mean(base_lifts)),
        "mean_base_ambiguous_chance": float(np.mean(base_chances)),
        "positive_control_high_lift_mean": float(np.mean(high_lifts)),
        "positive_control_safe_lift_mean": float(np.mean(safe_lifts)),
        "min_bin_coverage_steps": float(min_bin_coverage_a),
        "positive_control_band_pooled_coverage_min": float(min_pooled_pc_coverage),
        "positive_control_band_pooled_coverage_high": float(pooled_pc_coverage[HAZARD_BIN_HIGH]),
        "positive_control_band_pooled_coverage_safe": float(pooled_pc_coverage[HAZARD_BIN_SAFE]),
        "coverage_floor_per_seed_applied": float(floor_per_seed),
        "coverage_floor_pooled_applied": float(floor_pooled),
        "eval_tick_scale": float(eval_tick_scale),
        "sleep_sws_writes_base_min": base_sws_writes_min,
        "mean_base_ambiguous_rate": float(np.mean(base_rates)),
        "per_seed_base_ambiguous_lift": [float(x) for x in base_lifts],
        "per_seed_positive_control_high_lift": [float(x) for x in high_lifts],
        "per_seed_positive_control_safe_lift": [float(x) for x in safe_lifts],
        "base_ambiguous_rate_max_across_seeds": base_rate_max,
        "c1_elevation_headroom_available": float(1.0 - np.mean(base_rates)),
        "c1_elevation_headroom_available_max_based": float(1.0 - base_rate_max),
        "precision_margin_norm_base_mean": float(np.mean(base_margins)),
        "precision_margin_headroom_available": float(1.0 - np.mean(base_margins)),
        "precision_margin_headroom_available_max_based": float(1.0 - max(base_margins)),
        "sleep_mech285_draws_base_min": base_replay_min,
        "total_fatal_errors_stage_a": float(fatal_a),
    }

    gate_a_forced_for_smoke = False
    if not ready_a and dry_run:
        # --dry-run ONLY: Stage B runs anyway as a CODE-PATH positive control
        # (red-team F3 on the 981a draft: Stage B, Gate B, C1/C2 and every
        # verdict branch had never executed once, because a 45-tick block can
        # never populate every band). The manifest is dry_run-relocated by
        # emit_outcome, `ready` below is forced False, and the label can only
        # be substrate_not_ready_requeue -- this never launders a real run.
        unmet_a = [p["name"] for p in preconditions if not p.get("met")]
        print(f"[smoke] Gate A unmet ({unmet_a}) -- dry-run continues into "
              "Stage B as a code-path positive control "
              "(gate_a_forced_for_smoke=True)", flush=True)
        gate_a_forced_for_smoke = True
    if not ready_a and not dry_run:
        # ---- Stage A abort: HV / REVERSION never run (autopsy fix 3) ----
        arm_rows = [_arm_row(states[s], (BLOCK_BASELINE,)) for s in seeds]
        for s in seeds:
            print(f"verdict: FAIL  seed={s}  stage=A_only  "
                  f"base_ambig_rate={_bl(s, 'rates', HAZARD_BIN_AMBIGUOUS):.4f}",
                  flush=True)
        unmet = [p["name"] for p in preconditions if not p.get("met")]
        label = "substrate_not_ready_requeue"
        print(f"\nV3-EXQ-981a pooled verdict: FAIL  label={label}  "
              f"unmet={unmet}  (Stage B not run)", flush=True)
        combination_rule = (
            "Gate A (the DV/env pairing gate) is evaluated after EVAL_BASELINE "
            "and BEFORE EVAL_HYPERVIGILANT / EVAL_REVERSION are run at all. Any "
            "unmet Gate A precondition routes to substrate_not_ready_requeue and "
            "the hypervigilance blocks are skipped."
        )
        summary = (
            f"# V3-EXQ-981a -- MECH-027 (Stage A abort)\n\n"
            f"**Overall Status:** FAIL (label `{label}`)\n"
            f"**Unmet Gate A preconditions:** {unmet}\n"
            f"**Seeds:** {seeds}\n\n"
            f"## Gate A preconditions\n\n{_fmt_preconditions(preconditions)}\n\n"
            f"{combination_rule}\n"
        )
        metrics = dict(stage_a_metrics)
        metrics["criteria_met"] = 0.0
        for s in seeds:
            metrics[f"seed{s}_base_ambiguous_rate"] = _bl(s, "rates", HAZARD_BIN_AMBIGUOUS)
        manifest = _base_manifest(
            "FAIL", "non_contributory", label, dry_run, metrics,
            [
                {"name": "C1_false_alarm_elevation", "load_bearing": True,
                 "passed": False, "measured": None,
                 "threshold": FALSE_ALARM_ELEVATION_ABS,
                 "note": "not evaluated -- Stage B did not run"},
                {"name": "C2_reversion_recovery", "load_bearing": True,
                 "passed": False, "measured": None,
                 "threshold": REVERSION_RECOVERY_FLOOR,
                 "note": "not evaluated -- Stage B did not run"},
            ],
            combination_rule, arm_rows, False,
            "gate_a_unmet: " + ", ".join(unmet), preconditions,
            {"C1_false_alarm_elevation": False, "C2_reversion_recovery": False},
            summary, RED_TEAM_NOTE,
        )
        full_config = {
            "seeds": seeds, **config_slice(), "thresholds": _thresholds(),
        }
        out_path = write_flat_manifest(
            manifest, dry_run=dry_run, config=full_config, seeds=seeds,
            script_path=Path(__file__), started_at=t0,
            agent=[states[s]["agent"] for s in seeds],
        )
        return {"outcome": "FAIL", "manifest": manifest, "out_path": out_path}

    # ============ STAGE B: EVAL_HYPERVIGILANT + EVAL_REVERSION ============
    for seed in seeds:
        run_seed_stage_b(states[seed])

    arm_rows = [_arm_row(states[s], BLOCKS_ALL) for s in seeds]

    for r in arm_rows:
        print(f"verdict: {'PASS' if (r['hv_ambiguous_lift'] - r['base_ambiguous_lift']) >= FALSE_ALARM_ELEVATION_ABS else 'FAIL'}"
              f"  seed={r['seed']}  base_ambig_lift={r['base_ambiguous_lift']:.4f}"
              f"  hv_ambig_lift={r['hv_ambiguous_lift']:.4f}"
              f"  rev_ambig_lift={r['reversion_ambiguous_lift']:.4f}"
              f"  hv_ab_lift={r['hv_ab_ambiguous_lift']:.4f}"
              f"  recovered_fraction={r['recovered_fraction']:.4f}", flush=True)

    # C1/C2 on the AMBIGUOUS-band LIFT (red-team F1). Raw rates kept as metrics.
    hv_rates = [r["hv_ambiguous_rate"] for r in arm_rows]
    rev_rates = [r["reversion_ambiguous_rate"] for r in arm_rows]
    hv_lifts = [r["hv_ambiguous_lift"] for r in arm_rows]
    rev_lifts = [r["reversion_ambiguous_lift"] for r in arm_rows]
    hv_ab_lifts = [r["hv_ab_ambiguous_lift"] for r in arm_rows]
    recovered_fracs = [r["recovered_fraction"] for r in arm_rows]
    chance_shifts = [r["ambiguous_chance_shift_hv_minus_base"] for r in arm_rows]
    baseline_drifts = [r["baseline_ambiguous_lift_drift"] for r in arm_rows
                       if r["baseline_ambiguous_lift_drift"] == r["baseline_ambiguous_lift_drift"]]
    channel_c_contrasts = [r["channel_c_contrast_ambiguous_lift"] for r in arm_rows]

    mean_base_rate = float(np.mean(base_rates))
    mean_hv_rate = float(np.mean(hv_rates))
    mean_rev_rate = float(np.mean(rev_rates))
    mean_base_lift = float(np.mean(base_lifts))
    mean_hv_lift = float(np.mean(hv_lifts))
    mean_rev_lift = float(np.mean(rev_lifts))
    mean_hv_ab_lift = float(np.mean(hv_ab_lifts))
    mean_recovered = float(np.mean(recovered_fracs))
    mean_chance_shift = float(np.mean(chance_shifts))
    pooled_baseline_drift = float(np.mean(baseline_drifts)) if baseline_drifts else float("nan")
    pooled_channel_c_contrast = float(np.mean(channel_c_contrasts))

    pooled_elevation = mean_hv_lift - mean_base_lift
    pooled_elevation_raw_rate = mean_hv_rate - mean_base_rate
    pooled_recovered_fraction = (
        (mean_hv_lift - mean_rev_lift) / pooled_elevation
        if pooled_elevation > 1e-9 else 0.0
    )
    # F3: a monotone drift across the block order is plausible when BASELINE's
    # own second half already differs from its first half by half the bar.
    order_confound_plausible = bool(
        pooled_baseline_drift == pooled_baseline_drift
        and abs(pooled_baseline_drift) >= ORDER_DRIFT_FLOOR)
    # F4: channel (c) contributed measurably iff the paired contrast clears the floor.
    channel_c_inert = bool(abs(pooled_channel_c_contrast) < CHANNEL_C_CONTRAST_FLOOR)

    precision_base_mean = float(np.mean([r["block_precision_mean"][BLOCK_BASELINE] for r in arm_rows]))
    precision_hv_mean = float(np.mean([r["block_precision_mean"][BLOCK_HYPERVIGILANT] for r in arm_rows]))
    precision_ratio = (precision_hv_mean / precision_base_mean) if precision_base_mean > 1e-12 else 0.0

    eh_base_vals = [r["block_effective_horizon_mean"][BLOCK_BASELINE] for r in arm_rows
                    if r["block_effective_horizon_mean"][BLOCK_BASELINE] is not None]
    eh_hv_vals = [r["block_effective_horizon_mean"][BLOCK_HYPERVIGILANT] for r in arm_rows
                  if r["block_effective_horizon_mean"][BLOCK_HYPERVIGILANT] is not None]
    eh_base_mean = float(np.mean(eh_base_vals)) if eh_base_vals else None
    eh_hv_mean = float(np.mean(eh_hv_vals)) if eh_hv_vals else None
    horizon_ratio = ((eh_hv_mean / eh_base_mean)
                     if (eh_base_mean and eh_hv_mean and eh_base_mean > 1e-9) else None)

    draws_hv_vals = [r["block_draws_per_cycle"][BLOCK_HYPERVIGILANT] for r in arm_rows]
    draws_base_vals = [r["block_draws_per_cycle"][BLOCK_BASELINE] for r in arm_rows]
    draws_rev_vals = [r["block_draws_per_cycle"][BLOCK_REVERSION] for r in arm_rows]

    precision_margin_base_mean = float(np.mean(
        [r["block_precision_margin_norm_mean"][BLOCK_BASELINE] for r in arm_rows]))
    precision_margin_hv_mean = float(np.mean(
        [r["block_precision_margin_norm_mean"][BLOCK_HYPERVIGILANT] for r in arm_rows]))
    precision_margin_hv_elevation = precision_margin_hv_mean - precision_margin_base_mean

    commit_temp_base_mean = float(np.mean(
        [r["block_commit_temperature_eff_mean"][BLOCK_BASELINE] for r in arm_rows]))
    commit_temp_hv_mean = float(np.mean(
        [r["block_commit_temperature_eff_mean"][BLOCK_HYPERVIGILANT] for r in arm_rows]))
    commit_temp_hv_reduction = commit_temp_base_mean - commit_temp_hv_mean

    precision_scaled_commit_engaged_total = sum(
        r["block_precision_scaled_commit_engaged_count"][b]
        for r in arm_rows for b in (BLOCK_BASELINE, BLOCK_HYPERVIGILANT))
    e3_selection_total = sum(
        r["block_e3_selection_count"][b]
        for r in arm_rows for b in (BLOCK_BASELINE, BLOCK_HYPERVIGILANT))
    precision_scaled_commit_engaged_fraction = (
        precision_scaled_commit_engaged_total / e3_selection_total
        if e3_selection_total > 0 else 0.0)
    # F5: engagement per block (min over BASELINE and HV), so a BASELINE that
    # never engages the graded branch cannot pass on HV's engagement alone.
    engaged_frac_per_block = []
    for b in (BLOCK_BASELINE, BLOCK_HYPERVIGILANT):
        _eng = sum(r["block_precision_scaled_commit_engaged_count"][b] for r in arm_rows)
        _sel = sum(r["block_e3_selection_count"][b] for r in arm_rows)
        engaged_frac_per_block.append(_eng / _sel if _sel > 0 else 0.0)
    min_engaged_frac_per_block = float(min(engaged_frac_per_block))
    # F5: channel (a) selection range -- BASELINE fraction of graded picks that
    # were NOT the envelope argmin (pooled counts across seeds).
    _na = sum(r["block_precision_scaled_pick_active_count"][BLOCK_BASELINE] for r in arm_rows)
    _nn = sum(int(round(r["block_precision_scaled_pick_nonargmin_frac"][BLOCK_BASELINE]
                        * r["block_precision_scaled_pick_active_count"][BLOCK_BASELINE]))
              for r in arm_rows
              if r["block_precision_scaled_pick_nonargmin_frac"][BLOCK_BASELINE] >= 0.0)
    channel_a_selection_range = (_nn / float(_na)) if _na > 0 else 0.0
    # F2: pooled scorable AMBIGUOUS n in the two blocks C1/C2 read against BASELINE.
    pooled_ambiguous_n = {
        b: sum(r["block_bins"][b][HAZARD_BIN_AMBIGUOUS]["n"] for r in arm_rows)
        for b in BLOCKS_ALL}
    min_e3_selections = min(r["block_e3_selection_count"][b] for r in arm_rows for b in BLOCKS_ALL)

    # Sleep cycles must FIRE in the two pass-enabled blocks; in HV every
    # boundary call must be the suppressed None return (2026-09-09 repair).
    sleep_fires_vals = [r["block_sleep_fires"][b] for r in arm_rows
                        for b in (BLOCK_BASELINE, BLOCK_REVERSION, BLOCK_HV_AB)]
    min_sleep_fires = min(sleep_fires_vals) if sleep_fires_vals else 0
    sleep_draws_hv_max_vals = [r["block_sleep_mech285_draws_max"][BLOCK_HYPERVIGILANT] for r in arm_rows]
    sleep_draws_rev_min_vals = [r["block_sleep_mech285_draws_min"][BLOCK_REVERSION] for r in arm_rows]
    sws_writes_hv_max_vals = [r["block_sleep_sws_writes_max"][BLOCK_HYPERVIGILANT] for r in arm_rows]
    sws_writes_rev_min_vals = [r["block_sleep_sws_writes_min"][BLOCK_REVERSION] for r in arm_rows]
    hv_suppressed_fracs = [
        (r["block_sleep_cycle_none_returns"][BLOCK_HYPERVIGILANT]
         / float(max(1, r["block_sleep_boundary_calls"][BLOCK_HYPERVIGILANT])))
        for r in arm_rows]
    min_hv_suppressed_frac = min(hv_suppressed_fracs) if hv_suppressed_fracs else 0.0

    min_bin_coverage = min(
        r["block_bins"][b][hb]["n"] for r in arm_rows for b in BLOCKS_ALL
        for hb in (HAZARD_BIN_SAFE, HAZARD_BIN_AMBIGUOUS, HAZARD_BIN_HIGH))
    total_unscorable = sum(
        r["block_bins"][b][hb]["n_unscorable"] for r in arm_rows for b in BLOCKS_ALL
        for hb in (HAZARD_BIN_SAFE, HAZARD_BIN_AMBIGUOUS, HAZARD_BIN_HIGH))
    total_fatal = sum(r["block_fatal_errors"][b] for r in arm_rows for b in BLOCKS_ALL)

    gate_b_checks = [
        {
            "name": "precision_channel_non_degenerate",
            "measured": precision_ratio, "threshold": PRECISION_RATIO_FLOOR,
            "direction": "lower",
            "control": ("pooled EVAL_HYPERVIGILANT precision mean / EVAL_BASELINE "
                        f"precision mean, must clear {PRECISION_RATIO_FLOOR}x."),
        },
        {
            "name": "horizon_channel_non_degenerate",
            "measured": (horizon_ratio if horizon_ratio is not None else 1.0),
            "threshold": HORIZON_RATIO_CEIL, "direction": "upper",
            "control": ("structural effective_horizon under MODE_HV / under "
                        "MODE_BASE -- confirms the SD-MECH267-HORIZON-DEPTH "
                        "scoring window actually shortened."),
        },
        {
            "name": "replay_channel_non_degenerate",
            "measured": float(max(sws_writes_hv_max_vals) if sws_writes_hv_max_vals else 1.0),
            "threshold": float(SWS_WRITES_HV_CEIL), "direction": "upper",
            "control": ("MEASURED sws_n_writes across every EVAL_HYPERVIGILANT "
                        "boundary, on every seed (a None return counts as 0) -- "
                        "no SWS schema write reached E1.ContextMemory while the "
                        "passes were disabled. Informative only because Gate A "
                        "already proved BASELINE writes >= 1 on every firing; the "
                        "next check certifies the suppression was EXERCISED."),
        },
        {
            "name": "hv_offline_passes_suppressed_every_boundary",
            "measured": float(min_hv_suppressed_frac),
            "threshold": float(HV_OFFLINE_PASSES_SUPPRESSED_FRACTION_FLOOR),
            "direction": "lower",
            "control": ("smallest per-seed fraction of EVAL_HYPERVIGILANT boundary "
                        "calls that returned None (SleepLoopManager._run_cycle "
                        "honouring require_sleep_passes_enabled with sws_enabled "
                        "= rem_enabled = False). Must be 1.0: the suppression was "
                        "applied at every boundary a cycle would otherwise have "
                        "fired, i.e. the lever was exercised, not merely set."),
        },
        {
            "name": "replay_channel_reversion_reachable",
            "measured": float(min(sws_writes_rev_min_vals) if sws_writes_rev_min_vals else 0.0),
            "threshold": float(SWS_WRITES_BASELINE_FLOOR), "direction": "lower",
            "control": ("MEASURED sws_n_writes across every EVAL_REVERSION "
                        "firing -- the offline replay channel came back when the "
                        "suppression was lifted (passes re-enabled on the live "
                        "agent)."),
        },
        {
            "name": "sleep_cycle_fires_during_eval",
            "measured": float(min_sleep_fires), "threshold": float(SLEEP_CYCLE_FIRE_FLOOR),
            "direction": "lower",
            "control": ("smallest per-seed count of non-None "
                        "force_sleep_cycle_at_eval_boundary() returns over the "
                        "two pass-ENABLED blocks (EVAL_BASELINE, EVAL_REVERSION); "
                        "EVAL_HYPERVIGILANT is gated the other way above."),
        },
        {
            "name": "precision_scaled_commit_temperature_engaged",
            "measured": min_engaged_frac_per_block,
            "threshold": PRECISION_SCALED_COMMIT_ENGAGED_FRACTION_FLOOR,
            "direction": "lower",
            "control": ("SMALLEST per-block fraction (EVAL_BASELINE, "
                        "EVAL_HYPERVIGILANT; pooled across seeds) of real E3 "
                        "selections where precision_scaled_commit_active was "
                        "True (red-team F5: pooling the two blocks let a "
                        "never-engaging BASELINE pass on HV's engagement). This also "
                        "guards the 981a commit-threshold calibration: lowering "
                        "commitment_threshold to 4x running_variance could in "
                        "principle drop the committed branch out of engagement."),
        },
        {
            "name": "precision_margin_norm_elevated_under_hv",
            "measured": precision_margin_hv_elevation,
            "threshold": PRECISION_MARGIN_HV_ELEVATION_FLOOR, "direction": "lower",
            "control": ("pooled precision_margin_norm under EVAL_HYPERVIGILANT "
                        "minus under EVAL_BASELINE. Gate A already certified the "
                        "HEADROOM exists; this asserts the manipulation used it. "
                        "(981a drops commit_temperature_reduced_under_hv as a "
                        "separate check: T_eff = 1 + alpha*(1 - margin) with "
                        "alpha = 1, so it is the same quantity renamed.)"),
        },
        {
            "name": "hazard_bin_sample_coverage_all_blocks",
            "measured": float(min_bin_coverage), "threshold": float(floor_per_seed),
            "direction": "lower",
            "control": ("smallest (block, hazard-bin) SCORABLE step count across "
                        "every seed/block/bin, now including the two "
                        "hypervigilance blocks and the paired HV_AB block, against "
                        "the per-seed non-vacuity "
                        f"floor ({MIN_BIN_COVERAGE_STEPS_PER_SEED} x {eval_tick_scale:.4g})."),
        },
        {
            "name": "c1_band_pooled_coverage_hv_and_reversion",
            "measured": float(min(pooled_ambiguous_n[BLOCK_HYPERVIGILANT],
                                  pooled_ambiguous_n[BLOCK_REVERSION],
                                  pooled_ambiguous_n[BLOCK_HV_AB])),
            "threshold": float(floor_pooled), "direction": "lower",
            "control": ("smallest POOLED (across seeds) scorable AMBIGUOUS-band "
                        "step count over EVAL_HYPERVIGILANT, EVAL_REVERSION and "
                        "EVAL_HV_AB_PAIRED -- the band C1/C2 read -- against the "
                        f"pooled power floor ({MIN_BIN_COVERAGE_STEPS_POOLED} x "
                        f"{eval_tick_scale:.4g}). Red-team F2: a hypervigilant "
                        "agent can leave the band (smoke: 34 -> 9 ticks) and the "
                        "per-seed floor alone would let C1 be read on transit ticks."),
        },
        {
            "name": "e3_selections_per_block",
            "measured": float(min_e3_selections),
            "threshold": float(_scaled_floor(MIN_E3_SELECTIONS_PER_BLOCK, eval_tick_scale)),
            "direction": "lower",
            "control": ("smallest per-seed, per-block count of REAL E3 selections "
                        "(fresh select() calls; ticks between them re-issue the "
                        "held action). Red-team F2: ticks are not decisions -- "
                        "the E3 cadence is z_beta-driven and differs by block, so "
                        "the DV's effective n is the selection count, recorded "
                        "per block as block_e3_selection_count."),
        },
        {
            "name": "no_fatal_action_selection_errors",
            "measured": float(total_fatal), "threshold": 0.5, "direction": "upper",
            "control": ("total _agent_tick exceptions across every seed/block -- "
                        "each substitutes a RANDOM action into the DV stream."),
        },
    ]

    ready = bool(ready_a)
    try:
        preconditions = preconditions + p0_readiness_gate(gate_b_checks)
    except P0NotReady as e:
        preconditions = preconditions + e.preconditions
        ready = False

    # Falsify the two headroom ceilings from the run's OWN observations of the
    # same statistic each criterion reads (dv_headroom_observation_check
    # annotates the entry in place with headroom_ceiling_exceeded_by_observation).
    for _p in preconditions:
        if _p.get("name") == "c1_elevation_headroom":
            dv_headroom_observation_check(
                _p, [pooled_elevation], observed_name="pooled_elevation")
        elif _p.get("name") == "precision_margin_headroom":
            dv_headroom_observation_check(
                _p, [precision_margin_hv_elevation],
                observed_name="precision_margin_norm_hv_elevation")

    degeneracy = check_degeneracy({
        "precision_samples_hv": {
            "groups": [states[s]["block_results"][BLOCK_HYPERVIGILANT]["precision_samples"]
                       for s in seeds],
        },
        "ambiguous_lift_across_arms": [mean_base_lift, mean_hv_lift, mean_rev_lift],
    })
    non_degenerate = degeneracy["non_degenerate"]

    # ---- claim criteria (load-bearing) --------------------------------------
    c1_pass = bool(pooled_elevation >= FALSE_ALARM_ELEVATION_ABS)
    c1_nonoverlap_also_holds = bool(mean_hv_lift > float(np.max(base_lifts)))
    # 981's C2 "passed" on a ratio of two negatives (-0.04355 / -0.05198): HV was
    # BELOW baseline, so there was no elevation to revert. A reversion fraction
    # is only defined when the elevation it divides by is positive.
    c2_defined = bool(pooled_elevation > 1e-9)
    c2_pass = bool(c2_defined and pooled_recovered_fraction >= REVERSION_RECOVERY_FLOOR)

    all_pass = c1_pass and c2_pass
    criteria_met = sum([c1_pass, c2_pass])

    if not ready:
        status, evidence_direction, label = "FAIL", "non_contributory", "substrate_not_ready_requeue"
    elif not non_degenerate:
        status, evidence_direction, label = "FAIL", "non_contributory", "precision_channel_degenerate_vacuous_test"
    elif all_pass and not channel_c_inert:
        status, evidence_direction, label = "PASS", "supports", "hypervigilance_signature_reproduced_and_reverts"
    elif all_pass and channel_c_inert:
        # F4: the joint regime reproduced the signature and it reverted, but the
        # paired contrast shows suppressing replay contributed nothing
        # measurable -- (a)+(b) alone did it. The three-channel claim as
        # stated is only partly supported; recorded as a MEASURED split.
        status, evidence_direction, label = "PASS", "mixed", "hypervigilance_signature_reproduced_and_reverts_channels_ab_only_c_inert"
    elif c1_pass and not c2_pass and order_confound_plausible:
        # F3: an upward drift lands exactly here; BASELINE's own halves already
        # differ by half the bar, so the cell is not attributable.
        status, evidence_direction, label = "FAIL", "non_contributory", "false_alarm_elevation_confirmed_reversion_incomplete_order_confound_plausible"
    elif c1_pass and not c2_pass:
        status, evidence_direction, label = "FAIL", "mixed", "false_alarm_elevation_confirmed_reversion_incomplete"
    elif channel_a_selection_range < CHANNEL_A_RANGE_FLOOR:
        # F5: channel (a)'s only selection-layer consumer could not have moved
        # a membership rate by the bar (BASELINE non-argmin graded-pick
        # fraction below it), so this null is range-limited, not evidence.
        status, evidence_direction, label = "FAIL", "non_contributory", "false_alarm_elevation_not_confirmed_channel_a_range_limited"
    else:
        status, evidence_direction, label = "FAIL", "weakens", "false_alarm_elevation_not_confirmed"

    print(f"\nV3-EXQ-981a pooled verdict: {status}  label={label}  ({criteria_met}/2)", flush=True)

    criteria = [
        {"name": "C1_false_alarm_elevation", "load_bearing": True, "passed": c1_pass,
         "measured": pooled_elevation, "threshold": FALSE_ALARM_ELEVATION_ABS,
         "note": ("absolute elevation of the pooled AMBIGUOUS-band avoidance LIFT "
                  "(rate minus availability-matched chance; red-team F1), HV minus "
                  "BASELINE; feasibility certified by the c1_elevation_headroom "
                  "precondition; raw-rate elevation recorded as pooled_elevation_raw_rate")},
        {"name": "C2_reversion_recovery", "load_bearing": True, "passed": c2_pass,
         "measured": pooled_recovered_fraction, "threshold": REVERSION_RECOVERY_FLOOR,
         "note": ("undefined and recorded as FAIL when pooled_elevation <= 0 -- a "
                  "reversion fraction over a negative elevation is a ratio of two "
                  "negatives, which is how V3-EXQ-981's C2 'passed'")},
    ]
    combination_rule = (
        "TWO-STAGE. Gate A (positive control on pooled-count avoidance LIFT, "
        "per-seed and pooled band coverage, baseline offline-replay engagement "
        "via sws_n_writes, and the two dv_headroom feasibility checks) is "
        "evaluated after EVAL_BASELINE; EVAL_HYPERVIGILANT and "
        "EVAL_REVERSION run only if it is met. overall_pass = Gate A met AND "
        "Gate B met AND non_degenerate AND C1 (pooled AMBIGUOUS-band avoidant "
        f"rate elevation >= {FALSE_ALARM_ELEVATION_ABS} absolute, on a [0,1] DV) "
        "AND C2 (>= 50% of that elevation reverts, and the elevation is "
        "positive), all on the AMBIGUOUS-band LIFT. Any unmet precondition in "
        "either gate routes to substrate_not_ready_requeue, never a claim "
        "verdict. Red-team routings: PASS with |lift(HV) - lift(HV_AB_PAIRED)| "
        f"< {CHANNEL_C_CONTRAST_FLOOR} -> 'mixed' (channels a+b only, c inert); "
        "C1-and-not-C2 with |BASELINE half-drift| >= "
        f"{ORDER_DRIFT_FLOOR} -> non_contributory (order confound plausible); "
        "the weakens cell with BASELINE non-argmin graded-pick fraction < "
        f"{CHANNEL_A_RANGE_FLOOR} -> non_contributory (channel (a) range-limited)."
    )

    metrics = dict(stage_a_metrics)
    metrics.update({
        "mean_hv_ambiguous_rate": mean_hv_rate,
        "mean_reversion_ambiguous_rate": mean_rev_rate,
        "mean_hv_ambiguous_lift": mean_hv_lift,
        "mean_reversion_ambiguous_lift": mean_rev_lift,
        "mean_hv_ab_ambiguous_lift": mean_hv_ab_lift,
        "pooled_elevation": pooled_elevation,
        "pooled_elevation_raw_rate": pooled_elevation_raw_rate,
        "mean_ambiguous_chance_shift_hv_minus_base": mean_chance_shift,
        "pooled_baseline_ambiguous_lift_drift": pooled_baseline_drift,
        "order_confound_plausible": float(order_confound_plausible),
        "pooled_channel_c_contrast_ambiguous_lift": pooled_channel_c_contrast,
        "channel_c_inert": float(channel_c_inert),
        "channel_a_selection_range_nonargmin_frac_base": float(channel_a_selection_range),
        "min_engaged_fraction_per_block": min_engaged_frac_per_block,
        "min_e3_selections_per_block": float(min_e3_selections),
        "pooled_ambiguous_n_hv": float(pooled_ambiguous_n[BLOCK_HYPERVIGILANT]),
        "pooled_ambiguous_n_reversion": float(pooled_ambiguous_n[BLOCK_REVERSION]),
        "pooled_ambiguous_n_hv_ab": float(pooled_ambiguous_n[BLOCK_HV_AB]),
        "total_unscorable_in_safe": float(sum(r["block_unscorable_in_safe"][b] for r in arm_rows for b in BLOCKS_ALL)),
        "pooled_recovered_fraction": pooled_recovered_fraction,
        "mean_recovered_fraction": mean_recovered,
        "c1_nonoverlap_also_holds": float(c1_nonoverlap_also_holds),
        "c2_defined": float(c2_defined),
        "precision_base_mean": precision_base_mean,
        "precision_hv_mean": precision_hv_mean,
        "precision_ratio": precision_ratio,
        "effective_horizon_base_mean": eh_base_mean if eh_base_mean is not None else -1.0,
        "effective_horizon_hv_mean": eh_hv_mean if eh_hv_mean is not None else -1.0,
        "horizon_ratio": horizon_ratio if horizon_ratio is not None else -1.0,
        "draws_per_cycle_hv_max": float(max(draws_hv_vals) if draws_hv_vals else -1.0),
        "draws_per_cycle_base_min": float(min(draws_base_vals) if draws_base_vals else -1.0),
        "draws_per_cycle_reversion_min": float(min(draws_rev_vals) if draws_rev_vals else -1.0),
        "min_bin_coverage_steps_all_blocks": float(min_bin_coverage),
        "total_unscorable_ticks": float(total_unscorable),
        "total_fatal_errors": float(total_fatal),
        "criteria_met": float(criteria_met),
        "precision_margin_norm_hv_mean": precision_margin_hv_mean,
        "precision_margin_norm_hv_elevation": precision_margin_hv_elevation,
        "commit_temperature_eff_base_mean": commit_temp_base_mean,
        "commit_temperature_eff_hv_mean": commit_temp_hv_mean,
        "commit_temperature_eff_hv_reduction": commit_temp_hv_reduction,
        "precision_scaled_commit_engaged_total": float(precision_scaled_commit_engaged_total),
        "e3_selection_total": float(e3_selection_total),
        "precision_scaled_commit_engaged_fraction": precision_scaled_commit_engaged_fraction,
        "sleep_fires_min_per_seed_block": float(min_sleep_fires),
        "sleep_mech285_draws_hv_max": float(max(sleep_draws_hv_max_vals) if sleep_draws_hv_max_vals else -1.0),
        "sleep_mech285_draws_reversion_min": float(min(sleep_draws_rev_min_vals) if sleep_draws_rev_min_vals else -1.0),
        "sleep_sws_writes_hv_max": float(max(sws_writes_hv_max_vals) if sws_writes_hv_max_vals else -1.0),
        "sleep_sws_writes_reversion_min": float(min(sws_writes_rev_min_vals) if sws_writes_rev_min_vals else -1.0),
        "hv_offline_passes_suppressed_fraction_min": float(min_hv_suppressed_frac),
        "gate_a_met": float(ready_a),
        "gate_a_forced_for_smoke": float(gate_a_forced_for_smoke),
    })
    for r in arm_rows:
        s = r["seed"]
        metrics[f"seed{s}_base_ambiguous_rate"] = r["base_ambiguous_rate"]
        metrics[f"seed{s}_hv_ambiguous_rate"] = r["hv_ambiguous_rate"]
        metrics[f"seed{s}_reversion_ambiguous_rate"] = r["reversion_ambiguous_rate"]
        metrics[f"seed{s}_base_ambiguous_lift"] = r["base_ambiguous_lift"]
        metrics[f"seed{s}_hv_ambiguous_lift"] = r["hv_ambiguous_lift"]
        metrics[f"seed{s}_reversion_ambiguous_lift"] = r["reversion_ambiguous_lift"]
        metrics[f"seed{s}_hv_ab_ambiguous_lift"] = r["hv_ab_ambiguous_lift"]
        metrics[f"seed{s}_channel_c_contrast"] = r["channel_c_contrast_ambiguous_lift"]
        metrics[f"seed{s}_baseline_drift"] = r["baseline_ambiguous_lift_drift"]
        metrics[f"seed{s}_recovered_fraction"] = r["recovered_fraction"]
        metrics[f"seed{s}_commitment_threshold_after"] = r["commit_threshold_calibration"]["commitment_threshold_after"]

    seed_lines = "\n".join(
        f"| {r['seed']} | {r['base_ambiguous_lift']:.4f} | {r['hv_ambiguous_lift']:.4f} |"
        f" {r['reversion_ambiguous_lift']:.4f} | {r['hv_ab_ambiguous_lift']:.4f} | {r['recovered_fraction']:.4f} |"
        f" {r['block_precision_mean'][BLOCK_BASELINE]:.4g} |"
        f" {r['block_precision_mean'][BLOCK_HYPERVIGILANT]:.4g} |"
        for r in arm_rows)

    summary_markdown = f"""# V3-EXQ-981a -- MECH-027: control-plane hypervigilance signature probe

**Overall Status:** {status}  (label: `{label}`, {criteria_met}/2 load-bearing criteria)
**Supersedes:** V3-EXQ-981 (substrate_not_ready_requeue; confirmed autopsy 2026-09-03)
**Claim:** MECH-027 -- hypervigilance is a mis-tuned regime of elevated
gain/precision + shortened prediction horizon + suppressed replay.
**Seeds:** {seeds}

## Per-seed AMBIGUOUS-band avoidance LIFT (rate minus availability-matched chance)

| Seed | Baseline | Hypervigilant | Reversion | HV_AB paired | Recovered frac | precision(base) | precision(HV) |
|---|---|---|---|---|---|---|---|
{seed_lines}

## Preconditions (Gate A then Gate B)

{_fmt_preconditions(preconditions)}

## Criteria

- C1 (LOAD-BEARING) false-alarm elevation: {"PASS" if c1_pass else "FAIL"} (pooled_elevation={pooled_elevation:.4f} vs absolute bar {FALSE_ALARM_ELEVATION_ABS}; non-overlap diagnostic also holds: {c1_nonoverlap_also_holds})
- C2 (LOAD-BEARING) reversion recovery: {"PASS" if c2_pass else "FAIL"} (pooled_recovered_fraction={pooled_recovered_fraction:.4f} vs floor {REVERSION_RECOVERY_FLOOR}; defined={c2_defined})
- Channel (c) paired contrast lift(HV) - lift(HV_AB): {pooled_channel_c_contrast:.4f} (floor {CHANNEL_C_CONTRAST_FLOOR}; inert={channel_c_inert})
- BASELINE half-drift: {pooled_baseline_drift:.4f} (order-confound floor {ORDER_DRIFT_FLOOR}; plausible={order_confound_plausible})
- Channel (a) selection range (BASELINE non-argmin graded picks): {channel_a_selection_range:.4f} (floor {CHANNEL_A_RANGE_FLOOR})
- AMBIGUOUS chance shift HV - BASE: {mean_chance_shift:.4f} (corrected by reading lift)

{combination_rule}
"""

    manifest = _base_manifest(
        status, evidence_direction, label, dry_run, metrics, criteria,
        combination_rule, arm_rows, non_degenerate,
        degeneracy["degeneracy_reason"], preconditions,
        {"C1_false_alarm_elevation": bool(non_degenerate and c1_headroom_ok(preconditions)),
         "C2_reversion_recovery": bool(non_degenerate and c2_defined)},
        summary_markdown, RED_TEAM_NOTE,
    )

    full_config = {"seeds": seeds, **config_slice(), "thresholds": _thresholds()}

    if dry_run:
        print("[smoke] per-band scorable/unscorable counts and centre-cell "
              "calibration -- see the per-block prints above.", flush=True)

    out_path = write_flat_manifest(
        manifest, dry_run=dry_run, config=full_config, seeds=seeds,
        script_path=Path(__file__), started_at=t0,
        agent=[states[s]["agent"] for s in seeds],
    )
    return {"outcome": manifest["outcome"], "manifest": manifest, "out_path": out_path}


def c1_headroom_ok(preconditions: List[Dict[str, Any]]) -> bool:
    """C1 is non-degenerate only if the DV had the room to produce a passing
    value -- i.e. the c1_elevation_headroom precondition was met."""
    for p in preconditions:
        if p.get("name") == "c1_elevation_headroom":
            return bool(p.get("met"))
    return False


def _thresholds() -> Dict[str, Any]:
    return {
        "FALSE_ALARM_ELEVATION_ABS": FALSE_ALARM_ELEVATION_ABS,
        "HEADROOM_MARGIN_C1": HEADROOM_MARGIN_C1,
        "REVERSION_RECOVERY_FLOOR": REVERSION_RECOVERY_FLOOR,
        "PRECISION_RATIO_FLOOR": PRECISION_RATIO_FLOOR,
        "HORIZON_RATIO_CEIL": HORIZON_RATIO_CEIL,
        "POSITIVE_CONTROL_MARGIN": POSITIVE_CONTROL_MARGIN,
        "MIN_BIN_COVERAGE_STEPS_PER_SEED": MIN_BIN_COVERAGE_STEPS_PER_SEED,
        "MIN_BIN_COVERAGE_STEPS_POOLED": MIN_BIN_COVERAGE_STEPS_POOLED,
        "PRECISION_MARGIN_HV_ELEVATION_FLOOR": PRECISION_MARGIN_HV_ELEVATION_FLOOR,
        "PRECISION_SCALED_COMMIT_ENGAGED_FRACTION_FLOOR": PRECISION_SCALED_COMMIT_ENGAGED_FRACTION_FLOOR,
        "SLEEP_CYCLE_FIRE_FLOOR": SLEEP_CYCLE_FIRE_FLOOR,
        "SWS_WRITES_HV_CEIL": SWS_WRITES_HV_CEIL,
        "SWS_WRITES_BASELINE_FLOOR": SWS_WRITES_BASELINE_FLOOR,
        "HV_OFFLINE_PASSES_SUPPRESSED_FRACTION_FLOOR": HV_OFFLINE_PASSES_SUPPRESSED_FRACTION_FLOOR,
        "SWS_CONSOLIDATION_STEPS": SWS_CONSOLIDATION_STEPS,
        "REM_ATTRIBUTION_STEPS": REM_ATTRIBUTION_STEPS,
        "COMMIT_THRESHOLD_VARIANCE_MULTIPLE": COMMIT_THRESHOLD_VARIANCE_MULTIPLE,
        "MIN_E3_SELECTIONS_PER_BLOCK": MIN_E3_SELECTIONS_PER_BLOCK,
        "ORDER_DRIFT_FLOOR": ORDER_DRIFT_FLOOR,
        "CHANNEL_C_CONTRAST_FLOOR": CHANNEL_C_CONTRAST_FLOOR,
        "CHANNEL_A_RANGE_FLOOR": CHANNEL_A_RANGE_FLOOR,
        "HAZARD_SAFE_MAX": HAZARD_SAFE_MAX,
        "HAZARD_HIGH_MIN": HAZARD_HIGH_MIN,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--seeds", type=str, default=None)
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else (
        [SEEDS[0]] if args.dry_run else SEEDS
    )

    result = run_experiment(seeds, args.dry_run)
    out_path = result["out_path"]
    print(f"\nResult written to: {out_path}", flush=True)
    print(f"Status: {result['outcome']}", flush=True)
    for k, v in result["manifest"]["metrics"].items():
        print(f"  {k}: {v}", flush=True)

    raw = str(result["outcome"]).upper()
    return (raw if raw in ("PASS", "FAIL") else "FAIL"), out_path, args.dry_run


if __name__ == "__main__":
    _outcome_raw, _out_path, _dry = main()
    emit_outcome(outcome=_outcome_raw, manifest_path=_out_path, dry_run=_dry)
