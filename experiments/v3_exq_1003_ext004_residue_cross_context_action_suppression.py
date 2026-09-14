#!/opt/local/bin/python3
"""
V3-EXQ-1003: EXT-004 -- does residue suppress hazard-APPROACHING ACTIONS in a
novel context? (action-level DV, residue-isolating ERASED arm, headroom-gated;
supersedes V3-EXQ-991)

================================================================================
DO NOT QUEUE -- REFUSED 2026-09-09 (session wizardly-meninsky-e6c09c, W6-S5b item 2)
================================================================================
The 2026-09-09 redesign below is real and its DV measurement succeeded, but the
Step 4.5 red-team (fable, claude-fable-5-1) returned BLOCKING and all three
load-bearing findings were INDEPENDENTLY RE-MEASURED with this driver's own
functions before being accepted. No pre-registered criterion can discriminate.
FULL RECORD, with both measurements and the reproduction recipe:
  REE_assembly/evidence/planning/exq1003_ext004_action_level_refusal_20260909.md
  REE_assembly/evidence/planning/exq1003_redteam_fable_20260909.md
Governance flag raised (evidence_discrepancy, EXT-004 + ARC-013).

F1 THE ARMS ARE NOT RNG-PAIRED. run_seed seeds torch/numpy only before each
   arm's CONTEXT A (~1185, ~1304) and _ArmCell.__enter__ resets all RNG at arm
   entry, so EXPOSED's Context B starts 1600 Context-A steps into the stream
   while the ERASED twin's starts at the fresh seed state. CEM noise and E3's
   uncommitted multinomial (e3_selector.py:4249) draw from that stream.
   MEASURED, no manipulation difference: ERASED - EXPOSED = -0.126 (this
   session, 40-step twins) and 0.097 (red-team, 160-step twins), against a C1
   bar of 0.10. With the RNG state restored the un-erased clone reproduces
   EXPOSED to -0.0013, so the pairing fix works.
F2 AND FIXING F1 EXPOSES THE REAL BLOCKER. With the twins genuinely paired,
   erasing the ENTIRE Context-A residue field moves p_approach by -0.00103
   (this session) / +0.00013 (red-team, full 20x80 Context A) -- 97x to 770x
   under the 0.10 bar, and in the WRONG SIGN. The residue term's own
   approach-minus-retreat mean is NEGATIVE (-0.021 / -0.189): where it acts, it
   mildly favours approach. Mechanism: rho_residue * phi over 32 Gaussians at
   kernel_bandwidth 1.0, which config.py:3086 itself calls "~15x too wide for
   the z_world residual scale" -- one broad bump, not a map.
F3 THE CROSS-CONTEXT CONSTRUCT IS GONE BEFORE SCORING. RBFLayer.add_residue
   (residue/field.py:165-180) is a 32-slot ROUND ROBIN that MOVES a center to
   each new harm location; Context B is hazard-live and residue is written twice
   per committed harm step (agent.py:10747, e3_selector.py:4425). MEASURED: 0 of
   32 Context-A centers survive 40 Context-B steps (this session; 0/32 at 80 in
   the red-team). The design scores 1600. So the arm contrast is a
   sensitisation-GAIN contrast on Context-B locations, not transfer.

A successor needs RNG-paired arms, a FROZEN field during the read phase (a
harm-neutralised Context B, or a write-gated residue field), a residue channel
shown to move the readout by the bar under the REAL manipulation, and a
pre-registered reversed-effect label. None of that is designed here.
================================================================================

REDESIGNED 2026-09-09 (session wizardly-meninsky-e6c09c, campaign W6-S5b item 2)
================================================================================
THE 2026-09-04 DRAFT OF THIS DRIVER WAS BLOCKED, not by a bug but by a measured
construct-validity finding on its DV: `approach_rate` (up-gradient EXECUTED moves
per decision point) is FLOORED at 0.00-0.04 on this substrate because the executed
action stream in Context B is near-deterministically direction-locked (no optimizer
runs; CEM converges on one action). Measured 2026-09-04 and RE-MEASURED 2026-09-09
on the same driver code: fresh agent 0.0256 / 0.0175 / 0.0332 (seeds 42/43/45),
post-Context-A 0.0300 / 0.0042 (42/43); temperature does not de-lock it (entropy
0.10-0.76 nats at T 1-50 vs 1.609 uniform). A 0.08 suppression bar has no room
under a 0.00-0.04 control, and lowering the bar does not repair it (a locked
policy's between-arm delta is a per-seed Bernoulli lock direction).

THE DV MOVES ONE LAYER UP, TO THE COMMITTED-SELECTION LAYER, and it was MEASURED
BEFORE the bar was set (the chip's own requirement; scratchpad item2_probe/RESULTS.md,
2026-09-09, 5 cells x 240-400 Context-B steps, full 20x80 Context A for the EXPOSED
cells):
    p_approach = softmax(-scores / T_eff) probability mass on the APPROACH
                 candidates, divided by the mass on approach + retreat candidates,
                 read at each Context-B DECISION POINT on which a FRESH E3 selection
                 ran and the 32-candidate set contained at least one approach and
                 at least one retreat first-action.
  Free-policy range OBSERVED: per-cell means 0.4036 / 0.1626 / 0.2902 (fresh, seeds
  42/43/45) and 0.3947 / 0.0952 (post-Context-A EXPOSED, 42/43); within-cell SD
  0.26-0.41; per-tick min 0.0000-0.0349, max 0.9447-0.9985; 158 of 239 scored
  ticks strictly inside (0.05, 0.95). Score range was never zero (per-cell mean
  0.93-14.2). NOT degenerate on either axis, against an executed-action DV whose
  across-seed range was 0.016-0.026. T_eff was exactly 1.0 on every scored tick
  (the tonic/phasic temperature modulations are inert at these defaults).
  The COST, which the preconditions below carry: the MECH-090 held-action latch
  suppressed a fresh E3 selection on 33-88% of decision points, and of the fresh
  ones only 14-77% carried both directions in the candidate set (32 candidates
  decode to 2-4 distinct first actions), so usable n was 23-110 scored ticks per
  240-400 steps -- of order 130-730 per 1600-step cell, but 4x variable across
  seeds. Hence C5 is a per-cell SCORED-TICK floor and is a real gate.

WHAT THIS DV IS AND IS NOT. It is action-level: it reads the selection
distribution over the agent's own candidate first actions, upstream of the
multinomial/argmin quantizer (CLAUDE.md "Running the test suite": assert upstream
of the discrete quantizer, never on the sampled action), so a residue-driven
re-scoring of hazard-approaching candidates shows up here even when the committed
action is locked. It is NOT a behavioural rate; the executed `approach_rate` is
still RECORDED per cell (with its latch-replicated denominator, as a diagnostic)
so the two readouts can be compared. Two channels feed it and are recorded
separately: the SCORING channel (rho_residue * Phi_R per candidate,
e3_selector.py score_trajectory, read via the E3 per-candidate score
decomposition, e3_score_decomp_enabled -- diagnostics only, verified record-only
at source) and the PROPOSER channel (residue-biased CEM terrain,
hippocampal/module.py _terrain_informed_mean), which changes candidate
COMPOSITION and is recorded as candidate_approach_share. p_approach conditions on
the approach+retreat subset, so it isolates the scoring channel; a proposer-only
effect would appear in candidate_approach_share and n_unscorable_no_contrast.

SAMPLE-SIZE INTEGRITY. E3 diagnostics latch; a read on a tick with no fresh
selection re-records the previous one. This driver does NOT null
agent.e3.last_scores before select_action: REEAgent.select_action reads the
PREVIOUS tick's last_scores at three sites before reaching e3.select (agent.py
MECH-342 maintenance-release margin, SD-061 stuck-state detector, dACC payoff
proxy -- all inert on this config, but nulling would perturb the substrate under
any config that arms them). Instead freshness comes from the SHARED sentinel-key helper
experiments/_lib/fresh_select.py (FreshSelectProbe.watch around select_action:
select() reassigns last_score_diagnostics wholesale, so a stamped private key
is ABSENT afterwards iff a genuine selection ran -- the corpus discharge path
the e3-exemption-backlog contract requires of a new driver), with the
FreshSelectCounter's yield / replication_factor / hold-duration stats recorded
per cell. agent.e3.select is ALSO wrapped with a pass-through recorder, only
to capture the candidate list handed to select() so score/candidate alignment
is asserted, not assumed; the two freshness signals must agree on every tick
(n_freshness_disagreement, gated to 0 with n_align_mismatch in P6).
n_latched_dp counts decision points with no fresh selection; nothing is recorded
on them.

WHAT IS UNCHANGED AND VERIFIED (do not redo): the ERASED-arm residue isolation
(_detach_agent_buffers walks the whole reachable graph; post-clone assertion: the
clone differs from EXPOSED in exactly the residue_field state_dict entries);
--dry-run completes end-to-end.
================================================================================

SLEEP DRIVER: N/A (use_sleep_loop is not set; agent.sleep_loop is None by default).

WHY THIS IS A NEW NUMBER AND NOT V3-EXQ-991a
----------------------------------------------
The DV changes -- from an occupancy measure to an action-level choice measure --
and a third arm is added, so this is a redesign of the experimental design, not a
bug fix to an unchanged one (CLAUDE.md "EXQ Versioning": a new number is for "the
mechanism under test changed, or the experimental design is substantially
different"). The chip that routed this work says so explicitly: "New number if the
DV changes (it will)." `supersedes: V3-EXQ-991`. DESIGN HISTORY: the id V3-EXQ-1003 was reserved on
2026-09-04 for the executed-action version of this design, which was never
queued (its DV was found floored before the queue write); this 2026-09-09
redesign keeps the id because nothing under it ever ran or was recorded on
origin, and the WORKSPACE_STATE record of 2026-09-04T20:17:01Z is the only
prior reference.

WHY THIS RE-RUN EXISTS (read before changing any threshold)
-------------------------------------------------------------
V3-EXQ-991 ran 61 min on ree-worker-3 and was adjudicated **non_contributory** by
`failure_autopsy_ext-claim-probe-cluster_2026-09-03.md` (confirmed; ratified by
governance-20260903T2013 with a red-team pass). Four defects, all measurement or
design, none a substrate ceiling:

  (1) THE DV MEASURED THE WRONG THING. `harm_rate_B` = mean harm magnitude per
      Context-B step. Context B is size=8 with `use_proxy_fields=True`, and the
      proxy "hazard_approach" harm channel fires whenever `hazard_field >=
      proximity_approach_threshold` (0.15). Probed directly on this exact env
      config (2026-09-04): the hazard field spans 0.354-1.400 and **100% of cells
      clear the 0.15 threshold**, so a harm signal is emitted on essentially every
      step regardless of what the agent chooses. `harm_rate_B` therefore measures
      mean hazard-field OCCUPANCY, not the action-level suppression the ARC-013
      residue rider actually names.
  (2) THE `non_contributory` BRANCH WAS STRUCTURALLY UNREACHABLE. Non-vacuity
      required `naive_control_clean` = NAIVE's Context-A `n_harm_events == 0`.
      NAIVE sets `hazard_harm=0.0`, `harm_gradient_scale=0.0`,
      `contamination_spread=0.0` AND `proximity_harm_scale=0.0`, which removes
      every harm source in the env -- so that detector is constant-TRUE by
      construction and can never fail.
  (3) THERE WAS NO POSITIVE CONTROL. Nothing established that the DV could move
      at all before a null was read off it.
  (4) THE MANIPULATION WAS NOT ISOLATED TO RESIDUE. The driver's own
      `attribution_caveat` records it: EXPOSED-vs-NAIVE differs the agent's ENTIRE
      Context-A experience stream, and ARC-108's `w_chan`/`V-hat_t` (agent.py
      :3413-3415, persist across episodes) and the MECH-165 exploration buffer are
      both uncontrolled alternative carriers. A positive result was not
      attributable to residue.

WEAK-NULL RESIDUAL CARRIED, NOT ERASED (governance red-team amendment 4)
--------------------------------------------------------------------------
The downgrade to `non_contributory` does not mean V3-EXQ-991 measured nothing, and
this redesign does not treat it as unmeasured. Recorded verbatim on EXT-004's
`evidence_quality_note` and carried here as the prior:

  * The DV was NOT pinned -- `harm_rate_B` spanned 0.0024 to 0.1423 across the ten
    cells, a 59x range. Its problem was construct validity, not dynamic range.
  * The manipulation was real on both sides: EXPOSED reached residue mean_weight
    0.214-2.380 at coverage 1.0 on 5/5 seeds; NAIVE recorded 0 harm events and 0.0
    residue on 5/5.
  * And 4 of 5 seeds showed NO transfer (mean exposed 0.0549 vs naive 0.0480,
    ratio 1.18 -- if anything the wrong way) against a 15%-reduction bar at n=5.

So the prior is a WEAK, UNDERPOWERED null on the rider prediction -- weak because
the DV was measuring occupancy, underpowered at n=5. It is not evidence of absence,
and this run is designed to convert it into a real measurement rather than to
re-pose the same question at higher n.

THE NEW DV -- PRE-COMMIT APPROACH PROBABILITY MASS AT A REAL CHOICE (action-level)
------------------------------------------------------------------------------------
At every Context-B step, BEFORE `env.step()`:

    hf   = env.get_hazard_field()                      # [size, size], agent-independent
    here = hf[agent_x, agent_y]
    for each MOVEMENT action a in {0: (-1,0), 1: (1,0), 2: (0,-1), 3: (0,1)}:
        dest(a) = (agent_x+dx, agent_y+dy)
        -- SKIPPED when dest(a) is off-grid (the env leaves the agent in place on
           an out-of-bounds move, verified 2026-09-04), so a blocked move carries
           no approach/retreat information.
        grad(a) = hf[dest(a)] - here

    A step is a DECISION POINT iff BOTH an approach and a retreat are available:
    max_a grad(a) >= +GRAD_EPS AND min_a grad(a) <= -GRAD_EPS.

    On a decision point where E3 ran a FRESH selection this tick (see SAMPLE-SIZE
    INTEGRITY above), read the per-candidate scores E3 just scored
    (agent.e3.last_scores, the SAME vector both the argmin and the multinomial
    branch consume; lower = preferred) and each candidate's first action
    (Trajectory.actions[0, 0].argmax() % action_dim -- the decode
    e3_selector.select() itself applies to the winner). Classify each candidate
    as APPROACH (grad(first_action) > 0), RETREAT (< 0) or NEITHER (stay,
    off-grid, flat). If the set holds at least one of each:

        p_approach(tick) = sum_{approach} softmax(-scores/T_eff)
                         / (sum_{approach} + sum_{retreat}) softmax(-scores/T_eff)

    with T_eff the temperature actually handed to e3.select() that tick.
    Otherwise the tick is UNSCORABLE (n_unscorable_no_contrast) and records
    nothing to the DV.

    DV per cell:  p_approach_b = mean over scored ticks of p_approach(tick)

Arithmetic range [0, 1]; a candidate set scored uniformly sits at the candidate
approach share (~0.1-0.4 on this substrate: most candidates decode to one
direction plus stay). The executed-action `approach_rate` is recorded alongside
as `approach_rate_b` (diagnostic; latch-replicated denominator, as in the 09-04
draft), with `argmin_is_approach_frac` (how often the committed winner was an
approach candidate) bridging the two. `GRAD_EPS = 0.01` against a measured
cell-to-cell field delta of order 0.05-0.2 on this config.

THREE ARMS -- the third one is what makes a positive attributable
------------------------------------------------------------------
  EXPOSED  Context A hazard-live (harm_gradient_scale=0.30, hazard_harm=0.5): real
           harm, so `ResidueField.accumulate()` writes genuine structure.
  NAIVE    Context A harm-neutralised (hazard_harm=0, harm_gradient_scale=0,
           contamination_spread=0, proximity_harm_scale=0; hazard OBJECTS remain at
           the same count and env-seed-derived layout, so world structure is
           matched and only the capacity to harm is removed). The clean-slate arm.
  ERASED   **NEW -- the residue-only control, and defect (4)'s fix.** The EXPOSED
           agent is `deepcopy`d immediately after Context A and its harm-residue
           RBF state is zeroed (`rbf_field.weights`, `active_mask`,
           `next_center_idx`, `valence_vecs`, `sensitization_gain`,
           `total_residue`, `num_harm_events`, `_harm_history`), then it replays the
           SAME Context B. Every other piece of persistent cross-episode state --
           ARC-108's `w_chan`/`V-hat_t`, the MECH-165 exploration buffer, the E1/E2
           weights -- is bit-identical to EXPOSED's. This is exactly the third arm
           V3-EXQ-991's own red-team pass proposed and its author declined on
           compute grounds; the budget arithmetic below shows it is now affordable.

  Zeroing the RBF is COMPLETE, not partial: `RBFLayer.forward` computes
  `active_weights = self.weights * self.active_mask.float()`
  (ree_core/residue/field.py:143), so zeroing both tensors removes every
  accumulated contribution, and `ResidueField.evaluate` adds only
  `neural_field(z) * 0.1` on top (field.py:604-616). The `neural_field` is
  **untrained and identical across all three arms** -- this driver runs no
  optimizer at all, so it stays at its seeded initialisation everywhere -- so it
  cannot carry an arm difference.

  ATTRIBUTION, which is the point of the arm:
    EXPOSED < ERASED  ->  the suppression is carried by RESIDUE (ARC-013).
    EXPOSED ~ ERASED, both < NAIVE  ->  something carried it, but NOT residue --
                                        ARC-108/MECH-165 are the live candidates.
    EXPOSED ~ ERASED ~ NAIVE  ->  no cross-context suppression at all.

PRE-REGISTERED THRESHOLDS (constants below; never inferred post-hoc)
----------------------------------------------------------------------
  C1 (LOAD-BEARING, residue-isolated)
       mean_over_seeds(p_approach_ERASED - p_approach_EXPOSED) >= 0.10
  C2 (overall behavioural claim, the question V3-EXQ-991 asked)
       mean_over_seeds(p_approach_NAIVE - p_approach_EXPOSED) >= 0.10
  C3   effect >= 0.80 SD of the cross-seed paired C1 delta.
  C4   C1 delta positive on >= 4 of 6 seeds.
  C5   data quality: min SCORED-TICK count per cell >= 60.

  WHY 0.10 -- DERIVED FROM THE MEASURED RANGE, not from a predecessor (the DV is
  new) and not from construction alone:
    (a) NOISE. Within-cell SD of p_approach(tick) measured 0.26-0.41; at the
        C5 floor of 60 scored ticks the per-cell SE is <= 0.41/sqrt(60) = 0.053,
        so a paired per-seed delta has SE <= 0.075 and the 6-seed mean has
        SE <= 0.031. 0.10 is >= 3.2 pooled SE at the WORST admissible n, and
        ~5 SE at the expected n (130-730 scored ticks per 1600-step cell).
    (b) ROOM. Measured control-state means 0.16-0.40 (fresh) / 0.10-0.39
        (post-Context-A EXPOSED). C1 is a suppression, so the room is the ERASED
        arm's MEAN p_approach down to 0; at margin 2.0 the bar needs >= 0.20 of
        mean room, which the measured 0.29 fresh mean clears and which H2 checks
        on this run's own ERASED cells, mean-matched to how C1 aggregates.
    (c) The cross-seed SD of the per-cell MEAN was 0.098 (fresh); the design is
        PAIRED within seed (ERASED is EXPOSED's own twin), which removes that.
  C5 = 60 is the n at which a single scored tick moves a cell mean by <= 1.7% and
  (a) holds; it is the per-cell floor the 2026-09-09 probe showed can FAIL (23-31
  scored ticks in two 240-step cells), so it is a real gate, scaled up to the
  1600-step budget.

TWO dv_headroom PRECONDITIONS (kind `dv_headroom`, ree-v3 8e133d26ed)
------------------------------------------------------------------------
Both are built with `_metrics.dv_headroom_check` and evaluated by
`_metrics.p0_readiness_gate`; an unmet entry raises P0NotReady and this driver
writes `substrate_not_ready_requeue` / `non_contributory` -- never a verdict on
EXT-004 or ARC-013.

  H1 dv_headroom_instrument_range   **P0, BEFORE ANY SCORED CELL RUNS.** A fresh
      agent runs INSTRUMENT_PROBE_STEPS Context-B steps; on every scored tick the
      readout is computed twice from the SAME recorded scores: as-is, and with a
      penalty of +score_range added to every APPROACH candidate's score (a
      candidate-specific perturbation at the committed-selection layer -- the
      layer CLAUDE.md says a verify-lift control must be injected at). The two
      per-tick means are the control values, statistic "range". This certifies
      the readout is in the SCORE-SENSITIVE regime at the observed T_eff and
      score scale -- if the softmax were flat, p_approach would only report
      candidate composition and could not carry a scoring-channel effect.
      Required: range >= C1 x 1.0 (H1, bare feasibility under a synthetic
      selection-layer perturbation -- margin 2.0 is reserved for H2, the
      scientific room), and separately, as the P4 readiness entry, the FRACTION
      of the as-is approach mass the penalty removes >= DV_INSTRUMENT_SCORE_
      SENSITIVITY_FLOOR (0.50; scale-free so it cannot conflate with the probe
      seed's baseline level). It aborts before the compute, not after.
  H2 dv_headroom_suppression_room   POST-RUN, on the ERASED control arm. C1 is a
      SUPPRESSION criterion read on the MEAN over seeds of paired deltas, so the
      room that matters is mean_over_seeds(p_approach_ERASED) - 0.0, passed as
      `achievable=` (MEAN-MATCHED: `dv_achievable`'s "floor_headroom" is a
      min-based order statistic, which the validate_experiments
      dv_headroom-statistic-mismatch lint rejects for a mean criterion -- the
      V3-EXQ-972a shape). The min-based figure is still recorded. Falsified from
      the run's own C1 delta at emit time by dv_headroom_observation_check.

NON-DEGENERACY PRECONDITIONS (breach -> substrate_not_ready_requeue, NOT a verdict)
------------------------------------------------------------------------------------
  P1 exposed_residue_populated   EXPOSED mean_weight > 1e-4 AND coverage > 0 after
                                 Context A. CAN fail (it is an empirical property
                                 of how much harm Context A actually delivered).
  P2 erased_residue_zeroed       ERASED's residue mean_weight == 0 AND active
                                 centers == 0 at Context-B entry. CAN fail -- it
                                 verifies that the deepcopy-and-zero actually took,
                                 which is a real implementation risk, not a
                                 tautology.
  P3 scored_tick_density         min over cells of n_scored (fresh-selection
                                 decision points with both directions in the
                                 candidate set) >= 60. CAN fail -- measured
                                 23-31 in two 240-step probe cells.
  P4 dv_instrument_score_sensitivity  the fraction of approach mass removed by
                                 H1's perturbation >= 0.50. CAN fail (a flat
                                 softmax).
  P5 residue_term_live_in_e3_scores   on EXPOSED cells, the per-candidate residue
                                 score term (rho_residue * Phi_R, from E3's
                                 score decomposition) has a cross-candidate
                                 spread > 0 on the scored ticks -- the residue
                                 field DIFFERENTIATES candidates in the scoring
                                 channel the DV reads. CAN fail (a field whose
                                 evaluate() is flat over the candidates' z_world
                                 paths). Worst cell across seeds.
  P6 score_candidate_alignment   n_align_mismatch == 0 over every cell -- the
                                 recorded score vector always had exactly one
                                 entry per candidate handed to select().
  H1, H2                         the two dv_headroom entries above.

  DELIBERATELY NOT A GATE: `naive_residue_empty` (NAIVE's Context-A harm events ==
  0). This is V3-EXQ-991's `naive_control_clean`, and it is constant-TRUE by
  construction -- see defect (2). It is RECORDED as a diagnostic
  (`naive_residue_empty_diagnostic`) so the structural guarantee is still checked
  and visible, but it is not counted toward non-vacuity, because a precondition
  that cannot fail contributes nothing to non-vacuity and makes the gate look
  stronger than it is. Removing it, rather than keeping a green tautology, is what
  makes the `non_contributory` branch reachable: P1-P6, H1 and H2 can each
  fail on real data.

DV-SYMMETRY DECLARATION (mandatory, one line per arm)
--------------------------------------------------------
DV = `p_approach`, the softmax(-scores/T_eff) mass on approach candidates
conditioned on the approach+retreat subset, over per-candidate E3 scores whose
candidate SET is itself a CEM search terrain-biased by the residue field
(`hippocampal/module.py:475-476` `_terrain_informed_mean` reads
`residue_field.evaluate(z_world)` with no `rho_residue` gate; `e3_selector.py`
score_trajectory applies `rho_residue * Phi_R` at scoring). Symmetry group of a
softmax over candidate scores: permutation of candidate index order and a
uniform additive constant across ALL candidates (it cancels in the
normalisation). A monotone RESCALING is NOT a symmetry of a softmax (it is a
temperature change) -- and T_eff is recorded per tick so a rescaling would be
visible. The conditioning on approach+retreat is a fixed function of the
candidate first actions and the env gradient, not of the scores.
  EXPOSED  the field has genuine per-location structure from real harm events, so
           candidate scores differ by an amount depending on WHERE harm occurred --
           not a broadcast constant (candidate trajectories pass through different
           z_world regions) and not a monotone map of any other arm's score vector.
           NOT invariant.
  NAIVE    the field is structurally empty, so the residue term contributes only
           `neural_field(z)*0.1`, which still varies per candidate (different
           z_world points) and is not a constant. NOT invariant -- and it is a
           genuinely DIFFERENT function from EXPOSED's, not a shifted or rescaled
           one, which is what makes the contrast a manipulation rather than an
           identity.
  ERASED   identical to NAIVE in residue content but identical to EXPOSED in every
           other persistent state. NOT invariant, for NAIVE's reason.
  And the DV is read ONLY on fresh E3 selections (the e3.select wrapper), so a
  held/committed action's replication over the MECH-090 hold cannot re-record
  one selection as many; the executed-action diagnostic is NOT so protected and
  is labelled as such.

INTERPRETATION GRID (every branch, and the direction each records)
--------------------------------------------------------------------
  preconditions red, or H1/H2 unmet
      -> substrate_not_ready_requeue | non_contributory for BOTH claims.
         Nothing is concluded about EXT-004 or ARC-013.
  C1 pass (and C3, C4, C5)
      -> residue_carries_cross_context_action_suppression
         EXT-004 supports, ARC-013 supports.
  C1 fail, C2 pass
      -> cross_context_suppression_present_but_not_residue_attributable
         EXT-004 supports (the behavioural claim holds: an agent that incurred
         harm in Context A approaches hazards less in a novel Context B),
         ARC-013 weakens (with every other persistent state held fixed, removing
         the residue field did NOT remove the effect -- so residue is not the
         carrier). This branch is the whole reason the ERASED arm exists.
  C1 fail, C2 fail
      -> no_cross_context_action_level_suppression
         EXT-004 weakens, ARC-013 weakens.

WHAT A NULL MEANS HERE, AND WHAT IT DOES NOT (mandatory null declaration)
---------------------------------------------------------------------------
A `no_cross_context_action_level_suppression` result with all preconditions green
and both headroom gates met means: with a demonstrably live, score-sensitive,
action-level selection readout, an agent that incurred real harm in Context A did
not down-weight hazard-approaching candidates at its own decision points in a
structurally novel Context B relative to either a harm-naive agent or its own
residue-erased twin. With V3-EXQ-991's weak null
(4/5 seeds no transfer on an occupancy DV) that is the second null on this
prediction and the first on a construct-valid DV -- material evidence against the
EXT-004 rider as operationalised here.

It does NOT mean: (a) that residue leaves no trace -- P1 measures a populated,
covered field; (b) that residue never influences behaviour -- this DV asks only
about the pre-commit weighting of hazard-gradient candidates in a held-out
context, isolated to the scoring channel (a proposer-only effect would show in
candidate_approach_share, recorded but not gated); (c) that a longer Context A,
a larger context shift, or a different action-level operationalisation would also
return null. The scope is this DV, this context pair, this exposure budget.

A `cross_context_suppression_present_but_not_residue_attributable` result is NOT a
null: it is a positive finding about EXT-004 and a negative one about ARC-013, and
it should be routed to whichever of ARC-108 / MECH-165 the follow-up isolates.

COMPUTE BUDGET
-----------------
V3-EXQ-991 measured 61.1 min on ree-worker-3 for 10 cells x 1080 steps = 10,800
steps -> 0.339 s/step. Here: per seed, EXPOSED (1600 Context-A + 1600 Context-B),
NAIVE (1600 + 1600), ERASED (1600 Context-B only -- it inherits EXPOSED's Context
A by deepcopy, which is what makes the third arm affordable) = 8,000 steps, plus a
one-off 400-step instrument probe. 6 seeds -> 48,000 steps -> ~271 min; the
e3.select wrapper and per-tick softmax add negligible cost (measured 675 s for
five 240-400-step probe cells including their 1600-step Context-A phases).
estimated_minutes 330 with margin.

SEEDS: 42, 43, 45, 46, 47, 48. Seed 44 is excluded -- known reef-config instability
(v3_exq_538a / 539 / 540 precedent), same exclusion V3-EXQ-991 made.

RIDER TAGGING
----------------
`claim_ids = ["EXT-004", "ARC-013"]`. ARC-013 ("Residue is persistent latent-space
curvature; hippocampal paths form a cognitive map") is EXT-004's own named
`ree_mechanism` rider and is the claim the ERASED arm directly ablates, so the two
claims can and DO take different directions in the middle branch above --
`evidence_direction_per_claim` is emitted for exactly that reason. EXT-004's other
named riders are not tagged: IMPL-005 is an implementation claim this design does
not exercise, and INV-008 ("Precision is routed and depth-specific, not global")
names precision routing, which nothing here manipulates.

RED-TEAM (Step 4.5): see the queue entry `note` for the verdict and the model.
"""

import argparse
import copy
import types
import statistics
from collections import deque
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from ree_core.agent import REEAgent  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402

from experiment_protocol import emit_outcome  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator  # noqa: E402
from experiments._lib.fresh_select import FreshSelectProbe, FreshSelectCounter  # noqa: E402
from _metrics import (  # noqa: E402
    P0NotReady,
    dv_achievable,
    dv_headroom_check,
    dv_headroom_observation_check,
    p0_readiness_gate,
)

# ------------------------------------------------------------------ #
# Identity                                                           #
# ------------------------------------------------------------------ #
EXPERIMENT_TYPE = "v3_exq_1003_ext004_residue_cross_context_action_suppression"
QUEUE_ID = "V3-EXQ-1003"
SUPERSEDES = "V3-EXQ-991"
PREDECESSOR_RUN_ID = (
    "v3_exq_991_claim_probe_ext_004_residue_cross_context_penalty_20260903T122245Z_v3"
)
CLAIM_IDS = ["EXT-004", "ARC-013"]
EXPERIMENT_PURPOSE = "evidence"
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"
BACKLOG_ID = "EVB-1249"

ARM_EXPOSED = "EXPOSED"
ARM_NAIVE = "NAIVE"
ARM_ERASED = "ERASED"
ARMS = (ARM_EXPOSED, ARM_NAIVE, ARM_ERASED)

SEEDS = [42, 43, 45, 46, 47, 48]  # 44 excluded: known reef-config instability
RHO_RESIDUE = 0.5  # substrate default, IDENTICAL in every arm

P0_CONTEXT_A_EPISODES = 20
EVAL_CONTEXT_B_EPISODES = 20
STEPS_PER_EPISODE = 80

# ------------------------------------------------------------------ #
# Pre-registered thresholds -- see the docstring for each derivation  #
# ------------------------------------------------------------------ #
# 2026-09-09 redesign: DV = p_approach (pre-commit softmax mass), bars DERIVED
# from the measured free-policy range (module docstring "WHY 0.10").
THRESH_C1_RESIDUE_GAP = 0.10      # LOAD-BEARING: ERASED - EXPOSED p_approach
THRESH_C2_OVERALL_GAP = 0.10      # NAIVE - EXPOSED p_approach
THRESH_C3_EFFECT_SD = 0.80        # effect in SD of the cross-seed paired C1 delta
THRESH_C4_MIN_SEEDS = 4           # of 6, C1 delta positive
THRESH_C5_MIN_SCORED_TICKS = 60   # per cell, worst case (fresh-selection DPs with both directions)

GRAD_EPS = 0.01   # hazard-field delta below which a move carries no approach info

# Depth cap for the pre-deepcopy graph-attached-tensor walk (_detach_agent_buffers).
# 14 comfortably covers agent -> module -> attribute-object -> dict -> list -> tensor
# (the measured 2026-09-04 failure path was depth 8) without unbounded recursion.
_DETACH_MAX_DEPTH = 14

# dv_headroom wiring (ree-v3 8e133d26ed).
DV_NAME = "p_approach_gap_residue"
DV_HEADROOM_MARGIN = 2.0          # the bar must sit at most half the room away
DV_HEADROOM_CONTROL_ARM = ARM_ERASED
DV_BOUNDS = (0.0, 1.0)            # p_approach is a probability mass
DV_INSTRUMENT_SCORE_SENSITIVITY_FLOOR = 0.50  # P4: FRACTION of approach mass a +score_range penalty removes
DV_HEADROOM_MARGIN_H1 = 1.0       # H1: bare feasibility under a selection-layer perturbation
# E3 freshness is established with the SHARED sentinel-key helper
# (experiments/_lib/fresh_select.py, the corpus discharge path -- the
# e3-exemption-backlog contract pins the marker count, so a new driver must use
# the helper, never the blanket opt-out marker). This driver does NOT null
# agent.e3.last_scores / last_score_decomp before select_action: REEAgent
# .select_action reads the PREVIOUS tick's last_scores at three sites upstream of
# e3.select (MECH-342 maintenance-release margin, SD-061 stuck-state detector,
# dACC payoff proxy), so nulling would perturb the substrate. The e3.select
# wrapper in _step_episode is kept ONLY to capture the candidate list handed to
# select() for score/candidate alignment; the freshness gate is the probe.
FRESH_SELECT_NAMESPACE = "exq1003"
RESIDUE_TERM_SPREAD_FLOOR = 0.0   # P5: strict '>' -- any cross-candidate residue-term spread

# Non-degeneracy floors.
FLOOR_RESIDUE_MEAN_WEIGHT = 1e-4  # P1

INSTRUMENT_PROBE_SEED = 991003
INSTRUMENT_PROBE_STEPS = 400

CONTEXT_A_ENV_KWARGS_BY_ARM = {
    ARM_EXPOSED: dict(
        size=10, num_hazards=3, num_resources=5,
        harm_gradient_enabled=True, harm_gradient_scale=0.30,
        hazard_harm=0.5, resource_respawn_on_consume=True,
    ),
    ARM_NAIVE: dict(
        size=10, num_hazards=3, num_resources=5,
        harm_gradient_enabled=True, harm_gradient_scale=0.0,
        hazard_harm=0.0, contamination_spread=0.0,
        # CausalGridWorldV2 defaults use_proxy_fields=True, which drives a
        # SEPARATE continuous "hazard_approach" harm source independent of
        # hazard_harm / harm_gradient_scale / contamination_spread
        # (causal_grid_world.py:2506). V3-EXQ-991 confirmed empirically that
        # without this line a "NAIVE" env still produced 178/200 harm events.
        proximity_harm_scale=0.0,
        resource_respawn_on_consume=True,
    ),
}
# ERASED shares EXPOSED's Context A by construction (deepcopy), so it has no
# Context-A kwargs entry of its own.
CONTEXT_B_ENV_KWARGS = dict(
    size=8, num_hazards=2, num_resources=4,
    harm_gradient_enabled=True, harm_gradient_scale=0.30,
    resource_respawn_on_consume=True,
)

OUT_DIR = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"

_ZG = ZGoalStreamAccumulator()

# The four MOVEMENT actions (index 4 is stay; see CausalGridWorldV2.ACTIONS).
MOVES: Dict[int, Tuple[int, int]] = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}


# ------------------------------------------------------------------ #
# Observation helpers (carried unchanged from V3-EXQ-991)             #
# ------------------------------------------------------------------ #
def _obs(obs_dict: Dict[str, Any], key: str) -> Optional[torch.Tensor]:
    v = obs_dict.get(key)
    if v is None:
        return None
    v = v.float()
    if v.dim() == 1:
        v = v.unsqueeze(0)
    return v


def _split_obs(obs_dict: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
    body = obs_dict["body_state"].float()
    world = obs_dict["world_state"].float()
    if body.dim() == 1:
        body = body.unsqueeze(0)
    if world.dim() == 1:
        world = world.unsqueeze(0)
    return body, world


def _context_seeds(seed: int) -> Tuple[int, int]:
    """Deterministic env seeds for (Context A, Context B), paired across arms."""
    return int(seed) * 1000 + 1, int(seed) * 1000 + 2


def _build_agent(env: CausalGridWorldV2) -> REEAgent:
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
    )
    cfg.latent.alpha_world = 0.9   # SD-008: z_world fidelity for residue geography
    cfg.e3.rho_residue = float(RHO_RESIDUE)   # pinned; identical in every arm
    agent = REEAgent(cfg)
    # Diagnostics-only flag (verified record-only at source, e3_selector.py:
    # every `if self.e3_score_decomp_enabled:` branch writes a float into a dict
    # and nothing else): exposes the per-candidate score decomposition, from
    # which P5 reads the residue term's cross-candidate spread.
    agent.e3.e3_score_decomp_enabled = True
    return agent


# ------------------------------------------------------------------ #
# The DV                                                             #
# ------------------------------------------------------------------ #
def _gradient_options(env: CausalGridWorldV2) -> Dict[int, float]:
    """hazard-field gradient for each AVAILABLE movement action at the agent's cell.

    An off-grid destination is omitted entirely rather than scored as zero: the env
    leaves the agent in place on an out-of-bounds move (verified empirically on this
    config), so such an action carries no approach/retreat information and must not
    dilute the decision-point test.
    """
    hf = env.get_hazard_field()
    ax, ay = int(env.agent_x), int(env.agent_y)
    here = float(hf[ax, ay])
    out: Dict[int, float] = {}
    for a, (dx, dy) in MOVES.items():
        nx, ny = ax + dx, ay + dy
        if 0 <= nx < env.size and 0 <= ny < env.size:
            out[a] = float(hf[nx, ny]) - here
    return out


def _is_decision_point(grads: Dict[int, float]) -> bool:
    """True iff BOTH an approach and a retreat are actually available."""
    if len(grads) < 2:
        return False
    vals = list(grads.values())
    return max(vals) >= GRAD_EPS and min(vals) <= -GRAD_EPS


class ScoreReadoutRecorder:
    """Accumulates the pre-commit DV (p_approach) over a Context-B phase.

    Records ONLY on decision points where a FRESH E3 selection ran this tick
    (the e3.select wrapper in _step_episode fired) and the candidate set held
    both an approach and a retreat first action. Everything else is counted,
    never scored.
    """

    def __init__(self) -> None:
        self.n_steps = 0
        self.n_decision_points = 0
        self.n_fresh_dp = 0            # decision points with a fresh selection
        self.n_latched_dp = 0          # decision points with NO fresh selection
        self.n_unscorable_no_contrast = 0
        self.n_align_mismatch = 0
        self.n_scored = 0
        self.n_freshness_disagreement = 0   # probe.fresh != wrapper fired (must be 0)
        self.fresh_counter = FreshSelectCounter()
        self.p_approach: List[float] = []
        self.p_approach_penalised: List[float] = []   # H1 instrument control
        self.score_margin_norm: List[float] = []
        self.score_range: List[float] = []
        self.eff_temp: List[float] = []
        self.candidate_approach_share: List[float] = []
        self.n_approach_cands: List[int] = []
        self.n_retreat_cands: List[int] = []
        self.argmin_is_approach: List[bool] = []
        self.residue_term_spread: List[float] = []            # max - min over candidates
        self.residue_term_approach_minus_retreat: List[float] = []
        self.n_residue_term_missing = 0

    def observe(self, scores_t: Any, cands: List[Any], grads: Dict[int, float],
                action_dim: int, eff_temp: float, decomp: Optional[Dict[str, Any]]) -> None:
        import torch.nn.functional as F
        scores = scores_t.detach().float().reshape(-1)
        K = int(scores.numel())
        if K != len(cands):
            self.n_align_mismatch += 1
            return
        first = [int(c.actions[0, 0].argmax().item()) % int(action_dim) for c in cands]
        appr = [i for i, a in enumerate(first) if grads.get(a) is not None and grads[a] > 0.0]
        retr = [i for i, a in enumerate(first) if grads.get(a) is not None and grads[a] < 0.0]
        srange = float(scores.max().item() - scores.min().item())
        if not appr or not retr:
            self.n_unscorable_no_contrast += 1
            return
        t = float(eff_temp) if eff_temp and eff_temp > 0.0 else 1.0

        def _p(sc: Any) -> float:
            pr = F.softmax(-sc / t, dim=0)
            ma = float(pr[appr].sum().item())
            mr = float(pr[retr].sum().item())
            return (ma / (ma + mr)) if (ma + mr) > 0.0 else float("nan")

        p = _p(scores)
        pen = scores.clone()
        pen[appr] = pen[appr] + max(srange, 1e-6)
        p_pen = _p(pen)
        self.n_scored += 1
        self.p_approach.append(p)
        self.p_approach_penalised.append(p_pen)
        mean_a = float(scores[appr].mean().item())
        mean_r = float(scores[retr].mean().item())
        self.score_margin_norm.append((mean_a - mean_r) / srange if srange > 0.0 else float("nan"))
        self.score_range.append(srange)
        self.eff_temp.append(t)
        self.candidate_approach_share.append(len(appr) / float(len(appr) + len(retr)))
        self.n_approach_cands.append(len(appr))
        self.n_retreat_cands.append(len(retr))
        self.argmin_is_approach.append(bool(int(scores.argmin().item()) in appr))
        per = (decomp or {}).get("per_candidate") if isinstance(decomp, dict) else None
        if per and len(per) == K and all(isinstance(d, dict) and "residue_weighted" in d for d in per):
            rt = [float(d["residue_weighted"]) for d in per]
            self.residue_term_spread.append(max(rt) - min(rt))
            self.residue_term_approach_minus_retreat.append(
                float(np.mean([rt[i] for i in appr]) - np.mean([rt[i] for i in retr])))
        else:
            self.n_residue_term_missing += 1

    @staticmethod
    def _mean(v: List[float]) -> float:
        f = [x for x in v if x == x]
        return float(np.mean(f)) if f else float("nan")

    @property
    def p_approach_mean(self) -> float:
        return self._mean(self.p_approach)

    def summary(self) -> Dict[str, Any]:
        return {
            "p_approach_b": self.p_approach_mean,
            "p_approach_sd_b": (float(np.std(self.p_approach)) if len(self.p_approach) > 1 else float("nan")),
            "p_approach_min_b": (float(min(self.p_approach)) if self.p_approach else float("nan")),
            "p_approach_max_b": (float(max(self.p_approach)) if self.p_approach else float("nan")),
            "p_approach_penalised_mean_b": self._mean(self.p_approach_penalised),
            "n_scored_b": int(self.n_scored),
            "n_fresh_dp_b": int(self.n_fresh_dp),
            "n_latched_dp_b": int(self.n_latched_dp),
            "n_unscorable_no_contrast_b": int(self.n_unscorable_no_contrast),
            "n_align_mismatch_b": int(self.n_align_mismatch),
            "n_freshness_disagreement_b": int(self.n_freshness_disagreement),
            "fresh_select_b": self.fresh_counter.as_dict(self.n_steps, include_hist=False),
            "scored_fraction_of_dp_b": (float(self.n_scored) / float(self.n_decision_points)
                                        if self.n_decision_points else float("nan")),
            "score_margin_norm_mean_b": self._mean(self.score_margin_norm),
            "score_range_mean_b": self._mean(self.score_range),
            "eff_temp_min_b": (float(min(self.eff_temp)) if self.eff_temp else float("nan")),
            "eff_temp_max_b": (float(max(self.eff_temp)) if self.eff_temp else float("nan")),
            "candidate_approach_share_mean_b": self._mean(self.candidate_approach_share),
            "n_approach_cands_mean_b": self._mean([float(x) for x in self.n_approach_cands]),
            "n_retreat_cands_mean_b": self._mean([float(x) for x in self.n_retreat_cands]),
            "argmin_is_approach_frac_b": (float(np.mean(self.argmin_is_approach))
                                          if self.argmin_is_approach else float("nan")),
            "residue_term_spread_mean_b": self._mean(self.residue_term_spread),
            "residue_term_spread_max_b": (float(max(self.residue_term_spread))
                                          if self.residue_term_spread else float("nan")),
            "residue_term_approach_minus_retreat_mean_b": self._mean(self.residue_term_approach_minus_retreat),
            "n_residue_term_missing_b": int(self.n_residue_term_missing),
            "p_approach_per_tick_b": [round(float(x), 6) for x in self.p_approach],
        }


class ApproachRecorder:
    """Accumulates the EXECUTED-action diagnostic (the 09-04 draft's DV) over a
    Context-B phase. Latch-replicated denominator; recorded, never gated."""

    def __init__(self) -> None:
        self.n_decision_points = 0
        self.n_approach = 0
        self.n_steps = 0
        self.n_latched = 0

    def observe(self, grads: Dict[int, float], action_idx: int) -> None:
        self.n_steps += 1
        if not _is_decision_point(grads):
            return
        self.n_decision_points += 1
        g = grads.get(int(action_idx))
        if g is not None and g > 0.0:
            self.n_approach += 1

    @property
    def approach_rate(self) -> float:
        if self.n_decision_points == 0:
            return float("nan")
        return float(self.n_approach) / float(self.n_decision_points)


# ------------------------------------------------------------------ #
# Stepping                                                           #
# ------------------------------------------------------------------ #
def _step_episode(
    agent: REEAgent,
    env: CausalGridWorldV2,
    steps: int,
    recorder: Optional[ApproachRecorder] = None,
    score_rec: Optional[ScoreReadoutRecorder] = None,
) -> Tuple[float, int]:
    """Step one episode. Returns (cumulative_harm_magnitude, n_harm_events).

    When `recorder` is given, the executed-action diagnostic is accumulated.
    When `score_rec` is given, agent.e3.select is wrapped with a pass-through
    recorder for the episode so the DV is read ONLY on ticks where a fresh E3
    selection ran (module docstring, SAMPLE-SIZE INTEGRITY), with the candidate
    list handed to select() captured for score/candidate alignment.
    """
    _flat, obs_dict = env.reset()
    agent.reset()
    z_self_prev: Optional[torch.Tensor] = None
    action_prev: Optional[torch.Tensor] = None
    cum_harm = 0.0
    n_harm_events = 0

    cap: Dict[str, Any] = {"fired": False, "cands": None, "temp": None}
    _orig_select = agent.e3.select
    probe = FreshSelectProbe(FRESH_SELECT_NAMESPACE) if score_rec is not None else None
    if score_rec is not None:
        score_rec.fresh_counter.flush()   # no hold may span an episode
    if score_rec is not None:
        def _wrapped_select(candidates, temperature, *a, **kw):
            cap["fired"] = True
            cap["cands"] = candidates
            cap["temp"] = float(temperature)
            return _orig_select(candidates, temperature, *a, **kw)
        agent.e3.select = _wrapped_select

    try:
        for _step in range(steps):
            body, world = _split_obs(obs_dict)
            latent = agent.sense(
                obs_body=body, obs_world=world,
                obs_harm=_obs(obs_dict, "harm_obs"),
                obs_harm_a=_obs(obs_dict, "harm_obs_a"),
                obs_harm_history=_obs(obs_dict, "harm_history"),
            )

            if z_self_prev is not None and action_prev is not None:
                agent.record_transition(z_self_prev, action_prev, latent.z_self.detach())

            ticks = agent.clock.advance()
            wdim = latent.z_world.shape[-1]
            e1_prior = (
                agent._e1_tick(latent) if ticks.get("e1_tick", False)
                else torch.zeros(1, wdim, device=agent.device)
            )
            candidates = agent.generate_trajectories(latent, e1_prior, ticks)

            if agent.goal_state is not None:
                try:
                    energy = float(body[0, 3].item())
                except Exception:
                    energy = 1.0
                agent.update_z_goal(benefit_exposure=0.0, drive_level=max(0.0, 1.0 - energy))

            cap["fired"] = False
            cap["cands"] = None
            cap["temp"] = None
            if probe is not None:
                with probe.watch(agent) as sel:
                    action = agent.select_action(candidates, ticks, temperature=1.0)
                fresh = bool(sel.fresh)
            else:
                action = agent.select_action(candidates, ticks, temperature=1.0)
                fresh = bool(cap["fired"])
            action_idx = int(action.argmax().item()) % env.action_dim

            # The DV is read BEFORE the step, from the agent's actual decision point.
            grads = _gradient_options(env)
            is_dp = _is_decision_point(grads)
            if recorder is not None:
                recorder.observe(grads, action_idx)
            if score_rec is not None:
                score_rec.n_steps += 1
                score_rec.fresh_counter.record(fresh)
                if fresh != bool(cap["fired"]):
                    score_rec.n_freshness_disagreement += 1
                if is_dp:
                    score_rec.n_decision_points += 1
                    if not fresh:
                        score_rec.n_latched_dp += 1
                    else:
                        score_rec.n_fresh_dp += 1
                        scores_t = agent.e3.last_scores
                        cands = cap["cands"]
                        if scores_t is None or cands is None:
                            score_rec.n_align_mismatch += 1
                        else:
                            score_rec.observe(
                                scores_t, list(cands), grads, env.action_dim,
                                cap["temp"], getattr(agent.e3, "last_score_decomp", None),
                            )

            _flat_obs, harm_signal, done, info, obs_dict = env.step(action_idx)
            agent.update_residue(float(harm_signal))

            if harm_signal < 0:
                cum_harm += -float(harm_signal)
                n_harm_events += 1

            z_self_prev = latent.z_self.detach()
            action_prev = action.detach()
            if done:
                _flat, obs_dict = env.reset()
                agent.reset()
                z_self_prev = None
                action_prev = None
    finally:
        if score_rec is not None:
            try:
                del agent.e3.select   # restore the class method
            except Exception:
                agent.e3.select = _orig_select

    return cum_harm, n_harm_events


def _detach_agent_buffers(agent: REEAgent) -> Tuple[int, Dict[int, Any]]:
    """Detach graph-attached tensors anywhere in the agent's object graph.

    `copy.deepcopy` REFUSES a non-leaf tensor ("Only Tensors created explicitly by
    the user (graph leaves) support the deepcopy protocol"), and the agent's
    cross-episode state holds plenty of them: the experience buffers
    (`_self_experience_buffer`, `_world_experience_buffer`,
    `_action_experience_buffer`, `_e2_transition_buffer`), the hippocampal
    MECH-165 `_exploration_buffer`, and -- the case a shallow walk misses --
    tensors nested inside PLAIN PYTHON OBJECTS that are themselves held in a
    module attribute (measured 2026-09-04: the first version of this function
    walked only `agent.modules()` + one level of container, and `deepcopy` still
    died on a tensor reached via object -> dict -> object -> dict -> list).
    So this walks the whole reachable graph: containers (dict / list / tuple /
    deque), `nn.Module._modules` / `_buffers`, and any object's `__dict__` or
    `__slots__`, with an id-keyed `seen` set for cycles and a depth cap.

    Detaching is safe HERE because this driver runs no optimizer at all -- no
    backward pass is ever taken through any of this state -- and it is NECESSARY
    because those same buffers are exactly the cross-episode state the ERASED arm
    must INHERIT bit-identically.

    A `state_dict` round-trip is NOT an acceptable substitute: it carries
    parameters and registered buffers only, and the MECH-165 exploration buffer
    (`hippocampal._exploration_buffer`) plus the agent's experience buffers are
    plain python attributes, so a round-trip would silently drop the very carriers
    the ERASED arm exists to hold fixed. `nn.Parameter`s are returned untouched
    (they are leaves by construction, and replacing one with a plain tensor would
    change the module).

    The same walk also collects a `deepcopy` MEMO seeding every imported PYTHON
    MODULE object reachable from the agent to ITSELF, because `deepcopy` tries to
    pickle a module and dies (`TypeError: cannot pickle 'module' object`, measured
    2026-09-04 -- a module held as an ordinary attribute inside the agent's
    object graph). Sharing a module object between the two agents is exactly
    right: a module is imported code, carries no per-agent state, and is already
    shared by every other object in the process. Nothing that carries agent state
    is shared this way -- the memo contains module objects only.

    Returns `(n_tensors_detached, deepcopy_memo)`.
    """
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
            # Share, never copy -- see the memo paragraph in the docstring.
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


def _zero_residue_field(agent: REEAgent) -> None:
    """Erase the accumulated harm-residue trace IN PLACE, completely.

    `RBFLayer.forward` computes `active_weights = weights * active_mask.float()`
    (ree_core/residue/field.py:143), so zeroing both removes every accumulated
    contribution; the remaining `neural_field(z)*0.1` term (field.py:604-616) is
    untrained and identical across arms because this driver runs no optimizer.
    """
    rf = agent.residue_field
    with torch.no_grad():
        rf.rbf_field.weights.data.zero_()
        rf.rbf_field.active_mask.zero_()
        rf.rbf_field.next_center_idx.zero_()
        rf.rbf_field.valence_vecs.zero_()
        rf.rbf_field.sensitization_gain.zero_()
        rf.total_residue.zero_()
        rf.num_harm_events.zero_()
    try:
        rf._harm_history.clear()
    except Exception:
        pass


def _residue_state(agent: REEAgent) -> Dict[str, float]:
    stats = agent.residue_field.get_statistics()
    telem = agent.residue_field.get_coverage_telemetry()
    return {
        "mean_weight": float(stats["mean_weight"].item()),
        "active_centers": float(stats["active_centers"].item()),
        "coverage_pct": float(telem["residue_coverage_pct"]),
        "total_residue": float(agent.residue_field.total_residue.item()),
    }


# ------------------------------------------------------------------ #
# H1: instrument positive control (runs BEFORE any scored cell)      #
# ------------------------------------------------------------------ #
def instrument_positive_control(*, dry_run: bool) -> Dict[str, Any]:
    """Score-sensitivity positive control on the READOUT (H1 / P4), BEFORE any
    scored cell. A fresh agent runs Context B; on every scored tick p_approach
    is computed twice from the SAME recorded scores -- as-is, and with
    +score_range added to every APPROACH candidate (a candidate-specific
    perturbation at the committed-selection layer). If that cannot move the
    readout by DV_INSTRUMENT_SCORE_SENSITIVITY_FLOOR the softmax is flat at the
    observed temperature/score scale and the DV would only report candidate
    composition; nothing downstream would be interpretable.
    """
    steps = 40 if dry_run else INSTRUMENT_PROBE_STEPS
    torch.manual_seed(INSTRUMENT_PROBE_SEED)
    np.random.seed(INSTRUMENT_PROBE_SEED)
    env = CausalGridWorldV2(seed=INSTRUMENT_PROBE_SEED, **CONTEXT_B_ENV_KWARGS)
    agent = _build_agent(env)
    rec = ApproachRecorder()
    srec = ScoreReadoutRecorder()
    _step_episode(agent, env, steps, recorder=rec, score_rec=srec)
    p_base = srec.p_approach_mean
    p_pen = srec._mean(srec.p_approach_penalised)
    separation = (p_base - p_pen) if (p_base == p_base and p_pen == p_pen) else float("nan")
    # Scale-free sensitivity: the FRACTION of the as-is approach mass the
    # penalty removes. Absolute separation is what H1's dv_headroom range reads.
    fraction_removed = (separation / p_base) if (separation == separation and p_base > 0.0) else float("nan")
    summ = srec.summary()
    summ.pop("p_approach_per_tick_b", None)
    return {
        "p_approach_as_is": p_base,
        "p_approach_approach_penalised": p_pen,
        "separation": separation,
        "fraction_removed": fraction_removed,
        "control_values": [p_pen, p_base],
        "floor": DV_INSTRUMENT_SCORE_SENSITIVITY_FLOOR,
        "met": bool(fraction_removed == fraction_removed and fraction_removed >= DV_INSTRUMENT_SCORE_SENSITIVITY_FLOOR),
        "n_scored": int(srec.n_scored),
        "n_steps": int(steps),
        "executed_approach_rate_diagnostic": rec.approach_rate,
        "detail": summ,
    }


# ------------------------------------------------------------------ #
# Per-seed cells                                                     #
# ------------------------------------------------------------------ #
def _context_b_phase(
    agent: REEAgent, env_seed_b: int, n_eval: int, steps: int,
) -> Tuple[ApproachRecorder, ScoreReadoutRecorder, float, int]:
    env_b = CausalGridWorldV2(seed=env_seed_b, **CONTEXT_B_ENV_KWARGS)
    rec = ApproachRecorder()
    srec = ScoreReadoutRecorder()
    cum_harm_b = 0.0
    n_events_b = 0
    for ep in range(n_eval):
        h, n = _step_episode(agent, env_b, steps, recorder=rec, score_rec=srec)
        cum_harm_b += h
        n_events_b += n
    return rec, srec, cum_harm_b, n_events_b


def _row(arm: str, seed: int, rec: ApproachRecorder, srec: ScoreReadoutRecorder,
         residue: Dict[str, float],
         env_seed_a: int, env_seed_b: int, cum_harm_a: float, n_events_a: int,
         cum_harm_b: float, n_events_b: int) -> Dict[str, Any]:
    row = {
        "arm_id": arm,
        "seed": int(seed),
        "env_seed_a": int(env_seed_a),
        "env_seed_b": int(env_seed_b),
        # executed-action DIAGNOSTIC (the 09-04 draft's DV; latch-replicated)
        "approach_rate_b": rec.approach_rate,
        "n_decision_points_b": int(rec.n_decision_points),
        "n_approach_b": int(rec.n_approach),
        "n_steps_b": int(rec.n_steps),
        "decision_point_density_b": (
            float(rec.n_decision_points) / float(max(1, rec.n_steps))
        ),
        "context_a_cum_harm": float(cum_harm_a),
        "context_a_n_harm_events": int(n_events_a),
        "context_b_cum_harm": float(cum_harm_b),
        "context_b_n_harm_events": int(n_events_b),
        "residue_mean_weight_at_b_entry": residue["mean_weight"],
        "residue_active_centers_at_b_entry": residue["active_centers"],
        "residue_coverage_pct_at_b_entry": residue["coverage_pct"],
        "residue_total_at_b_entry": residue["total_residue"],
    }
    row.update(srec.summary())   # the DV (p_approach_b) and its per-tick record
    return row


def run_seed(seed: int, *, dry_run: bool) -> List[Dict[str, Any]]:
    """Run all three arms for one seed. ERASED is derived from EXPOSED post-A."""
    p0 = 2 if dry_run else P0_CONTEXT_A_EPISODES
    n_eval = 2 if dry_run else EVAL_CONTEXT_B_EPISODES
    steps = 20 if dry_run else STEPS_PER_EPISODE
    env_seed_a, env_seed_b = _context_seeds(seed)
    rows: List[Dict[str, Any]] = []

    config_slice_common = {
        "rho_residue": RHO_RESIDUE,
        "alpha_world": 0.9,
        "context_b_env_kwargs": CONTEXT_B_ENV_KWARGS,
        "p0_context_a_episodes": p0,
        "eval_context_b_episodes": n_eval,
        "steps_per_episode": steps,
        "env_seed_a": env_seed_a,
        "env_seed_b": env_seed_b,
        "grad_eps": GRAD_EPS,
        "e3_score_decomp_enabled": True,
        "dv": "p_approach",
    }

    # ---- EXPOSED (and, from its post-Context-A state, ERASED) --------------
    print(f"Seed {seed} Condition {ARM_EXPOSED}", flush=True)
    with arm_cell(
        seed,
        config_slice=dict(config_slice_common,
                          arm=ARM_EXPOSED,
                          context_a_env_kwargs=CONTEXT_A_ENV_KWARGS_BY_ARM[ARM_EXPOSED]),
        script_path=Path(__file__),
    ) as cell:
        torch.manual_seed(seed)
        np.random.seed(seed)
        env_a = CausalGridWorldV2(
            seed=env_seed_a, **CONTEXT_A_ENV_KWARGS_BY_ARM[ARM_EXPOSED]
        )
        agent = _build_agent(env_a)
        cum_harm_a = 0.0
        n_events_a = 0
        for ep in range(p0):
            h, n = _step_episode(agent, env_a, steps)
            cum_harm_a += h
            n_events_a += n
            if (ep + 1) % 10 == 0 or ep == p0 - 1:
                print(
                    f"  [train] ext004_action_suppression seed={seed} "
                    f"arm={ARM_EXPOSED} ep {ep + 1}/{p0 + n_eval} phase=context_a",
                    flush=True,
                )
        # Snapshot BEFORE Context B so ERASED inherits exactly the state EXPOSED
        # carries into Context B -- ARC-108 w_chan/V-hat_t and the MECH-165
        # exploration buffer included -- differing ONLY in residue content.
        n_detached, _dc_memo = _detach_agent_buffers(agent)
        n_shared_modules = len(_dc_memo)
        erased_agent = copy.deepcopy(agent, _dc_memo)
        _zero_residue_field(erased_agent)
        # Verify the clone differs from EXPOSED in the residue field and NOTHING
        # else: every other state_dict tensor must be bit-identical. This is the
        # design's isolation claim, asserted rather than assumed.
        _sd_a, _sd_e = agent.state_dict(), erased_agent.state_dict()
        _differing = sorted(
            k for k in _sd_a
            if not torch.equal(_sd_a[k].to(torch.float64) if _sd_a[k].is_floating_point()
                               else _sd_a[k],
                               _sd_e[k].to(torch.float64) if _sd_e[k].is_floating_point()
                               else _sd_e[k])
        )
        _unexpected = [k for k in _differing if "residue_field" not in k]
        if _unexpected:
            raise AssertionError(
                "ERASED clone differs from EXPOSED outside the residue field, so "
                "the arm does not isolate residue: "
                f"{_unexpected[:8]} (n={len(_unexpected)}). Fix the clone; do NOT "
                "score this run."
            )
        print(
            f"  [erase] seed={seed} detached={n_detached} tensors, "
            f"shared_modules={n_shared_modules}; clone differs "
            f"from EXPOSED in {len(_differing)} state_dict entries, all under "
            f"residue_field",
            flush=True,
        )

        residue_exposed = _residue_state(agent)
        rec, srec, cum_harm_b, n_events_b = _context_b_phase(
            agent, env_seed_b, n_eval, steps
        )
        _ZG.observe(agent)
        row_exposed = _row(ARM_EXPOSED, seed, rec, srec, residue_exposed,
                           env_seed_a, env_seed_b, cum_harm_a, n_events_a,
                           cum_harm_b, n_events_b)
        cell.stamp(row_exposed)
    rows.append(row_exposed)
    print(
        f"  [eval] seed={seed} arm={ARM_EXPOSED} "
        f"p_approach={row_exposed['p_approach_b']:.4f} "
        f"n_scored={row_exposed['n_scored_b']} "
        f"approach_rate_exec={row_exposed['approach_rate_b']:.4f} "
        f"n_decision_points={row_exposed['n_decision_points_b']}",
        flush=True,
    )
    print(f"verdict: {'PASS' if row_exposed['n_scored_b'] > 0 else 'FAIL'}",
          flush=True)

    # ---- ERASED (Context B only; inherits EXPOSED's Context A) -------------
    print(f"Seed {seed} Condition {ARM_ERASED}", flush=True)
    with arm_cell(
        seed,
        config_slice=dict(config_slice_common,
                          arm=ARM_ERASED,
                          context_a_env_kwargs=CONTEXT_A_ENV_KWARGS_BY_ARM[ARM_EXPOSED],
                          derived_from=ARM_EXPOSED,
                          residue_zeroed_post_context_a=True),
        script_path=Path(__file__),
        # The ERASED cell is NOT an independent function of (substrate, config,
        # seed): it is derived from the EXPOSED cell's post-Context-A agent, so
        # it shares mutable cross-cell state and must never be reused as a mint.
        extra_ineligible_reasons=["derived_by_deepcopy_from_exposed_cell"],
    ) as cell:
        residue_erased = _residue_state(erased_agent)
        rec_e, srec_e, cum_harm_b_e, n_events_b_e = _context_b_phase(
            erased_agent, env_seed_b, n_eval, steps
        )
        _ZG.observe(erased_agent)
        row_erased = _row(ARM_ERASED, seed, rec_e, srec_e, residue_erased,
                          env_seed_a, env_seed_b, cum_harm_a, n_events_a,
                          cum_harm_b_e, n_events_b_e)
        cell.stamp(row_erased)
    rows.append(row_erased)
    print(
        f"  [eval] seed={seed} arm={ARM_ERASED} "
        f"p_approach={row_erased['p_approach_b']:.4f} "
        f"n_scored={row_erased['n_scored_b']} "
        f"approach_rate_exec={row_erased['approach_rate_b']:.4f} "
        f"n_decision_points={row_erased['n_decision_points_b']} "
        f"residue_mean_weight={row_erased['residue_mean_weight_at_b_entry']:.6g}",
        flush=True,
    )
    print(f"verdict: {'PASS' if row_erased['n_scored_b'] > 0 else 'FAIL'}",
          flush=True)

    # ---- NAIVE -------------------------------------------------------------
    print(f"Seed {seed} Condition {ARM_NAIVE}", flush=True)
    with arm_cell(
        seed,
        config_slice=dict(config_slice_common,
                          arm=ARM_NAIVE,
                          context_a_env_kwargs=CONTEXT_A_ENV_KWARGS_BY_ARM[ARM_NAIVE]),
        script_path=Path(__file__),
    ) as cell:
        torch.manual_seed(seed)
        np.random.seed(seed)
        env_a_n = CausalGridWorldV2(
            seed=env_seed_a, **CONTEXT_A_ENV_KWARGS_BY_ARM[ARM_NAIVE]
        )
        agent_n = _build_agent(env_a_n)
        cum_harm_a_n = 0.0
        n_events_a_n = 0
        for ep in range(p0):
            h, n = _step_episode(agent_n, env_a_n, steps)
            cum_harm_a_n += h
            n_events_a_n += n
            if (ep + 1) % 10 == 0 or ep == p0 - 1:
                print(
                    f"  [train] ext004_action_suppression seed={seed} "
                    f"arm={ARM_NAIVE} ep {ep + 1}/{p0 + n_eval} phase=context_a",
                    flush=True,
                )
        residue_naive = _residue_state(agent_n)
        rec_n, srec_n, cum_harm_b_n, n_events_b_n = _context_b_phase(
            agent_n, env_seed_b, n_eval, steps
        )
        _ZG.observe(agent_n)
        row_naive = _row(ARM_NAIVE, seed, rec_n, srec_n, residue_naive,
                         env_seed_a, env_seed_b, cum_harm_a_n, n_events_a_n,
                         cum_harm_b_n, n_events_b_n)
        cell.stamp(row_naive)
    rows.append(row_naive)
    print(
        f"  [eval] seed={seed} arm={ARM_NAIVE} "
        f"p_approach={row_naive['p_approach_b']:.4f} "
        f"n_scored={row_naive['n_scored_b']} "
        f"approach_rate_exec={row_naive['approach_rate_b']:.4f} "
        f"n_decision_points={row_naive['n_decision_points_b']}",
        flush=True,
    )
    print(f"verdict: {'PASS' if row_naive['n_scored_b'] > 0 else 'FAIL'}",
          flush=True)

    return rows


# ------------------------------------------------------------------ #
# Gates and analysis                                                 #
# ------------------------------------------------------------------ #
def _by(rows: List[Dict[str, Any]], arm: str) -> List[Dict[str, Any]]:
    return [r for r in rows if r["arm_id"] == arm]


def _finite(vals: List[float]) -> List[float]:
    return [v for v in vals if v == v]


def build_preconditions(rows: List[Dict[str, Any]],
                        instrument: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The four non-headroom preconditions. Every one can fail on real data.

    `naive_residue_empty` is deliberately NOT here -- it is constant-TRUE by
    construction and is recorded as a diagnostic instead. See the docstring.
    """
    exposed = _by(rows, ARM_EXPOSED)
    erased = _by(rows, ARM_ERASED)
    checks: List[Dict[str, Any]] = []

    checks.append({
        "name": "exposed_residue_populated",
        "kind": "readiness",
        "description": (
            "EXPOSED's residue field must actually carry structure at Context-B "
            "entry -- worst cell across seeds. An empty field means there was "
            "nothing to transfer and the contrast is against nothing"
        ),
        "control": "Context A hazard-live (hazard_harm=0.5, harm_gradient_scale=0.30)",
        "measured": (
            float(min(r["residue_mean_weight_at_b_entry"] for r in exposed))
            if exposed else 0.0
        ),
        "threshold": FLOOR_RESIDUE_MEAN_WEIGHT,
        "direction": "lower",
    })
    checks.append({
        "name": "erased_residue_zeroed",
        "kind": "readiness",
        "description": (
            "the ERASED arm's residue must be genuinely gone at Context-B entry -- "
            "worst (highest) active-centre count across seeds must be at most 0. "
            "This verifies the deepcopy-and-zero took; an incomplete erase would "
            "silently turn the residue-isolation arm into a second EXPOSED"
        ),
        "control": "deepcopy of the EXPOSED agent post-Context-A, RBF state zeroed",
        "measured": (
            float(max(r["residue_active_centers_at_b_entry"] for r in erased))
            if erased else 1.0
        ),
        "threshold": 0.0,
        "direction": "upper",
        "comparator": "<=",
    })
    checks.append({
        "name": "scored_tick_density",
        "kind": "readiness",
        "description": (
            "minimum count of SCORED ticks per cell (Context-B decision points on "
            "which a fresh E3 selection ran AND the candidate set held both an "
            "approach and a retreat first action) -- the DV's denominator. At 60 "
            "a single tick moves a cell mean by <= 1.7% and the paired C1 SE bound "
            "in the docstring holds. Measured 23-31 in two 240-step probe cells, "
            "so this floor can fail"
        ),
        "control": "Context B hazard field spans 0.354-1.400 on this config",
        "measured": float(min(r["n_scored_b"] for r in rows)) if rows else 0.0,
        "threshold": float(THRESH_C5_MIN_SCORED_TICKS),
        "direction": "lower",
    })
    checks.append({
        "name": "dv_instrument_score_sensitivity",
        "kind": "readiness",
        "description": (
            "FRACTION of the as-is p_approach removed when +score_range is added to "
            "every approach candidate's score (1 - penalised/as-is, on the "
            "instrument probe's mean over scored ticks) -- certifies the readout is "
            "in the score-sensitive softmax regime at the observed T_eff and score "
            "scale, not merely reporting candidate composition. Scale-free, so it "
            "does not conflate with the probe seed's baseline level"
        ),
        "control": (
            "a fresh agent's own Context-B scores, perturbed at the committed-"
            "selection layer on the approach candidates only"
        ),
        "measured": float(instrument["fraction_removed"]),
        "threshold": float(DV_INSTRUMENT_SCORE_SENSITIVITY_FLOOR),
        "direction": "lower",
    })
    _rt = [r["residue_term_spread_mean_b"] for r in exposed
           if r["residue_term_spread_mean_b"] == r["residue_term_spread_mean_b"]]
    checks.append({
        "name": "residue_term_live_in_e3_scores",
        "kind": "readiness",
        "description": (
            "worst EXPOSED cell's mean cross-candidate spread of the residue score "
            "term (rho_residue * Phi_R per candidate, from E3's score decomposition) "
            "over its scored ticks -- the residue field must DIFFERENTIATE "
            "candidates in the scoring channel the DV reads, or the manipulation "
            "cannot reach the DV through that channel at all"
        ),
        "control": "EXPOSED cells (populated field); ERASED/NAIVE spreads recorded as diagnostics",
        "measured": float(min(_rt)) if _rt else 0.0,
        "threshold": float(RESIDUE_TERM_SPREAD_FLOOR),
        "direction": "lower",
        "comparator": ">",
    })
    checks.append({
        "name": "score_candidate_alignment",
        "kind": "readiness",
        "description": (
            "total ticks on which the recorded E3 score vector did not have exactly "
            "one entry per candidate handed to select(), PLUS ticks on which the "
            "shared FreshSelectProbe sentinel and the e3.select capture wrapper "
            "disagreed about freshness -- must be 0"
        ),
        "control": "e3.select wrapper captures the candidate list select() received",
        "measured": float(sum(r["n_align_mismatch_b"] + r["n_freshness_disagreement_b"] for r in rows)) if rows else 0.0,
        "threshold": 0.0,
        "direction": "upper",
        "comparator": "<=",
    })
    return checks


def build_headroom_checks(rows: List[Dict[str, Any]],
                          instrument: Dict[str, Any]) -> Dict[str, Any]:
    """H1 (instrument range, P0) and H2 (suppression room, post-run)."""
    out: Dict[str, Any] = {"h1": None, "h2": None, "h2_reason": ""}

    inst_vals = _finite(list(instrument["control_values"]))
    if len(inst_vals) == 2:
        out["h1"] = dv_headroom_check(
            "dv_headroom_instrument_range",
            dv_name=DV_NAME,
            criterion_threshold=THRESH_C1_RESIDUE_GAP,
            control_values=inst_vals,
            statistic="range",
            margin=DV_HEADROOM_MARGIN_H1,
            description=(
                "can p_approach move at all on this instrument? RANGE between the "
                "as-is and the approach-penalised readout over the SAME recorded "
                "scores of a fresh agent's Context-B decision points, measured "
                "BEFORE any scored cell runs"
            ),
            control="approach candidates penalised by +score_range at the selection layer",
        )

    erased_vals = _finite([r["p_approach_b"] for r in _by(rows, ARM_ERASED)])
    if len(erased_vals) >= 1:
        out["h2"] = dv_headroom_check(
            "dv_headroom_suppression_room",
            dv_name=DV_NAME,
            criterion_threshold=THRESH_C1_RESIDUE_GAP,
            # MEAN-MATCHED (validate_experiments dv_headroom-statistic-mismatch
            # lint, V3-EXQ-972a shape): C1 is the MEAN over seeds of paired
            # deltas, so the achievable suppression is mean(ERASED) - 0.0, not
            # the min-based order statistic. The min-based figure is recorded
            # as erased_floor_headroom_min_based in dv_headroom_gate.
            achievable=float(np.mean(erased_vals) - DV_BOUNDS[0]),
            margin=DV_HEADROOM_MARGIN,
            description=(
                "is there arithmetic ROOM BELOW the control arm for the registered "
                "suppression? C1 requires EXPOSED to sit at least "
                f"{THRESH_C1_RESIDUE_GAP} below {DV_HEADROOM_CONTROL_ARM} on the "
                "MEAN over seeds, so the quantity that matters is "
                "mean_over_seeds(p_approach_ERASED) - 0.0. Falsified from this "
                "run's own C1 delta at emit time (dv_headroom_observation_check)"
            ),
            control=(
                "the residue-erased twin of the EXPOSED agent -- identical in every "
                "other persistent state, so its p_approach is the no-residue "
                "baseline the criterion is measured against"
            ),
            control_arm=DV_HEADROOM_CONTROL_ARM,
        )
    else:
        out["h2_reason"] = (
            f"no finite {DV_HEADROOM_CONTROL_ARM} p_approach to measure "
            f"floor headroom against"
        )
    return out


def analyse(rows: List[Dict[str, Any]], seeds: List[int]) -> Dict[str, Any]:
    by_arm_seed = {(r["arm_id"], r["seed"]): r for r in rows}
    c1_deltas: List[float] = []
    c2_deltas: List[float] = []
    per_seed: List[Dict[str, Any]] = []
    for s in seeds:
        e = by_arm_seed.get((ARM_EXPOSED, s))
        n = by_arm_seed.get((ARM_NAIVE, s))
        x = by_arm_seed.get((ARM_ERASED, s))
        if not (e and n and x):
            continue
        ok = all(r["p_approach_b"] == r["p_approach_b"] for r in (e, n, x))
        d1 = (x["p_approach_b"] - e["p_approach_b"]) if ok else float("nan")
        d2 = (n["p_approach_b"] - e["p_approach_b"]) if ok else float("nan")
        if ok:
            c1_deltas.append(d1)
            c2_deltas.append(d2)
        per_seed.append({
            "seed": int(s),
            "p_approach_exposed": e["p_approach_b"],
            "p_approach_naive": n["p_approach_b"],
            "p_approach_erased": x["p_approach_b"],
            "c1_delta_erased_minus_exposed": d1,
            "c2_delta_naive_minus_exposed": d2,
            "n_scored_min": int(min(e["n_scored_b"], n["n_scored_b"], x["n_scored_b"])),
            # executed-action DIAGNOSTIC (the 09-04 draft's DV), for comparison
            "approach_rate_exec_exposed": e["approach_rate_b"],
            "approach_rate_exec_naive": n["approach_rate_b"],
            "approach_rate_exec_erased": x["approach_rate_b"],
            "candidate_approach_share_exposed": e["candidate_approach_share_mean_b"],
            "candidate_approach_share_erased": x["candidate_approach_share_mean_b"],
            "candidate_approach_share_naive": n["candidate_approach_share_mean_b"],
            "residue_term_spread_exposed": e["residue_term_spread_mean_b"],
            "residue_term_spread_erased": x["residue_term_spread_mean_b"],
            "residue_term_spread_naive": n["residue_term_spread_mean_b"],
        })

    def _mean(v: List[float]) -> float:
        return float(statistics.fmean(v)) if v else float("nan")

    c1_mean = _mean(c1_deltas)
    c2_mean = _mean(c2_deltas)
    c1_sd = float(statistics.pstdev(c1_deltas)) if len(c1_deltas) > 1 else 0.0
    effect_sd = (
        float(c1_mean / c1_sd) if c1_sd > 1e-12
        else (float("inf") if c1_mean == c1_mean and c1_mean > 0 else 0.0)
    )
    n_seeds_positive = sum(1 for d in c1_deltas if d > 0)
    min_decision_points = min((r["n_decision_points_b"] for r in rows), default=0)
    min_scored = min((r["n_scored_b"] for r in rows), default=0)

    c1 = bool(c1_mean == c1_mean and c1_mean >= THRESH_C1_RESIDUE_GAP)
    c2 = bool(c2_mean == c2_mean and c2_mean >= THRESH_C2_OVERALL_GAP)
    c3 = bool(effect_sd == effect_sd and effect_sd >= THRESH_C3_EFFECT_SD)
    c4 = bool(n_seeds_positive >= THRESH_C4_MIN_SEEDS)
    c5 = bool(min_scored >= THRESH_C5_MIN_SCORED_TICKS)

    return {
        "per_seed": per_seed,
        "mean_p_approach_exposed": _mean(
            _finite([r["p_approach_b"] for r in _by(rows, ARM_EXPOSED)])),
        "mean_p_approach_naive": _mean(
            _finite([r["p_approach_b"] for r in _by(rows, ARM_NAIVE)])),
        "mean_p_approach_erased": _mean(
            _finite([r["p_approach_b"] for r in _by(rows, ARM_ERASED)])),
        "mean_approach_rate_exec_exposed": _mean(
            _finite([r["approach_rate_b"] for r in _by(rows, ARM_EXPOSED)])),
        "mean_approach_rate_exec_naive": _mean(
            _finite([r["approach_rate_b"] for r in _by(rows, ARM_NAIVE)])),
        "mean_approach_rate_exec_erased": _mean(
            _finite([r["approach_rate_b"] for r in _by(rows, ARM_ERASED)])),
        "min_scored_ticks": int(min_scored),
        "c1_residue_gap_mean": c1_mean,
        "c2_overall_gap_mean": c2_mean,
        "c1_delta_sd": c1_sd,
        "effect_sd": effect_sd,
        "n_seeds_c1_positive": int(n_seeds_positive),
        "n_paired_seeds": len(c1_deltas),
        "min_decision_points": int(min_decision_points),
        "c1_residue_gap_pass": c1,
        "c2_overall_gap_pass": c2,
        "c3_effect_sd_pass": c3,
        "c4_seed_consistency_pass": c4,
        "c5_data_quality_pass": c5,
        # C1 is the LOAD-BEARING criterion; C3/C4/C5 qualify it. C2 is reported
        # alongside and drives the middle branch of the interpretation grid.
        "residue_attributable_pass": bool(c1 and c3 and c4 and c5),
    }


# ------------------------------------------------------------------ #
# Main                                                               #
# ------------------------------------------------------------------ #
def run_experiment(dry_run: bool) -> Dict[str, Any]:
    t0 = time.perf_counter()
    seeds = SEEDS[:2] if dry_run else SEEDS

    # ---- design-time audit: C1 must lie inside the DV's arithmetic room ----
    required = THRESH_C1_RESIDUE_GAP * DV_HEADROOM_MARGIN
    if not (0.0 < required < (DV_BOUNDS[1] - DV_BOUNDS[0])):
        raise ValueError(
            f"C1 requires headroom {required} but p_approach is bounded in "
            f"{DV_BOUNDS}: the criterion is unsatisfiable by arithmetic."
        )
    print(
        f"[{QUEUE_ID}] design-audit OK: C1 headroom required {required:.4g} lies "
        f"inside the DV's bounds {DV_BOUNDS}",
        flush=True,
    )

    # ---- H1 / P4: instrument positive control, BEFORE any scored cell ------
    instrument = instrument_positive_control(dry_run=dry_run)
    print(
        f"[{QUEUE_ID}] instrument score-sensitivity control: "
        f"p_approach_as_is={instrument['p_approach_as_is']:.4f} "
        f"approach_penalised={instrument['p_approach_approach_penalised']:.4f} "
        f"separation={instrument['separation']:.4f} fraction_removed={instrument['fraction_removed']:.4f} "
        f"n_scored={instrument['n_scored']} "
        f"(floor {DV_INSTRUMENT_SCORE_SENSITIVITY_FLOOR} on the fraction, met={instrument['met']})",
        flush=True,
    )

    rows: List[Dict[str, Any]] = []
    for s in seeds:
        rows.extend(run_seed(s, dry_run=dry_run))

    headroom = build_headroom_checks(rows, instrument)
    precondition_checks = build_preconditions(rows, instrument)
    gate_checks = list(precondition_checks)
    for key in ("h1", "h2"):
        if headroom[key] is not None:
            gate_checks.append(headroom[key])

    try:
        preconditions = p0_readiness_gate(gate_checks)
        gate_green = True
        gate_reason = ""
    except P0NotReady as exc:
        preconditions = list(exc.preconditions)
        gate_green = False
        gate_reason = str(exc.reason)
    if headroom["h2"] is None:
        gate_green = False
        gate_reason = (gate_reason + " | " + headroom["h2_reason"]).strip(" |")

    analysis = analyse(rows, list(seeds))

    # Falsify H2's mean-matched ceiling from the run's OWN C1 delta (the same
    # statistic C1 reads, in its orientation): an observed suppression above the
    # asserted room refutes the ceiling outright.
    for _p in preconditions:
        if _p.get("name") == "dv_headroom_suppression_room":
            dv_headroom_observation_check(
                _p, [analysis["c1_residue_gap_mean"]], observed_name="c1_residue_gap_mean")

    # ---- interpretation grid ----------------------------------------------
    if not gate_green:
        outcome = "FAIL"
        label = "substrate_not_ready_requeue"
        direction = "non_contributory"
        per_claim = {c: "non_contributory" for c in CLAIM_IDS}
        text = (
            "SUBSTRATE/INSTRUMENT NOT READY -- not a verdict on EXT-004 or "
            "ARC-013. " + gate_reason
        )
    elif analysis["residue_attributable_pass"]:
        outcome = "PASS"
        label = "residue_carries_cross_context_action_suppression"
        direction = "supports"
        per_claim = {"EXT-004": "supports", "ARC-013": "supports"}
        text = (
            "EXT-004 and ARC-013 SUPPORTED: an agent that incurred real harm in "
            "Context A put "
            f"{analysis['c1_residue_gap_mean']*100:.1f} percentage points less "
            "pre-commit selection mass on hazard-approaching candidates than its "
            "OWN residue-erased twin at decision points in a structurally novel "
            "Context B, "
            "with every other persistent cross-episode state (ARC-108 w_chan / "
            "V-hat_t, the MECH-165 exploration buffer, E1/E2 weights) held "
            "bit-identical by construction. The residue field is the carrier."
        )
    elif analysis["c2_overall_gap_pass"]:
        outcome = "FAIL"
        label = "cross_context_suppression_present_but_not_residue_attributable"
        direction = "mixed"
        per_claim = {"EXT-004": "supports", "ARC-013": "weakens"}
        text = (
            "SPLIT RESULT. EXT-004's claim is SUPPORTED at the selection layer: the "
            "harm-exposed agent put "
            f"{analysis['c2_overall_gap_mean']*100:.1f} percentage points less "
            "pre-commit mass on hazard-approaching candidates than the harm-naive "
            "agent in a novel context. But ARC-013 is "
            "WEAKENED: the residue-erased twin, identical in every other "
            "persistent state, showed only a "
            f"{analysis['c1_residue_gap_mean']*100:.1f} percentage-point "
            "difference from EXPOSED, below the "
            f"{THRESH_C1_RESIDUE_GAP*100:.0f}pp bar -- so removing the residue "
            "field did NOT remove the effect and residue is not the carrier. "
            "Route the follow-up to ARC-108 (w_chan / V-hat_t) and MECH-165 "
            "(exploration buffer), the two uncontrolled carriers V3-EXQ-991's own "
            "attribution_caveat named."
        )
    else:
        outcome = "FAIL"
        label = "no_cross_context_action_level_suppression"
        direction = "weakens"
        per_claim = {"EXT-004": "weakens", "ARC-013": "weakens"}
        text = (
            "EXT-004 and ARC-013 WEAKENED: with a demonstrably score-sensitive, "
            "action-level selection readout (instrument control moved p_approach by "
            f"{instrument['separation']:.3f} under an approach-candidate penalty) and "
            "both headroom gates met, an agent that incurred real harm in Context A "
            "did NOT down-weight hazard-approaching candidates in a novel Context B "
            "relative to either the "
            f"harm-naive agent (gap {analysis['c2_overall_gap_mean']*100:.1f}pp) or "
            "its own residue-erased twin (gap "
            f"{analysis['c1_residue_gap_mean']*100:.1f}pp). With V3-EXQ-991's weak "
            "null this is the second null on this prediction and the FIRST on a "
            "construct-valid, action-level DV with measured free-policy range -- "
            "V3-EXQ-991's harm_rate measured hazard-field occupancy, and the "
            "2026-09-04 draft's executed approach_rate was floored by a locked "
            "policy. Scope: this DV (scoring channel), this context pair, this "
            "exposure budget."
        )

    criteria = [
        {"name": "C1_residue_isolated_gap", "load_bearing": True,
         "passed": analysis["c1_residue_gap_pass"]},
        {"name": "C2_overall_behavioural_gap", "load_bearing": False,
         "passed": analysis["c2_overall_gap_pass"]},
        {"name": "C3_effect_sd", "load_bearing": False,
         "passed": analysis["c3_effect_sd_pass"]},
        {"name": "C4_seed_consistency", "load_bearing": False,
         "passed": analysis["c4_seed_consistency_pass"]},
        {"name": "C5_data_quality", "load_bearing": False,
         "passed": analysis["c5_data_quality_pass"]},
    ]
    criteria_non_degenerate = {c["name"]: bool(gate_green) for c in criteria}

    naive_rows = _by(rows, ARM_NAIVE)
    naive_diagnostic = {
        "note": (
            "RECORDED, NOT GATED. V3-EXQ-991 used this as a non-vacuity gate "
            "(`naive_control_clean`), but it is constant-TRUE by construction -- "
            "the NAIVE Context A zeroes hazard_harm, harm_gradient_scale, "
            "contamination_spread AND proximity_harm_scale, so no harm source "
            "remains and the detector can never fail. Keeping it as a gate made "
            "the non_contributory branch structurally unreachable; keeping it as a "
            "diagnostic preserves the check without inflating the gate."
        ),
        "max_context_a_harm_events": (
            int(max(r["context_a_n_harm_events"] for r in naive_rows))
            if naive_rows else None
        ),
        "max_residue_mean_weight_at_b_entry": (
            float(max(r["residue_mean_weight_at_b_entry"] for r in naive_rows))
            if naive_rows else None
        ),
    }

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    manifest: Dict[str, Any] = {
        "run_id": f"{EXPERIMENT_TYPE}_{ts}_v3",
        "queue_id": QUEUE_ID,
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": ARCHITECTURE_EPOCH,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "backlog_id": BACKLOG_ID,
        "supersedes": SUPERSEDES,
        "outcome": outcome,
        "timestamp_utc": ts,
        "evidence_direction": direction,
        "evidence_direction_per_claim": per_claim,
        "non_degenerate": bool(gate_green),
        "degeneracy_reason": "" if gate_green else gate_reason,
        "interpretation": {
            "label": label,
            "text": text,
            "method": "three_arm_exposed_naive_erased_precommit_p_approach_cross_context",
            "preconditions": preconditions,
            "criteria_non_degenerate": criteria_non_degenerate,
        },
        "criteria": criteria,
        "combination_rule": (
            "load-bearing C1 (residue-isolated gap) AND C3 AND C4 AND C5 -> PASS / "
            "supports both claims. C1 false but C2 true -> the split branch: "
            "EXT-004 supports, ARC-013 weakens. C1 and C2 both false -> weakens "
            "both. Any precondition or headroom entry unmet -> "
            "substrate_not_ready_requeue / non_contributory for both, regardless "
            "of the criteria."
        ),
        "analysis": analysis,
        "arm_results": rows,
        "instrument_positive_control": instrument,
        "dv_headroom_gate": {
            "h1_instrument_range": headroom["h1"],
            "h2_suppression_room": headroom["h2"],
            "h2_reason": headroom["h2_reason"],
            "control_arm": DV_HEADROOM_CONTROL_ARM,
            "margin": DV_HEADROOM_MARGIN,
            "criterion_threshold": THRESH_C1_RESIDUE_GAP,
            "erased_floor_headroom_min_based": (
                float(dv_achievable(
                    _finite([r["p_approach_b"] for r in _by(rows, ARM_ERASED)]),
                    statistic="floor_headroom", dv_bounds=DV_BOUNDS))
                if _finite([r["p_approach_b"] for r in _by(rows, ARM_ERASED)])
                else None
            ),
            "erased_floor_headroom_mean_matched": (
                float(np.mean(_finite([r["p_approach_b"] for r in _by(rows, ARM_ERASED)])) - DV_BOUNDS[0])
                if _finite([r["p_approach_b"] for r in _by(rows, ARM_ERASED)])
                else None
            ),
        },
        "readout_note": (
            "DV = p_approach: softmax(-scores/T_eff) mass on approach candidates "
            "over approach+retreat candidates, read only on fresh E3 selections at "
            "decision points with both directions present (2026-09-09 redesign; "
            "free-policy range measured before the bar was set -- see docstring). "
            "approach_rate_b is the 09-04 draft's executed-action DV, recorded as a "
            "diagnostic with a latch-replicated denominator."
        ),
        "naive_residue_empty_diagnostic": naive_diagnostic,
        "predecessor_weak_null": {
            "run_id": PREDECESSOR_RUN_ID,
            "queue_id": SUPERSEDES,
            "adjudicated": "non_contributory (construct-validity grounds)",
            "dv_was": "harm_rate_B (mean harm magnitude per Context-B step)",
            "dv_span_across_cells": [0.0024047619435522295, 0.14226806797087194],
            "n_seeds_no_transfer": 4,
            "n_seeds": 5,
            "mean_rel_ratio_exposed_over_naive": 1.1802665357349584,
            "note": (
                "carried as a prior, not erased: the DV was NOT pinned (59x across "
                "cells) and 4/5 seeds showed no transfer, but the DV measured "
                "hazard-field occupancy rather than action-level suppression, and "
                "the design isolated no channel. A null HERE, on an action-level "
                "DV with a residue-only control arm, is a second and stronger null."
            ),
        },
        "registered_thresholds": {
            "C1_residue_gap": THRESH_C1_RESIDUE_GAP,
            "C2_overall_gap": THRESH_C2_OVERALL_GAP,
            "C3_effect_sd": THRESH_C3_EFFECT_SD,
            "C4_min_seeds": THRESH_C4_MIN_SEEDS,
            "C5_min_scored_ticks": THRESH_C5_MIN_SCORED_TICKS,
            "grad_eps": GRAD_EPS,
            "dv_instrument_score_sensitivity_floor": DV_INSTRUMENT_SCORE_SENSITIVITY_FLOOR,
            "dv_headroom_margin_h1": DV_HEADROOM_MARGIN_H1,
            "residue_term_spread_floor": RESIDUE_TERM_SPREAD_FLOOR,
            "dv_headroom_margin": DV_HEADROOM_MARGIN,
            "dv_name": DV_NAME,
        },
    }

    full_config = {
        "seeds": list(seeds),
        "arms": list(ARMS),
        "rho_residue": RHO_RESIDUE,
        "alpha_world": 0.9,
        "p0_context_a_episodes": 2 if dry_run else P0_CONTEXT_A_EPISODES,
        "eval_context_b_episodes": 2 if dry_run else EVAL_CONTEXT_B_EPISODES,
        "steps_per_episode": 20 if dry_run else STEPS_PER_EPISODE,
        "context_a_env_kwargs_by_arm": CONTEXT_A_ENV_KWARGS_BY_ARM,
        "context_b_env_kwargs": CONTEXT_B_ENV_KWARGS,
        "grad_eps": GRAD_EPS,
        "e3_score_decomp_enabled": True,
        "dv": "p_approach",
        "supersedes": SUPERSEDES,
        "thresholds": manifest["registered_thresholds"],
    }
    manifest["config"] = full_config
    manifest["_started_at"] = t0
    return manifest


if __name__ == "__main__":
    _t0 = time.perf_counter()
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    print(f"{QUEUE_ID}: EXT-004 action-level cross-context suppression probe",
          flush=True)
    print(f"  dry_run={args.dry_run}", flush=True)

    manifest = run_experiment(args.dry_run)
    manifest.pop("_started_at", None)

    out_path = write_flat_manifest(
        manifest,
        OUT_DIR,
        dry_run=bool(args.dry_run),
        config=manifest.get("config"),
        seeds=list(manifest["config"]["seeds"]),
        script_path=Path(__file__),
        z_goal_stream_stats=_ZG.stats(),
        started_at=_t0,
    )

    a = manifest["analysis"]
    print(f"\n[{QUEUE_ID}] Results", flush=True)
    print(
        f"  p_approach  exposed={a['mean_p_approach_exposed']:.4f}"
        f"  erased={a['mean_p_approach_erased']:.4f}"
        f"  naive={a['mean_p_approach_naive']:.4f}"
        f"  (min_scored_ticks={a['min_scored_ticks']})",
        flush=True,
    )
    print(
        f"  C1 residue gap={a['c1_residue_gap_mean']:.4f} "
        f"(bar {THRESH_C1_RESIDUE_GAP})  "
        f"C2 overall gap={a['c2_overall_gap_mean']:.4f}",
        flush=True,
    )
    print(
        f"  non_degenerate={manifest['non_degenerate']} outcome={manifest['outcome']} "
        f"label={manifest['interpretation']['label']}",
        flush=True,
    )
    print(f"  manifest -> {out_path}", flush=True)

    _outcome_raw = str(manifest["outcome"]).upper()
    emit_outcome(
        outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=str(out_path),
        dry_run=args.dry_run,
    )
