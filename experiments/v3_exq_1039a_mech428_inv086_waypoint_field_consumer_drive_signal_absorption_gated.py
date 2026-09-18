#!/opt/local/bin/python3
"""V3-EXQ-1039a -- waypoint_field_consumer_reach H1 (drive axis), ABSORPTION-GATED re-run.

SUPERSEDES V3-EXQ-1039. Same scientific question, same registered leg
(REE_assembly/evidence/planning/hypothesis_space_registry.v1.json, qid
`waypoint_field_consumer_reach`, hid `H-wpfield-objective-sparsity`) -- an INSTRUMENT
REPAIR, not a new hypothesis, so an alphabetic suffix rather than a new number.

AUTHORITY. The CONFIRMED artifact
`REE_assembly/evidence/planning/failure_autopsy_V3-EXQ-1039_2026-09-16.md` (+ `.json`),
user Step 8 gate 2026-09-16T13:06:22Z, applied by governance-20260916 (REE_assembly
db6d20ebee). Section 14 "Routing / Primary" items 1-5 are the design spec below; every
threshold the gate itself decided is carried verbatim (see RATIFIED CHOICES).

=== WHAT 1039 MEASURED, AND WHY IT ANSWERED NOTHING (established; not re-derived here) ===

V3-EXQ-1039 returned PASS with label `training_signal_does_not_convert_h1_not_supported`.
The autopsy established that verdict is a VACUOUS null: `shaped_rl` was BIT-IDENTICAL to
`sparse_rl` on every recorded behavioural metric on all 3 seeds while carrying a training
return two-plus orders of magnitude larger; `demo_warmstart`'s final CE loss was 3.465736 =
ln(32), i.e. exactly chance over `num_candidates=32`, the signature of an exactly-zero
gradient; the readiness gate built to catch precisely this (C0d) PASSED at 0.667 while the
head's output stayed at chance, because it tested a BOOLEAN `|weight_norm_after -
weight_norm_before| > 1e-9` plus `LateralPFCAnalog.candidate_summary_degenerate` -- a field
that is only COMPUTED when `rule_readout_consumer` is on and `k >= 2`
(`ree_core/pfc/lateral_pfc_analog.py` ~:429-450), and `rule_readout_consumer` defaults
False on this recipe, so it read its `__init__` value. `z_goal` ticks_active was 0 of 94434.

Neither treatment manipulation is shown to have reached the consumer's behaviour-determining
parameters. So the H1 (drive-axis) leg is UNRESOLVED, not answered, and the whole content of
this re-run is: make the two things the 1039 instrument could not distinguish --
"the manipulation was ABSORBED (it never reached a parameter)" versus "it COULD have
converted into different committed behaviour and did not" -- separable BEFORE any flat null
is admissible as evidence about reward density.

=== WHAT CHANGED FROM 1039 (exhaustive; everything else is carried verbatim) ===

(A) CONSUMES THE SHARED ABSORPTION/CONVERSION TELEMETRY BUILD, rather than instrumenting
    driver-side. The substrate entry `sd-allon-training-signal-absorption-telemetry` that
    the 1039 autopsy routed has since LANDED in `experiments/_lib/allon_training.py`
    (ree-v3 996dec30) -- it did not exist when this chip's 2026-09-16 pre-flight authorised
    driver-side instrumentation. `_train_all_on_agent` now returns `absorption` and
    `conversion` blocks; this driver CAPTURES the return value of each of its two
    `_train_all_on_agent` calls per cell (the P0 warm call and the arm's own P1 RL call),
    which is what makes the autopsy's "recorded PER ARM AND PER PHASE" literal rather than
    aspirational -- 1039 discarded both return values. No absorption or conversion statistic
    is recomputed here that the shared build already computes.

(B) ONE DRIVER-SIDE ADDITION, and only one: a per-tick histogram of distinct first-action
    classes (`_FirstActionClassTap`). The shared build records
    `conversion.distinct_first_action_classes_{n,mean,sd,min,max}`; the RATIFIED gate is
    ">= 3 on a MAJORITY of ticks", which is a median/quantile statement and is NOT
    recoverable from n/mean/sd/min/max. The tap wraps
    `allon_training._distinct_first_action_classes` -- the shared build's own helper, whose
    return value it records and passes through unchanged -- and is restored in a `finally`.
    It performs no computation the recipe did not already perform, draws no RNG, and cannot
    change any arm's behaviour. This is the ONLY place this driver adds telemetry the shared
    build does not already own, and it is named here so it is not mistaken for duplication.

(C) THE 1039 C0d GATE IS REPLACED BY THREE SEPARATE, NUMERIC PRECONDITIONS (autopsy items
    (c) and (d)): C0d records the lateral-PFC bias head's last-linear weight-norm DELTA as a
    FLOAT on the WORST cell (not a boolean, not a majority-fraction); C0e routes on the
    shared build's `conversion.candidate_summary_degenerate_frac`, which is COMPUTED from the
    summaries the head actually consumed and is therefore live on this recipe regardless of
    `rule_readout_consumer` (`conversion.candidate_summary_degenerate_computed: True` marks
    it); and `hidden_dead_relu_frac` is recorded (this driver sets
    `lateral_pfc_capture_head_diagnostics=True`, which `lateral_pfc_analog.py:454-472`
    documents and
    this driver's author verified at source to be a layer-by-layer unroll of the SAME
    `Sequential` ops -- bit-identical `bias_raw`, no RNG draw, no parameter write).

(D) THE PAIRED READINESS GATE the Step 8 user gate ratified: ABSORPTION (A1/A2/A3) AND
    CONVERSION-CAPABILITY (B1/B2). Autopsy section 14: "Items 1 and 2 are a pair: shipping 1
    without 2 reproduces the defect this artifact diagnoses." Every one of the five
    self-routes `substrate_not_ready_requeue` on failure -- never a substrate-verdict label.

(E) `criteria_non_degenerate` for C1 is now ABSORPTION and CONVERSION-CAPABILITY and EVAL
    MOBILITY (autopsy item 4), not mobility alone as in 1039.

(F) THE BUDGET IS UNCHANGED (autopsy item 5: "Reconsider the 90-episode budget only after
    (1)-(4)"). `ZWORLD_P0_EPISODES=20`, `P0_WARMUP_EPISODES=25`, `RL_EPISODES=90`,
    `N_DEMO_EPISODES=25`, `STEPS_PER_EPISODE=80`, `SEEDS=[42,43,44]` -- byte-identical to
    1039. This run is not a power-bump and must not be read as one.

NOT CHANGED, DELIBERATELY: `_ShapedWaypointEnv` is carried verbatim, including the fact that
`_dist_prev` is not reset on waypoint respawn, so a stationary policy still accrues ~1.5
shaped bonus per episode (autopsy section 3a). The autopsy names that as the REASON
absorption alone is insufficient and answers it with the conversion-capability gate (B1/B2);
it does NOT route changing the manipulation. Altering the wrapper would change what this leg
measures and would make 1039a a different experiment rather than a repair of this one.

=== RATIFIED CHOICES (from the Step 8 gate outcome, 2026-09-16T13:06:22Z) ===

  * first-action-diversity floor `>= 3` distinct classes (NOT MECH-341's stricter 4.5). The
    gate asked this explicitly (autopsy section 16 item 3) and answered 3. `>= 3` sits
    between the registered 2-class clamp (ARC-065 SP-CEM pool
    `floor_clamped_at_2_distinct_first_action_classes`) and MECH-341's 4.5.
  * the eval-side bias-head ABLATION sensitivity check (B2) is part of the ratified gate.
  * the substrate entry stays severity `degrading` (not carved into a narrow `corrupting`
    sub-class).

AND ONE FURTHER RATIFIED AMENDMENT, 2026-09-18 (AskUserQuestion, option B), raised by this
driver's own Step 4.5 red-team and decided by the user before queuing:

  * **precondition A2 is quantified over `A2_GATED_ARMS` (`shaped_rl`), not over every
    treatment cell as the autopsy's section 14 item 1 literally said.** The unshaped arms'
    surviving-advantage fractions are RECORDED as telemetry rather than gated. Full
    reasoning at `A2_GATED_ARMS`; in one line: a BLOCKED `sparse_rl` legitimately shows no
    surviving advantage (the EMA baseline decays below the threshold in a reward-free
    stretch), so the original quantifier would have self-routed the single most likely
    H1-negative result to `substrate_not_ready_requeue` -- an instrument label on the
    manipulation's own control regime.
  * `lpfc_bias_saturated_frac` stays RECORDED, NOT GATED (option C declined), and
    `e3.e3_score_decomp_enabled` stays OFF (option D declined, deferred to its own
    `/implement-substrate` pass).

=== THE FIVE READINESS PRECONDITION SETS, AND WHAT EACH CAN FAIL ON ===

ABSORPTION -- "did the manipulation reach a parameter?" (autopsy item 1)
  A1  `shaped_vs_sparse_return_separation`. Per seed, `shaped_rl`'s P1 mean episode return
      must exceed `sparse_rl`'s by at least `RETURN_SEPARATION_SD` standard deviations of
      `sparse_rl`'s OWN per-episode return distribution. Self-calibrating to this run's own
      noise scale rather than importing an absolute literal -- the same discipline 1039 used
      for `LIFT_MARGIN_FRAC`, and for the same reason (no piloted effect size exists at this
      budget). WORST SEED is reported, not the mean. Applies to the shaped-vs-sparse PAIR
      only: `demo_warmstart` runs the same unshaped env as `sparse_rl` by construction, so
      its returns are expected to match and its absorption evidence is A3 instead.
  A2  `adv_surviving_frac_clears_floor`. The fraction of sampled REINFORCE outcome-buffer
      terms clearing `abs(adv) >= ADV_MIN_THRESHOLD` (0.005), per treatment cell, WORST
      CELL. This is the key absorption number: ~0 means the run received no gradient at all,
      whatever its reward was. `None` (no term ever drawn) is scored 0.0, never skipped.
  A3  `demo_warmstart_ce_below_chance`. `demo_warmstart`'s phase-A CE, averaged over the
      final tenth of its updates, must fall to at most `DEMO_CE_CHANCE_FRAC` of
      `ln(mean n_candidates)` measured in that same loop. 1039's 3.465736 = ln(32) is the
      exact failure this catches. WORST SEED (max CE).

CONVERSION CAPABILITY -- "could it have converted into different committed behaviour?"
(autopsy item 2; equally load-bearing, "shipping 1 without 2 reproduces the defect")
  B1  `first_action_diversity_majority_ticks`. Distinct first-action classes per E3 tick
      must be `>= FIRST_ACTION_CLASS_FLOOR` (3) on strictly more than half of the recorded
      ticks, in EVERY treatment cell. WORST CELL. Recorded per arm and per phase: the P0
      warm call legitimately records ZERO ticks, because the shared build's conversion block
      is `if is_p1:`-gated at source, so there is nothing to gate on there and this is
      reported as `n_ticks: 0` rather than as a measured zero.
  B2  `bias_ablation_changes_committed_action`. Each trained agent is evaluated TWICE from a
      bit-identical RNG and env state -- lateral-PFC bias head TRAINED, then ZEROED -- and at
      least one committed action must differ, on at least `ABLATION_MIN_SEEDS` (2) of 3 seeds,
      in EVERY treatment arm. WORST ARM. Under proposer-pool collapse a fully absorbed
      manipulation still yields identical eval behaviour, so A1-A3 green alone would emit the
      same vacuous null behind a clean gate (autopsy section 14 item 2 "Why").
      This asserts only that two action sequences DIFFER on one machine within one process --
      never an exact committed action, which `torch.multinomial` makes non-portable across
      machine classes (CLAUDE.md "Running the test suite").

CARRIED FROM 1039, UNCHANGED: C0 (greedy_oracle clears ORACLE_FLOOR), C0b (SD-070 P0a warmup
moved the world encoder on a majority of cells), C0c (the oracle-random achievable span
clears MIN_ACHIEVABLE_SPAN), R1 (sparse_rl stays blocked -- the premise), R2 (treatment arms
are not a stationary argmax-into-wall eval policy).

ALL TEN readiness/premise gates are CONJUNCTIVE and precede any verdict criterion. That is
the point, not an oversight: the autopsy's finding is that 1039's single non-conjunctive
readiness gate is exactly what let a vacuous null through. A `substrate_not_ready_requeue`
self-route naming WHICH gate failed is an informative outcome of this design, not a wasted
run -- it is the discrimination 1039 could not make.

=== EVERYTHING BELOW HERE IS CARRIED FROM V3-EXQ-1039 ===

QUESTION. V3-EXQ-1004 established that the SD-WAYPOINT-FIELD observable makes the pending
waypoint's direction decodable and BEHAVIOURALLY SUFFICIENT FOR A SUPERVISED READER cloned
from an oracle (0.573 -> 0.839 BC-imitation accuracy; 0.35 -> 58.6 visits/ep). That reader is
NOT a REE consumer. This run asks the orthogonal drive-axis question the 1004 autopsy fanned
out: for an ACTUAL REEAgent consumer (z_world via the standard latent stack; action selection
via the real E1-candidate-generation -> E3-selection loop, `agent.generate_trajectories` +
`agent.select_action`, exactly the `_train_all_on_agent` "ree_trained_allon" recipe already
used by V3-EXQ-724/734/737/742), holding the field ON and the consumer fixed, does the
TRAINING SIGNAL determine whether perceivability converts into navigation?

DECLARED NULL (H1, from the registry). visits/ep is FLAT across training signals -- i.e. a
denser (shaped) or warm-started signal does NOT lift the consumer off the sparse-reward
floor. Meeting this null supports "sparsity is not the residual blocker". A CLEARED lift
under either dense arm, with the sparse arm still blocked, supports H1.
**In 1039a that null is admissible ONLY behind the paired readiness gate above.**

CAVEAT CARRIED FORWARD. V3-EXQ-1004's docstring reports an A2C reader at 0.00 (field OFF) /
0.10 (field ON) visits/ep after 400 episodes on seed 42 only -- that figure is DOCSTRING-ONLY
(never committed, no manifest, no per-episode trace) and was measured BEFORE the SD-094
contamination gate, on a self-contaminating env. It MOTIVATES this leg's existence; it is not
cited below as a baseline, a bar, or a comparison point, and no threshold in this script is
derived from it.

WHAT THIS IS **NOT**, STATED PLAINLY. The per-tick REWARD-DRIVEN learning in
`_train_all_on_agent`'s P1 phase does NOT train E3's own channel-scoring heads: it trains the
LATERAL-PFC BIAS head and the OFC DEVALUATION-BIAS head (both of which modulate E3's
competitive scoring downstream) through a two-head REINFORCE update keyed to episode return.
E3's own scoring-head parameters are in neither optimiser's parameter list and receive no
gradient from this training signal. So this run exercises the REAL E1-generate -> E3-select
decision loop on every tick and REAL reward-contingent learning through the modulatory heads
that bias that loop, but does not itself train E3's core scoring network. It does not license
"E3's selection mechanism was trained on this reward" as a description of what happened.

THE THREE TRAINING-SIGNAL ARMS (field ON, consumer construction IDENTICAL across arms)
  sparse_rl       stock env reward only. REINFORCE via
                  `_train_all_on_agent(p0_episodes=0, p1_episodes=RL_EPISODES)`, UNMODIFIED.
  shaped_rl       IDENTICAL call, on the SAME env wrapped in `_ShapedWaypointEnv`: every
                  `step()` reward augmented with a MONOTONE, NON-NEGATIVE per-step progress
                  bonus toward the pending waypoint. Only the reward channel is touched.
                  NOT Ng-Harada-Russell potential-DIFFERENCE shaping, deliberately -- see
                  `_ShapedWaypointEnv`'s own docstring (1039 Step 4.5 red-team F1).
  demo_warmstart  N_DEMO_EPISODES of AUXILIARY supervised warm-start on the SAME lateral-PFC
                  bias head P1 REINFORCE later trains, THEN the identical RL procedure and
                  budget as sparse_rl. The warm-start never overrides the agent's own action.

NAVIGATION ISOLATION (verbatim rationale from V3-EXQ-1004). The TRAINING/EVAL env has
num_hazards=0, num_resources=0, energy_decay=0.0, hazard_free_contamination_gate=True
(SD-094). The waypoint field stays ON in every arm.

WHY A SEPARATE, RESOURCE-BEARING WARM ENV. The SD-070 P0a encoder-warmup recipe
(`run_zworld_p0`, invoked via `zworld_p0_episodes=`) trains the world encoder against a
resource-proximity target (SD-018) and needs a live resource signal. `world_obs_dim` is 275
regardless of entity counts, so the SAME agent transfers to the navigation-isolated env
without re-initialisation. This warmup is GENERIC (unrelated to waypoints).

DV-SYMMETRY INVARIANCE (queue-experiment Step 3 mandatory declaration), per arm.
  sparse_rl / shaped_rl / demo_warmstart: the DV is waypoints-visited-per-episode, a
  behavioural ROLLOUT COUNT from an independently, fully retrained policy per (seed, arm)
  cell -- not a selection over a shared externally-scored candidate set, so the
  broadcast-additive-constant symmetry has no purchase (there is no shared scored-candidate
  axis across arms for a constant to cancel on). Not a rank/order statistic derived from one
  shared scalar under a monotone transform (each arm's count comes from its own distinct
  trained policy's rollout). Not a permutation-symmetric aggregate over interchangeable seeds
  (each seed differs by construction: distinct RNG stream, distinct training trajectory,
  distinct final policy). The manipulation is not invariant under any of the three.
  greedy_oracle / random_walk: anchors, not manipulated arms; no manipulation to be invariant
  under. Their DV is the same rollout count, read for the achievable span only.
  B2's own DV (does any committed action differ under head ablation) is a SEQUENCE-INEQUALITY
  over a paired rollout, so it is invariant under nothing the ablation does not itself move:
  zeroing the head is not a broadcast constant on the score axis (the head's output is
  per-candidate and enters E3's score_bias), not a monotone rescaling, and not a permutation.

CLAIMS. INV-086 and MECH-428 are carried as READ-ACROSS ONLY -- NEITHER claim's own
`what_would_answer` regime is exercised: INV-086 needs a WM-decay regime + feedback-channel
ablation; MECH-428 needs a seeding-sparse z_goal_norm + forced-seed control. This run
instantiates neither. `evidence_direction_per_claim` is `non_contributory` for both,
unconditionally, matching 1039 and the sibling H2 leg (V3-EXQ-1030).

experiment_purpose: diagnostic -- a GOV-FANOUT-1 discrimination leg; it does not test either
claim's own hypothesis and PROMOTES/DEMOTES NOTHING. Routes to /failure-autopsy for
adjudication before any governance action.

GOV-REUSE-1 (Step 2.4): the decisive readout is the paired absorption/conversion readiness
block evaluated alongside visits/ep. Checked V3-EXQ-1039's own manifest
(`v3_exq_1039_..._20260915T025330Z_v3.json`, substrate_hash
5a74d5279d7c7753d4ccd34d281bc89d798c04162899d40db7f97ecdc1ea0884) -- it carries NONE of
`absorption`, `conversion`, `adv_surviving_frac`, `distinct_first_action_classes` or
`candidate_summary_degenerate_computed`, because the shared telemetry build landed AFTER it
ran; and the 1039 autopsy's own GOV-REUSE check against v3_exq_884 / v3_exq_1030 /
v3_exq_1004 found 0/3 carrying even the behavioural readout. Additionally the telemetry build
edits `experiments/_lib/allon_training.py`, which is inside `substrate_hash`, so no prior run
is substrate-compatible with this one by construction. NOT RECOVERABLE -> run.

ethics_preflight:
  involves_negative_valence: false
  involves_suffering_like_state: false
  involves_self_model: false
  involves_inescapability_or_helplessness: false
  involves_offline_replay_over_harm: false
  involves_social_mind_or_language: false
  involves_human_data_or_clinical_context: false
  decision: allow

SLEEP DRIVER: none (no sleep flag set by this driver; use_sleep_loop / sws_enabled /
rem_enabled / use_sleep_aggregation_cluster all OFF/default).

STEP 4.5 RED-TEAM (fable, cross-model, one pass): CONTESTED -- 5 findings, all verified
against the source before disposition. F1 (bias-head clamp saturation is a zero-gradient
route that A2 cannot see, since n_adv_surviving is incremented before compute_bias is
called) and F4 (B1's class count ranges over the full candidate pool while the head's
conversion surface is the F-eligible prefix) and F5 (the per-tick tap counted env steps,
not fresh E3 selections, so the ratified MAJORITY was commitment-hold-weighted) are
FIXED: F5 by identity-deduplicating the cached candidate object, F1/F4 by recording
lpfc_bias_abs_mean vs bias_scale, lpfc_bias_saturated_frac and modulatory_shortlist_size
per fresh tick. F2 (A2's "every treatment cell" quantifier ranges over the UNSHAPED
sparse control, whose surviving-advantage fraction legitimately tends to 0 in exactly the
H1-favourable world, self-routing that result to substrate_not_ready_requeue) and the
GATE half of F1, and F3 (autopsy item 3's lPFC SHARE of the summed modulatory accumulator
needs e3.e3_score_decomp_enabled, whose ~24 gated sites are not verifiably
behaviour-neutral) all amend a gate set ratified at a user gate, so they were RAISED AS A
DECISION rather than decided by the session. RESOLVED 2026-09-18 by the user (option B):
F2 FIXED by narrowing A2 to A2_GATED_ARMS; the F1 gate half and F3 DECLINED -- the
saturation statistic stays recorded-not-gated and e3_score_decomp_enabled stays off. Full
findings, measurements and option table:
REE_assembly evidence/planning/exq1039a_absorption_gate_redteam_staged_20260918.md.
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "experiments") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "experiments"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from experiment_protocol import emit_outcome  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402
from experiments._metrics import check_degeneracy  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell, reset_all_rng  # noqa: E402
import experiments._lib.allon_training as allon  # noqa: E402
from experiments._lib.allon_training import (  # noqa: E402
    _train_all_on_agent,
    _consumed_summaries,
    _obs_harm,
    _obs_harm_a,
    _obs_harm_history,
    POLICY_TEMPERATURE,
)
from experiments._lib.zworld_encoder_guard import (  # noqa: E402
    latent_stack_snapshot,
    assert_world_encoder_trained,
)
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
import experiments.v3_exq_724_competence_localization_diagnostic as x724  # noqa: E402

EXPERIMENT_TYPE = (
    "v3_exq_1039a_mech428_inv086_waypoint_field_consumer_drive_signal_absorption_gated"
)
QUEUE_ID = "V3-EXQ-1039a"
SUPERSEDES = "V3-EXQ-1039"
CLAIM_IDS = ["INV-086", "MECH-428"]
EXPERIMENT_PURPOSE = "diagnostic"
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"
DEVICE = torch.device("cpu")

# --- env: navigation-isolated train/eval config (V3-EXQ-1004 rationale, reapplied) -------
GRID_SIZE = 12
N_WAYPOINTS = 3
STEPS_PER_EPISODE = 80
WAYPOINT_VISIT_REWARD = 0.2
WAYPOINT_FIELD_DECAY = 0.25
WAYPOINT_COMPLETION_REWARD = 0.8
SEQUENCE_COMMITMENT_TIMEOUT = 20
HAZARD_FREE_CONTAMINATION_GATE = True

# --- env: dedicated P0a warmup config (generic; resources present for SD-018 target) -----
WARM_NUM_HAZARDS = 2
WARM_NUM_RESOURCES = 3

# --- agent / encoder (SD-008: alpha_world >= 0.9 for z_world fidelity) -------------------
ALPHA_WORLD = 0.9
ALPHA_SELF = 0.3

# --- budget: BYTE-IDENTICAL to V3-EXQ-1039 (autopsy item 5 -- no power bump) -------------
ZWORLD_P0_EPISODES = 20
P0_WARMUP_EPISODES = 25
RL_EPISODES = 90
N_DEMO_EPISODES = 25
EVAL_EPISODES = 10

SHAPING_COEF = 1.0

SEEDS: List[int] = [42, 43, 44]
MIN_SEEDS = 2                  # strict majority of 3

TREATMENT_ARMS: Tuple[str, ...] = ("sparse_rl", "shaped_rl", "demo_warmstart")
ANCHOR_ARMS: Tuple[str, ...] = ("greedy_oracle", "random_walk")
ARM_ORDER: Tuple[str, ...] = TREATMENT_ARMS + ANCHOR_ARMS

# --- pre-registered bars carried verbatim from V3-EXQ-1039 -------------------------------
ORACLE_FLOOR = 4.0
MIN_ACHIEVABLE_SPAN = 2.0
BASELINE_BLOCKED_FRAC = 0.35
LIFT_MARGIN_FRAC = 0.20
MIN_DISTINCT_CELLS = 8.0

# --- NEW pre-registered bars: the paired absorption / conversion gate ---------------------
# A1. shaped_rl's P1 mean episode return must exceed sparse_rl's by this many standard
# deviations of sparse_rl's OWN per-episode return distribution. A SEPARATION-IN-SD bar,
# self-calibrating to this run's own noise scale, because no piloted effect size exists at
# this budget (same discipline, and the same stated cost, as 1039's span-relative
# LIFT_MARGIN_FRAC). 1.0 sd is the weakest bar that still excludes an overlap-dominated
# pair; the 1039 observation it must not fail to clear was 250-500x, so this bar is
# deliberately far below the effect the manipulation is designed to produce -- it is an
# ABSORPTION floor, not a capability bar.
RETURN_SEPARATION_SD = 1.0
# The separation denominator is max(sparse_sd, shaped_sd, RETURN_SD_EPS): using BOTH arms'
# noise, not only the baseline's, because a deterministic sparse arm (sd exactly 0, which
# this driver's own dry run produced) would otherwise divide by EPS and report a separation
# of ~3e6 -- an artifact of the epsilon, not a measurement, and unreadable to anyone
# checking the bar. RETURN_SEPARATION_CAP bounds the reported value for the genuinely
# two-sided-deterministic case (both sds 0), where the separation is real but unbounded;
# capping only ever moves the number DOWN, i.e. toward the failing side of this floor gate,
# so it cannot manufacture a pass. The raw per-seed means and sds are recorded alongside.
RETURN_SD_EPS = 1e-6
RETURN_SEPARATION_CAP = 1000.0
# A2. Fraction of sampled REINFORCE terms clearing abs(adv) >= ADV_MIN_THRESHOLD (0.005,
# defined in allon_training). The autopsy requires "above zero"; a bare > 0 would pass on a
# single surviving term in tens of thousands, which is not a gradient the manipulation could
# have converted through. 5% is a floor well clear of that degenerate case and still far
# below any level at which it would function as a learning-capability bar.
ADV_SURVIVING_FRAC_FLOOR = 0.05
# WHICH ARMS A2 GATES ON -- a USER-RATIFIED AMENDMENT of the autopsy's wording, decided by
# AskUserQuestion on 2026-09-18 (option B) after this driver's own Step 4.5 red-team pass.
#
# THE AUTOPSY SAID "in every treatment cell" (section 14 item 1). The red-team showed that
# quantifier is unmeetable in exactly the world the hypothesis predicts: the REINFORCE
# advantage is `ep_return - EMA_baseline` with EMA_DECAY 0.9 (`allon_training.py:412,
# 796-797`), so after one 0.2 waypoint visit the baseline decays below ADV_MIN_THRESHOLD
# within ~14 reward-free episodes and a BLOCKED sparse_rl's terms are skipped by
# construction. V3-EXQ-1039 recorded sparse_rl@seed44 at 0.0 visits/ep, and this driver's
# own dry run reproduced the shape (sparse and demo at 0.0, shaped at 1.0). Under the
# original quantifier the result "the sparse signal produced no gradient, the dense one
# did, and behaviour still did not move" -- a genuine H1-negative finding, and the single
# most likely outcome -- would self-route to `substrate_not_ready_requeue`: an instrument
# label pinned on the manipulation's own CONTROL regime, from a requeue that cannot clear
# by re-running.
#
# WHY NARROWING DOES NOT WEAKEN THE GATE. A2 exists so a FLAT NULL between arms is not read
# as "reward density does not matter" when in fact no arm received a gradient (the chip's
# own framing: "before any flat null is admissible"). That failure is fully detected by
# gating the arm whose manipulation IS the training signal: if shaped_rl -- the arm carrying
# the denser signal -- shows no surviving advantage, nothing in this design could have
# converted and the run is genuinely not ready. A blocked sparse_rl showing none is the
# EXPECTED reading of a working control, not an instrument failure.
#
# demo_warmstart is excluded for a different and structural reason: it trains on the SAME
# unshaped env as sparse_rl, so its P1 returns match sparse's by construction and its
# absorption evidence is precondition A3 (phase-A CE falling below chance), not A2.
#
# NOT EXEMPT FROM SCRUTINY, ONLY FROM THE GATE: both ungated arms' surviving fractions are
# recorded per cell under the precondition's `ungated_arms_surviving_frac`, so an autopsy
# reads them without re-running anything.
A2_GATED_ARMS: Tuple[str, ...] = ("shaped_rl",)
# A3. demo_warmstart's phase-A CE must fall to at most this fraction of chance
# (ln(mean n_candidates)) over the final tenth of its updates. 1039 measured exactly
# 1.0 x chance (3.465736 = ln(32)); 0.9 is the weakest bar that excludes that reading.
DEMO_CE_CHANCE_FRAC = 0.9
DEMO_CE_WINDOW_FRAC = 0.10     # final tenth of the CE updates
# B1. RATIFIED at the Step 8 user gate 2026-09-16T13:06:22Z: floor 3, NOT MECH-341's 4.5.
FIRST_ACTION_CLASS_FLOOR = 3
# "a MAJORITY of ticks" -- strictly more than half, hence comparator ">" not ">=".
FIRST_ACTION_MAJORITY_FRAC = 0.5
# B2. RATIFIED: ">= 2 of 3 seeds" must show at least one differing committed action.
ABLATION_MIN_SEEDS = MIN_SEEDS
ABLATION_EVAL_EPISODES = 5     # paired ablation rollout length (eval-only; not the DV)
# C0d. The bias head's last-linear weight-norm delta AS A FLOAT (autopsy item (d)), worst
# cell, not 1039's boolean-on-a-majority.
HEAD_DELTA_FLOOR = 1e-9
# C0e. Fraction of P1 ticks whose CONSUMED candidate summaries were degenerate, computed by
# the shared build from the summaries the head actually read (autopsy item (c)).
SUMMARY_DEGENERATE_FRAC_CEILING = 0.5

# --- dry-run / probe budgets --------------------------------------------------------------
DRY_SEEDS = [42]
DRY_ZWORLD_P0 = 2
DRY_P0_WARMUP = 2
DRY_RL = 3
DRY_DEMO = 2
DRY_STEPS = 10
DRY_EVAL = 2
DRY_ABLATION_EVAL = 1


def _mean(vals: List[float]) -> float:
    return float(statistics.fmean(vals)) if vals else 0.0


def _num(value: Any, default: Optional[float] = None) -> Optional[float]:
    """Coerce a telemetry field to a finite float, or `default`.

    ABSENT IS NOT ZERO (the shared build's own invariant): a `None` from an empty sample
    must not silently become a measured 0.0 anywhere a gate reads it, so every call site
    passes the default it actually wants and says why.
    """
    if value is None or isinstance(value, bool):
        return default
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    return f if math.isfinite(f) else default


# ---------------------------------------------------------------------------------------
# Env builders (verbatim from V3-EXQ-1039)
# ---------------------------------------------------------------------------------------
def _make_warm_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        seed=seed, size=GRID_SIZE, use_proxy_fields=True, subgoal_mode=True,
        num_waypoints=N_WAYPOINTS, waypoint_visit_reward=WAYPOINT_VISIT_REWARD,
        subgoal_arrival_position_check=True,
        num_hazards=WARM_NUM_HAZARDS, num_resources=WARM_NUM_RESOURCES,
        waypoint_completion_reward=WAYPOINT_COMPLETION_REWARD,
        sequence_commitment_timeout=SEQUENCE_COMMITMENT_TIMEOUT,
        waypoint_proximity_field_enabled=True,
        waypoint_field_decay=WAYPOINT_FIELD_DECAY,
    )


def _make_train_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        seed=seed, size=GRID_SIZE, use_proxy_fields=True, subgoal_mode=True,
        num_waypoints=N_WAYPOINTS, waypoint_visit_reward=WAYPOINT_VISIT_REWARD,
        subgoal_arrival_position_check=True,
        num_hazards=0, num_resources=0, energy_decay=0.0,
        hazard_free_contamination_gate=HAZARD_FREE_CONTAMINATION_GATE,
        waypoint_completion_reward=WAYPOINT_COMPLETION_REWARD,
        sequence_commitment_timeout=SEQUENCE_COMMITMENT_TIMEOUT,
        waypoint_proximity_field_enabled=True,
        waypoint_field_decay=WAYPOINT_FIELD_DECAY,
    )


class _ShapedWaypointEnv:
    """Proxy around a navigation-isolated CausalGridWorldV2 that adds a MONOTONE, NON-
    NEGATIVE progress bonus toward the pending waypoint to the reward channel `step()`
    returns: `bonus = max(0, manhattan_prev - manhattan_next)`. Observations, transitions
    and env dynamics are untouched -- only the scalar reward is augmented. shaping_coef=0.0
    would be byte-identical to the unwrapped env; sparse_rl never uses this wrapper at all,
    so there is no shared-code path that could silently leak shaping into the sparse arm.

    CARRIED VERBATIM FROM V3-EXQ-1039 -- INCLUDING ITS KNOWN ARTIFACT. `_dist_prev` is not
    reset when the pending waypoint respawns (`sequence_commitment_timeout` fires a fresh
    randomised waypoint every 20 uncredited steps, and completing a sequence respawns the
    whole set), so a respawn can register as a large spurious "approach" and a STATIONARY
    policy still accrues ~1.5 shaped bonus per episode (failure_autopsy_V3-EXQ-1039
    section 3a). This is NOT fixed here, deliberately: the autopsy names it as the reason
    absorption evidence alone is insufficient, and answers it with the conversion-capability
    gate (B1/B2) rather than by changing the manipulation. Changing the wrapper would change
    what this leg measures and make 1039a a different experiment rather than a repair of
    this one -- it is an instrument repair, not a redesign.

    NOT Ng-Harada-Russell potential-difference shaping, and deliberately so -- V3-EXQ-1039's
    Step 4.5 red-team pass (fable-5.1, cross-model) found that design BLOCKING.
    `_train_all_on_agent`'s P1 REINFORCE credits every tick in an episode with the SAME
    whole-episode return (`ep_reward`, summed once, broadcast to every `(cand_features, sel)`
    tuple). A telescoping potential-difference term `gamma*Phi(s')-Phi(s)` collapses, under
    exactly this whole-episode-sum objective, to the single boundary quantity
    `Phi(s_T)-Phi(s_0)` -- it adds ZERO per-step reward density to what the learner is
    actually credited with, and that boundary term is dominated by where the episode happens
    to truncate rather than by anything the training-signal manipulation is supposed to test.
    A monotone per-step progress bonus is additive (not a cancelling difference) across the
    whole-episode sum, so it genuinely increases `ep_reward` and its correlation with
    within-episode approach behaviour. Clamped to `max(0, ...)` so it cannot introduce a NEW
    negative `harm_signal` into `agent.update_residue` beyond what the unshaped env can
    already produce -- and in this navigation-isolated config (hazards=0, resources=0) the
    unshaped env's `harm_signal` is provably non-negative in every step, so `shaped_rl`'s
    stays non-negative like every other arm's and the E3 harm-triggered commitment-abort
    branch (gated on `harm_signal < 0`) never fires differently across arms on this account.
    The progress bonus DOES add a larger positive value into `update_residue`'s
    benefit-accumulation path than the other arms see on progress steps -- disclosed, not
    eliminated (V3-EXQ-1039 red-team F2): REE's reward-contingent residue/benefit machinery
    reads the same reward channel this experiment's question is about, and decoupling them
    would require editing the shared, heavily-audited `_train_all_on_agent`, which this
    driver deliberately does not do."""

    def __init__(self, env: CausalGridWorldV2, shaping_coef: float) -> None:
        self._env = env
        self.shaping_coef = float(shaping_coef)
        self._dist_prev: Optional[int] = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)

    def reset(self):
        flat, obs = self._env.reset()
        self._dist_prev = _pending_waypoint_manhattan(self._env)
        return flat, obs

    def step(self, action):
        flat, r, done, info, obs = self._env.step(action)
        dist_next = _pending_waypoint_manhattan(self._env)
        bonus = 0.0
        if self._dist_prev is not None and dist_next is not None:
            bonus = max(0.0, float(self._dist_prev - dist_next))
        self._dist_prev = dist_next
        shaped = float(r) + self.shaping_coef * bonus
        return flat, shaped, done, info, obs


# ---------------------------------------------------------------------------------------
# Agent builder. V3-EXQ-724/734/737/742 "all-ON" recipe, IDENTICAL to V3-EXQ-1039 except
# for `lateral_pfc_capture_head_diagnostics=True` (and its construction-time assert).
#
# WHY THAT ONE FLAG IS SAFE TO ADD, verified at source rather than assumed. It is the gate
# for `hidden_dead_relu_frac`, which failure_autopsy_V3-EXQ-1039 item (d) requires this run
# to record. `ree_core/pfc/lateral_pfc_analog.py:454-472` unrolls `rule_bias_head`'s
# `Sequential` layer-by-layer (Linear -> ReLU -> Linear) under the flag instead of calling it
# once; its own comment states, and reading confirms, that this is the SAME sequence of ops
# `nn.Sequential.__call__` performs, so `bias_raw` is bit-identical. The extra work is two
# `no_grad` reductions written to `_last_hidden_dead_relu_frac` /
# `_last_rule_summary_magnitude_ratio`. NO RNG DRAW, no parameter write, no branch on the
# returned bias -- the same "no-op by construction, not by flag" property the shared
# absorption/conversion build is built on.
# ---------------------------------------------------------------------------------------
def _make_agent(env: CausalGridWorldV2, seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    kwargs = x724._base_config_kwargs(env)
    kwargs.update(x724._all_on_extra_kwargs())
    kwargs["alpha_world"] = ALPHA_WORLD
    kwargs["alpha_self"] = ALPHA_SELF
    kwargs["self_dim"] = 32
    kwargs["world_dim"] = 32
    # THE KEY IS `lateral_pfc_capture_head_diagnostics`, NOT `capture_head_diagnostics`.
    # `REEConfig.from_dims` SILENTLY SWALLOWS an unrecognised kwarg -- it neither raises
    # nor sets an attribute (memory `reference-reeconfig-from-dims-silent-kwargs`), so the
    # short spelling read as wired while `agent.lateral_pfc.config.capture_head_diagnostics`
    # stayed False and `absorption.hidden_dead_relu_frac_available` came back False with
    # n=0. Caught in this driver's own Step 2.5a probe and confirmed at source:
    # `ree_core/agent.py:1006` reads `config.lateral_pfc_capture_head_diagnostics` and
    # passes it into `LateralPFCConfig(capture_head_diagnostics=...)`.
    kwargs["lateral_pfc_capture_head_diagnostics"] = True
    cfg = x724.REEConfig.from_dims(**kwargs)
    agent = x724.REEAgent(cfg)
    # ASSERT THE WIRING, do not trust it. A silently-unwired diagnostics flag is exactly
    # how autopsy item (d) would go unrecorded again, and it is free to check here.
    _lp = getattr(agent, "lateral_pfc", None)
    if _lp is not None and not bool(
        getattr(getattr(_lp, "config", None), "capture_head_diagnostics", False)
    ):
        raise RuntimeError(
            "lateral_pfc.config.capture_head_diagnostics is False after construction -- "
            "hidden_dead_relu_frac (failure_autopsy_V3-EXQ-1039 item (d)) would not be "
            "recorded. Check the REEConfig kwarg spelling."
        )
    return agent


# ---------------------------------------------------------------------------------------
# Oracle / anchors (env ground truth; never reads the field -- cannot launder the
# manipulation into its own reference). Verbatim from V3-EXQ-1039/1004/1030.
# ---------------------------------------------------------------------------------------
def _oracle_action(env: CausalGridWorldV2) -> int:
    idx = int(getattr(env, "_next_waypoint_idx", 0))
    wps = getattr(env, "waypoints", []) or []
    if not wps or idx >= len(wps):
        return 4
    wx, wy = int(wps[idx][0]), int(wps[idx][1])
    ax, ay = int(env.agent_x), int(env.agent_y)
    dx, dy = wx - ax, wy - ay
    if abs(dx) >= abs(dy) and dx != 0:
        return 1 if dx > 0 else 0
    if dy != 0:
        return 3 if dy > 0 else 2
    return 4


def _pending_waypoint_manhattan(env: CausalGridWorldV2) -> Optional[int]:
    idx = int(getattr(env, "_next_waypoint_idx", 0))
    wps = getattr(env, "waypoints", []) or []
    if not wps or idx >= len(wps):
        return None
    wx, wy = int(wps[idx][0]), int(wps[idx][1])
    return abs(wx - int(env.agent_x)) + abs(wy - int(env.agent_y))


def _random_action(env: CausalGridWorldV2, rng: np.random.RandomState) -> int:
    return int(rng.randint(0, int(env.action_dim)))


# ---------------------------------------------------------------------------------------
# B1 instrument -- the ONE driver-side telemetry addition in this run (see docstring (B)).
# ---------------------------------------------------------------------------------------
class _FirstActionClassTap:
    """Read-only per-E3-TICK tap on the shared recipe's conversion-side state.

    WHY IT EXISTS, AND WHY IT IS NOT A DUPLICATE OF THE SHARED BUILD. The
    `sd-allon-training-signal-absorption-telemetry` build (ree-v3 996dec30) already records
    `conversion.distinct_first_action_classes_{n,mean,sd,min,max}` over P1 ticks. The gate
    ratified at the Step 8 user gate (2026-09-16T13:06:22Z) is ">= 3 on a MAJORITY of
    ticks", which is a median/quantile statement and is NOT recoverable from
    n/mean/sd/min/max -- a distribution can hold mean 3.5 with a minority above 3. This
    records the per-tick integers the shared helper ALREADY computes and returns, so the
    ratified fraction can be evaluated exactly.

    IT CANNOT CHANGE ANY ARM'S BEHAVIOUR. It calls the original, appends the returned value,
    and returns it unchanged; it draws no RNG, writes no parameter, and takes no branch on
    the value. The module attribute is restored in `__exit__`'s `finally` path, so an
    exception inside the training call cannot leave the shared module patched for another
    cell -- or for another experiment sharing this interpreter.

    WHAT ELSE IT RECORDS, AND WHY IT IS RECORDED RATHER THAN GATED. Step 4.5 red-team
    (fable, CONTESTED) established two recording gaps that make a flat null less
    attributable than the autopsy intends, both verified at source before acting on them:

      * `lpfc_bias_abs_mean` per tick, against `lateral_pfc.config.bias_scale`. On this
        recipe `rule_readout_consumer` is False, so `compute_bias` takes the HARD-CLAMP
        branch `bias_raw.clamp(-bias_scale, +bias_scale)`
        (`ree_core/pfc/lateral_pfc_analog.py:483-486`) rather than the gradient-preserving
        scaled-tanh branch above it -- and that module's own comment (`:171-179`) states
        that a saturated hard clamp has ZERO gradient, so REINFORCE cannot move the head.
        Meanwhile `_lpfc_reinforce_loss` increments `n_adv_surviving` BEFORE `compute_bias`
        is ever called (`allon_training.py:411-421`), so precondition A2 certifies the
        ADVANTAGE, never the gradient. This driver's own dry run produced exactly that
        pair in `shaped_rl`: `lpfc_adv_surviving_frac = 1.0` with a head weight-norm delta
        of EXACTLY 0.0, while `hidden_dead_relu_frac = 0.4375` (not dead) and
        `candidate_summary_degenerate_frac = 0.0` (not degenerate) rule out the two
        zero-gradient routes the design already records. Recording `bias_abs_mean == 
        bias_scale` is what makes the third route (clamp saturation) readable.
      * `modulatory_shortlist_size` per tick. With `use_f_eligibility_demotion=True` the
        lateral-PFC term can only change the committed action among the F-eligible prefix
        (`ree_core/predictors/e3_selector.py:4150` writes the size), so B1's class count
        over the FULL candidate pool can be high while the conversion surface is one
        candidate wide.

    Both are RECORDED, NOT GATED: the gate set was ratified at the Step 8 user gate and
    adding a precondition to it would change what this run measures, which is not this
    session's call. Recording them is what autopsy item 3 already asks for and is the
    difference between a null an autopsy can attribute and V3-EXQ-1039's, which it could
    not. The remaining half of item 3 -- the lPFC channel's SHARE of the summed modulatory
    accumulator -- needs `e3.e3_score_decomp_enabled`, whose ~24 gated sites could not be
    audited as behaviour-neutral in this session; it is raised as an open decision rather
    than switched on unverified.

    ONE COUNT PER FRESH E3 SELECTION, NOT PER ENV STEP. `_distinct_first_action_classes` is
    called on every P1 env step (`allon_training.py:707-710`, not gated on
    `ticks["e3_tick"]`), but `agent.generate_trajectories` returns the SAME cached
    `_committed_candidates` OBJECT while E3 has not ticked (`ree_core/agent.py:6315-6320`),
    and E3 ticks once per `heartbeat.e3_steps_per_tick` steps. Counting every call would
    therefore re-count one selection ~10x and weight the ratified MAJORITY by
    commitment-hold duration rather than by distinct selections -- the latched-diagnostic
    pseudo-replication hazard. Identity (`is`) against the previous object is an exact
    test for that cache, and the tap holds the reference so the comparison cannot be
    confused by a reused id. `n_held_ticks` is recorded so the true denominator stays
    auditable rather than being inferred from `n_ticks`.
    """

    def __init__(self, agent=None) -> None:
        self.agent = agent
        self.values: List[int] = []
        self.shortlist_sizes: List[float] = []
        self.bias_abs_means: List[float] = []
        self.n_held_ticks: int = 0
        self._prev = None
        self._orig = None

    def __enter__(self) -> "_FirstActionClassTap":
        self._orig = allon._distinct_first_action_classes
        orig = self._orig
        tap = self

        def _recording(candidates):
            v = orig(candidates)
            if candidates is tap._prev:
                # Held tick: E3 did not re-select, agent.generate_trajectories returned
                # the cached object. Record NOTHING; count it as the denominator does.
                tap.n_held_ticks += 1
                return v
            tap._prev = candidates
            if v is not None:
                tap.values.append(int(v))
            e3 = getattr(tap.agent, "e3", None)
            diag = getattr(e3, "last_score_diagnostics", None)
            if isinstance(diag, dict):
                sz = diag.get("modulatory_shortlist_size")
                if isinstance(sz, (int, float)) and not isinstance(sz, bool):
                    tap.shortlist_sizes.append(float(sz))
            lp = getattr(tap.agent, "lateral_pfc", None)
            bam = getattr(lp, "_last_bias_abs_mean", None)
            if isinstance(bam, (int, float)) and not isinstance(bam, bool):
                tap.bias_abs_means.append(float(bam))
            return v

        allon._distinct_first_action_classes = _recording
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self._orig is not None:
            allon._distinct_first_action_classes = self._orig
        return False

    def stats(self) -> Dict[str, Any]:
        """n / fraction-at-or-above-floor / median / min / max over the recorded ticks.

        An EMPTY sample reports `n_ticks: 0` and `None` for every derived statistic --
        never 0.0, which a gate would be unable to tell from a real measured zero. The P0
        warm call legitimately lands here: the shared build's conversion block is
        `if is_p1:`-gated at source, so `_distinct_first_action_classes` is never called
        during a `p1_episodes=0` call and there is genuinely nothing to record.
        """
        n = len(self.values)
        out: Dict[str, Any] = {
            "n_ticks": int(n),
            "floor": int(FIRST_ACTION_CLASS_FLOOR),
        }
        if n == 0:
            out.update({
                "n_ge_floor": 0, "frac_ge_floor": None,
                "median": None, "min": None, "max": None, "mean": None,
            })
            return out
        ge = sum(1 for v in self.values if v >= FIRST_ACTION_CLASS_FLOOR)
        srt = sorted(self.values)
        mid = n // 2
        median = float(srt[mid]) if n % 2 else float((srt[mid - 1] + srt[mid]) / 2.0)
        out.update({
            "n_ge_floor": int(ge),
            "frac_ge_floor": float(ge) / float(n),
            "median": median,
            "min": float(srt[0]),
            "max": float(srt[-1]),
            "mean": float(sum(self.values)) / float(n),
        })
        return out

    def side_stats(self) -> Dict[str, Any]:
        """The RECORDED-NOT-GATED conversion telemetry (see the class docstring)."""
        out: Dict[str, Any] = {"n_held_ticks": int(self.n_held_ticks)}
        sl = self.shortlist_sizes
        out["modulatory_shortlist_size_n"] = len(sl)
        out["modulatory_shortlist_size_mean"] = _mean(sl) if sl else None
        out["modulatory_shortlist_size_min"] = float(min(sl)) if sl else None
        out["modulatory_shortlist_size_max"] = float(max(sl)) if sl else None
        ba = self.bias_abs_means
        scale = None
        lp = getattr(self.agent, "lateral_pfc", None)
        cfg = getattr(lp, "config", None)
        bs = getattr(cfg, "bias_scale", None)
        if isinstance(bs, (int, float)) and not isinstance(bs, bool):
            scale = float(bs)
        out["lpfc_bias_scale"] = scale
        out["lpfc_bias_abs_mean_n"] = len(ba)
        out["lpfc_bias_abs_mean_mean"] = _mean(ba) if ba else None
        out["lpfc_bias_abs_mean_max"] = float(max(ba)) if ba else None
        # SATURATION FRACTION: what share of fresh selections had the head pinned at its
        # own clamp rail, where the hard-clamp branch's gradient is exactly zero. None
        # (not 0.0) when unmeasurable, so absence cannot read as "never saturated".
        if ba and scale is not None and scale > 0.0:
            tol = 1e-6 * max(1.0, scale)
            out["lpfc_bias_saturated_frac"] = (
                float(sum(1 for b in ba if abs(b - scale) <= tol)) / float(len(ba))
            )
        else:
            out["lpfc_bias_saturated_frac"] = None
        return out


# ---------------------------------------------------------------------------------------
# Rollout / eval counting (verbatim from V3-EXQ-1039).
# ---------------------------------------------------------------------------------------
EVAL_TEMPERATURE = 0.2


def _rollout_counts(env: CausalGridWorldV2, act_fn, n_episodes: int,
                    steps_per_episode: int) -> Dict[str, Any]:
    visits, seqs, cells, steps = [], [], [], []
    for _ep in range(n_episodes):
        flat, obs = env.reset()
        v = s = 0
        seen = {(int(env.agent_x), int(env.agent_y))}
        n_steps = 0
        for _t in range(steps_per_episode):
            n_steps += 1
            a = act_fn(env, obs, flat)
            flat, _r, done, info, obs = env.step(a)
            tt = str(info.get("transition_type", "") or "")
            if tt == "waypoint":
                v += 1
            elif tt == "sequence_complete":
                v += 1
                s += 1
            seen.add((int(env.agent_x), int(env.agent_y)))
            if done:
                break
        visits.append(float(v))
        seqs.append(float(s))
        cells.append(float(len(seen)))
        steps.append(float(n_steps))
    return {
        "waypoints_visited_per_ep": _mean(visits),
        "sequences_completed_per_ep": _mean(seqs),
        "n_eval_episodes": int(n_episodes),
        "distinct_cells_per_ep": _mean(cells),
        "steps_per_ep": _mean(steps),
    }


def _agent_eval_act(agent) -> Any:
    def _fn(env: CausalGridWorldV2, obs: Dict[str, Any], flat: torch.Tensor) -> int:
        with torch.no_grad():
            action = agent.act(flat, temperature=EVAL_TEMPERATURE)
        if not torch.isfinite(action).all():
            return int(np.random.randint(0, int(env.action_dim)))
        return int(action[0].argmax().item())
    return _fn


def _oracle_eval_act(env: CausalGridWorldV2, obs: Dict[str, Any], flat: torch.Tensor) -> int:
    return _oracle_action(env)


def _make_random_eval_act(seed: int):
    rng = np.random.RandomState(seed + 777)

    def _fn(env: CausalGridWorldV2, obs: Dict[str, Any], flat: torch.Tensor) -> int:
        return _random_action(env, rng)
    return _fn


# ---------------------------------------------------------------------------------------
# B2 instrument -- eval-side lateral-PFC bias-head ABLATION sensitivity.
# ---------------------------------------------------------------------------------------
def _committed_action_trace(agent, seed: int, n_episodes: int,
                            steps_per_episode: int) -> List[int]:
    """One deterministic-from-`seed` eval rollout, returning the committed action classes.

    PAIRING IS THE WHOLE POINT. The two calls this function serves (head TRAINED, head
    ZEROED) must differ ONLY by the head, so this resets every RNG stream to the same value
    and builds a fresh env from the same seed on each call. `reset_all_rng` is the same
    helper `arm_cell` uses at cell entry; calling it again here is safe because it happens
    AFTER all training and AFTER the primary DV rollout, so it cannot perturb either.
    """
    reset_all_rng(seed + 900_000_007)
    env = _make_train_env(seed)
    trace: List[int] = []
    for _ep in range(n_episodes):
        flat, _obs = env.reset()
        agent.reset()
        for _t in range(steps_per_episode):
            with torch.no_grad():
                action = agent.act(flat, temperature=EVAL_TEMPERATURE)
            if not torch.isfinite(action).all():
                trace.append(-1)
                a_idx = 0
            else:
                a_idx = int(action[0].argmax().item())
                trace.append(a_idx)
            flat, _r, done, _info, _obs = env.step(a_idx)
            if done:
                break
    return trace


def _bias_ablation_sensitivity(agent, seed: int, n_episodes: int,
                               steps_per_episode: int) -> Dict[str, Any]:
    """Does zeroing the lateral-PFC bias head change ANY committed action?

    Returns `differs: False` with `reason` set when the check could not be performed at all
    (no lateral-PFC on this agent) -- never a bare False that a gate would read as a measured
    negative. A cell that cannot be measured must not certify itself.
    """
    lpfc = getattr(agent, "lateral_pfc", None)
    if lpfc is None:
        return {"measurable": False, "differs": False, "n_differing": None,
                "n_compared": 0, "reason": "no lateral_pfc on this agent"}
    trained = _committed_action_trace(agent, seed, n_episodes, steps_per_episode)
    saved = [p.detach().clone() for p in lpfc.bias_head_parameters()]
    try:
        with torch.no_grad():
            for p in lpfc.bias_head_parameters():
                p.zero_()
        ablated = _committed_action_trace(agent, seed, n_episodes, steps_per_episode)
    finally:
        with torch.no_grad():
            for p, s in zip(lpfc.bias_head_parameters(), saved):
                p.copy_(s)
    n_cmp = min(len(trained), len(ablated))
    n_diff = sum(1 for i in range(n_cmp) if trained[i] != ablated[i])
    # A LENGTH difference is itself a behavioural difference (an episode terminated at a
    # different step), so it counts -- not counting it would under-report the sensitivity
    # this gate exists to detect.
    length_diff = len(trained) != len(ablated)
    return {
        "measurable": True,
        "differs": bool(n_diff > 0 or length_diff),
        "n_differing": int(n_diff),
        "n_compared": int(n_cmp),
        "len_trained": len(trained), "len_ablated": len(ablated),
        "length_differs": bool(length_diff),
    }


# ---------------------------------------------------------------------------------------
# demo_warmstart phase A -- auxiliary supervised warm-start on the lateral-PFC bias head.
# Carried from V3-EXQ-1039 (including its red-team F4 miss-skip fix); the ONLY change is
# that the CE trace and the per-update candidate count are now recorded, so precondition A3
# can compare the final-window CE against ln(mean n_candidates) -- 1039 recorded only the
# single LAST loss value, which is what left `3.465736 == ln(32)` to be spotted by hand in
# an autopsy rather than gated on by the run itself.
# ---------------------------------------------------------------------------------------
def _demo_warmstart(agent, env: CausalGridWorldV2, seed: int, n_episodes: int,
                    steps_per_episode: int, denom: int) -> Dict[str, Any]:
    has_lpfc = getattr(agent, "lateral_pfc", None) is not None
    if not has_lpfc or n_episodes <= 0:
        return {"demo_warmstart_ran": False, "demo_ce_final_loss": None,
                "demo_ce_window_mean": None, "demo_ce_chance_reference": None,
                "demo_n_ce_updates": 0}
    bias_opt = torch.optim.Adam(list(agent.lateral_pfc.bias_head_parameters()), lr=1e-3)
    last_loss: Optional[float] = None
    ce_trace: List[float] = []
    n_cand_trace: List[int] = []
    n_labelled_ticks = 0
    n_target_matched = 0
    n_target_defaulted = 0
    for ep in range(n_episodes):
        _flat, obs_dict = env.reset()
        agent.reset()
        for _step in range(steps_per_episode):
            body = obs_dict["body_state"].float()
            world = obs_dict["world_state"].float()
            if body.dim() == 1:
                body = body.unsqueeze(0)
            if world.dim() == 1:
                world = world.unsqueeze(0)
            latent = agent.sense(
                obs_body=body, obs_world=world,
                obs_harm=_obs_harm(obs_dict), obs_harm_a=_obs_harm_a(obs_dict),
                obs_harm_history=_obs_harm_history(obs_dict),
            )
            ticks = agent.clock.advance()
            wdim = latent.z_world.shape[-1]
            e1_prior = (
                agent._e1_tick(latent) if ticks.get("e1_tick", False)
                else torch.zeros(1, wdim, device=agent.device)
            )
            candidates = agent.generate_trajectories(latent, e1_prior, ticks)
            oracle_a = _oracle_action(env)
            # Gate the auxiliary CE update on a FRESH E3 tick: on a held tick, agent.py
            # returns the cached action and generate_trajectories returns cached candidates
            # BEFORE e3.select() is reached, so an ungated update would re-train on the same
            # cached candidate_summaries once per held step -- weighting the CE loss by
            # commitment-hold duration rather than by genuinely distinct selections.
            if ticks.get("e3_tick", False) and candidates and len(candidates) >= 2:
                cand_features = _consumed_summaries(agent, candidates)
                if cand_features is not None and torch.isfinite(cand_features).all():
                    # V3-EXQ-1039 red-team F4: a miss (no candidate's first action equals
                    # the oracle direction) SKIPS the update rather than defaulting to
                    # candidate 0 -- training "prefer whatever the proposer ranks first" on
                    # a miss is not learning from demonstration.
                    target_idx = None
                    for ci, c in enumerate(candidates):
                        if (
                            getattr(c, "actions", None) is not None
                            and c.actions.shape[1] >= 1
                            and int(c.actions[:, 0, :].argmax(-1).reshape(-1)[0].item())
                            == oracle_a
                        ):
                            target_idx = min(ci, cand_features.shape[0] - 1)
                            break
                    if target_idx is None:
                        n_target_defaulted += 1
                    else:
                        n_target_matched += 1
                        bias = agent.lateral_pfc.compute_bias(cand_features)
                        log_p = torch.log_softmax(-bias / POLICY_TEMPERATURE, dim=0)
                        loss = -log_p[min(target_idx, log_p.shape[0] - 1)]
                        if torch.isfinite(loss):
                            bias_opt.zero_grad()
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(
                                agent.lateral_pfc.bias_head_parameters(), 1.0
                            )
                            bias_opt.step()
                            last_loss = float(loss.detach())
                            # A3 instrument: the CE and the SIZE of the softmax it was taken
                            # over, per update -- ln(that size) is the chance level the CE
                            # must fall below, and it is not a constant (the candidate set
                            # size varies per tick).
                            ce_trace.append(last_loss)
                            n_cand_trace.append(int(log_p.shape[0]))
                            n_labelled_ticks += 1

            action = agent.select_action(candidates, ticks)
            if action is None or not torch.isfinite(action).all():
                idx = int(np.random.randint(0, env.action_dim))
                action = torch.zeros(1, env.action_dim, device=agent.device)
                action[0, idx] = 1.0
            _flat, _harm, done, _info, obs_dict = env.step(action)
            if done:
                break
        cur = ep + 1
        if cur % 20 == 0 or cur == n_episodes:
            print(f"  [train] demo_warmstart seed={seed} phase=DEMO ep {cur}/{denom}",
                  flush=True)

    n_up = len(ce_trace)
    window = max(1, int(round(n_up * DEMO_CE_WINDOW_FRAC))) if n_up else 0
    ce_window_mean = _mean(ce_trace[-window:]) if window else None
    mean_n_cand = _mean([float(c) for c in n_cand_trace]) if n_cand_trace else None
    chance_ref = (
        float(math.log(mean_n_cand)) if mean_n_cand and mean_n_cand > 1.0 else None
    )
    return {
        "demo_warmstart_ran": True,
        "demo_ce_final_loss": last_loss,
        "demo_ce_window_mean": ce_window_mean,
        "demo_ce_window_n_updates": int(window),
        "demo_n_ce_updates": int(n_up),
        "demo_mean_n_candidates": mean_n_cand,
        "demo_ce_chance_reference": chance_ref,
        "demo_n_labelled_ticks": int(n_labelled_ticks),
        "demo_n_target_matched": int(n_target_matched),
        "demo_n_target_defaulted": int(n_target_defaulted),
    }


# ---------------------------------------------------------------------------------------
# Per-cell runner
# ---------------------------------------------------------------------------------------
_ZG = ZGoalStreamAccumulator()

# The absorption keys this driver PROJECTS out of the shared build's block into flat,
# gate-readable row fields. Listed once so the projection and the gates cannot drift apart.
_ABSORPTION_PROJECT = (
    "p0_ep_return_n", "p0_ep_return_mean", "p0_ep_return_sd",
    "p1_ep_return_n", "p1_ep_return_mean", "p1_ep_return_sd",
    "p1_ep_return_min", "p1_ep_return_max",
    "p1_ep_advantage_n", "p1_ep_advantage_mean", "p1_ep_advantage_sd",
    "adv_min_threshold",
    "lpfc_adv_n_sampled", "lpfc_adv_n_surviving", "lpfc_adv_surviving_frac",
    "lpfc_adv_abs_mean", "lpfc_adv_abs_max",
    "ofc_adv_n_sampled", "ofc_adv_n_surviving", "ofc_adv_surviving_frac",
    "ofc_adv_n_bias_skipped",
    "lpfc_bias_head_present",
    "lpfc_bias_head_last_linear_weight_norm_pre",
    "lpfc_bias_head_last_linear_weight_norm_post",
    "lpfc_bias_head_last_linear_weight_norm_delta",
    "lpfc_bias_head_head_weight_norm_delta",
    "ofc_deval_head_last_linear_weight_norm_delta",
    "hidden_dead_relu_frac_available",
    "hidden_dead_relu_frac_n", "hidden_dead_relu_frac_mean",
    "hidden_dead_relu_frac_min", "hidden_dead_relu_frac_max",
)

_CONVERSION_PROJECT = (
    "n_p1_diag_ticks", "modulatory_authority_normalize_basis",
    "e3_raw_score_range_mean_n", "e3_raw_score_range_mean_mean",
    "e3_raw_score_std_mean_mean",
    "score_bias_abs_mean_mean", "score_bias_range_mean_mean",
    "score_bias_to_raw_range_ratio_mean",
    "modulatory_authority_active_mean", "modulatory_authority_scale_factor_mean",
    "modulatory_authority_ratio_mean", "modulatory_authority_ratio_competitive_mean",
    "modulatory_authority_range_mean",
    "distinct_first_action_classes_n", "distinct_first_action_classes_mean",
    "distinct_first_action_classes_sd", "distinct_first_action_classes_min",
    "distinct_first_action_classes_max",
    "candidate_summary_post_pre_norm_ratio_n",
    "candidate_summary_post_pre_norm_ratio_mean",
    "candidate_summary_degenerate_frac", "candidate_summary_degeneracy_floor",
    "candidate_summary_degenerate_computed",
)


def _project(block: Optional[Dict[str, Any]], keys: Tuple[str, ...],
             prefix: str) -> Dict[str, Any]:
    """Flatten selected shared-build telemetry keys under `prefix`.

    A MISSING BLOCK yields every key as None, never as 0/False -- the shared build's own
    "absent is None" invariant, carried across the projection so a gate reading a projected
    field cannot mistake "the recipe returned nothing" for "the recipe measured zero".
    """
    src = block if isinstance(block, dict) else {}
    return {f"{prefix}_{k}": src.get(k) for k in keys}


def _run_treatment_cell(arm_id: str, seed: int, zworld_p0: int, p0_warm: int, rl_eps: int,
                        n_demo: int, eval_eps: int, steps: int,
                        ablation_eval_eps: int) -> Dict[str, Any]:
    warm_env = _make_warm_env(seed)
    agent = _make_agent(warm_env, seed)

    before = latent_stack_snapshot(agent)
    zworld_env = _make_warm_env(seed + 500_000_003)  # F6-style offset: distinct layout stream
    # PHASE P0. Captured, not discarded (1039 discarded it) -- this is the "per phase" half
    # of the autopsy's "recorded per arm and per phase".
    with _FirstActionClassTap(agent) as tap_p0:
        warm_stats = _train_all_on_agent(
            agent, warm_env, seed=seed, p0_episodes=p0_warm, p1_episodes=0,
            steps_per_episode=steps, rung_id=f"h1_{arm_id}",
            total_denominator=max(1, p0_warm),
            zworld_p0_episodes=zworld_p0, zworld_p0_env=zworld_env,
        )
    zworld_report = assert_world_encoder_trained(
        agent, before, p0=zworld_p0, strict=False,
        context="v3_exq_1039a.warmup", escape_hint="strict=False deliberate",
    )

    # Kept from 1039 for continuity with its manifest, and so the OLD boolean readout and
    # the NEW float delta can be compared in one place by a later reader. The GATE (C0d)
    # routes on the shared build's float delta, never on this boolean.
    bias_norm_before = None
    if getattr(agent, "lateral_pfc", None) is not None:
        bias_norm_before = agent.lateral_pfc.get_state()["rule_bias_head_last_linear_weight_norm"]

    extra: Dict[str, Any] = {}
    # PHASE P1 -- the arm's own manipulation.
    with _FirstActionClassTap(agent) as tap_p1:
        if arm_id == "sparse_rl":
            train_env = _make_train_env(seed)
            rl_stats = _train_all_on_agent(
                agent, train_env, seed=seed, p0_episodes=0, p1_episodes=rl_eps,
                steps_per_episode=steps, rung_id="h1_sparse_rl",
                total_denominator=max(1, rl_eps),
            )
        elif arm_id == "shaped_rl":
            train_env = _ShapedWaypointEnv(_make_train_env(seed), SHAPING_COEF)
            rl_stats = _train_all_on_agent(
                agent, train_env, seed=seed, p0_episodes=0, p1_episodes=rl_eps,
                steps_per_episode=steps, rung_id="h1_shaped_rl",
                total_denominator=max(1, rl_eps),
            )
            train_env = train_env._env  # unwrap for eval below (unshaped rollout counts)
        elif arm_id == "demo_warmstart":
            train_env = _make_train_env(seed)
            demo_stats = _demo_warmstart(agent, train_env, seed, n_demo, steps,
                                         max(1, n_demo))
            extra.update(demo_stats)
            rl_stats = _train_all_on_agent(
                agent, train_env, seed=seed, p0_episodes=0, p1_episodes=rl_eps,
                steps_per_episode=steps, rung_id="h1_demo_warmstart",
                total_denominator=max(1, rl_eps),
            )
        else:
            raise ValueError(f"unknown treatment arm {arm_id!r}")

    # PRIMARY DV first, before anything that resets an RNG stream.
    eval_env = _make_train_env(seed)
    row = _rollout_counts(eval_env, _agent_eval_act(agent), eval_eps, steps)
    row["zworld_encoder_trained"] = bool(zworld_report.get("zworld_encoder_trained", False))
    if getattr(agent, "lateral_pfc", None) is not None:
        lpfc_state = agent.lateral_pfc.get_state()
        bias_norm_after = lpfc_state["rule_bias_head_last_linear_weight_norm"]
        row["lpfc_bias_head_moved_legacy_bool"] = bool(
            bias_norm_before is not None
            and abs(bias_norm_after - bias_norm_before) > 1e-9
        )
        # The 1039 C0d field, RECORDED but NOT GATED ON: it reads LateralPFCAnalog's own
        # flag, which is only computed when rule_readout_consumer is on and k >= 2 and is
        # therefore vacuous on this recipe (autopsy item (c)). C0e gates on the shared
        # build's COMPUTED fraction instead.
        row["lpfc_candidate_summary_degenerate_legacy_flag"] = bool(
            lpfc_state["candidate_summary_degenerate"]
        )
    else:
        row["lpfc_bias_head_moved_legacy_bool"] = False
        row["lpfc_candidate_summary_degenerate_legacy_flag"] = True

    # Shared-build telemetry, per phase.
    row.update(_project(warm_stats.get("absorption"), _ABSORPTION_PROJECT, "p0abs"))
    row.update(_project(warm_stats.get("conversion"), _CONVERSION_PROJECT, "p0conv"))
    row.update(_project(rl_stats.get("absorption"), _ABSORPTION_PROJECT, "p1abs"))
    row.update(_project(rl_stats.get("conversion"), _CONVERSION_PROJECT, "p1conv"))
    row["p0_n_ticks"] = warm_stats.get("n_p0_ticks")
    row["p1_n_ticks"] = rl_stats.get("n_p1_ticks")
    # B1 driver-side histogram, per phase (see _FirstActionClassTap).
    row["first_action_class_hist_p0"] = tap_p0.stats()
    row["first_action_class_hist_p1"] = tap_p1.stats()
    # RECORDED, NOT GATED (Step 4.5 red-team dispositions; see _FirstActionClassTap).
    row["conversion_side_telemetry_p0"] = tap_p0.side_stats()
    row["conversion_side_telemetry_p1"] = tap_p1.side_stats()

    # B2, LAST: it resets RNG streams for its paired rollouts, so it runs after the DV.
    row["bias_ablation"] = _bias_ablation_sensitivity(agent, seed, ablation_eval_eps, steps)

    row.update(extra)
    _ZG.observe(agent)
    return row


def _run_anchor_cell(arm_id: str, seed: int, eval_eps: int, steps: int) -> Dict[str, Any]:
    env = _make_train_env(seed)
    if arm_id == "greedy_oracle":
        return _rollout_counts(env, _oracle_eval_act, eval_eps, steps)
    if arm_id == "random_walk":
        return _rollout_counts(env, _make_random_eval_act(seed), eval_eps, steps)
    raise ValueError(f"unknown anchor {arm_id}")


def _arm_config_slice(arm_id: str, zworld_p0: int, p0_warm: int, rl_eps: int, n_demo: int,
                      eval_eps: int, steps: int) -> Dict[str, Any]:
    base = {
        "arm_id": arm_id, "steps_per_episode": int(steps), "eval_episodes": int(eval_eps),
        "grid_size": GRID_SIZE, "n_waypoints": N_WAYPOINTS,
        "waypoint_visit_reward": WAYPOINT_VISIT_REWARD,
        "waypoint_completion_reward": WAYPOINT_COMPLETION_REWARD,
        "waypoint_field_decay": WAYPOINT_FIELD_DECAY, "field_on": True,
        "alpha_world": ALPHA_WORLD, "alpha_self": ALPHA_SELF,
        "lateral_pfc_capture_head_diagnostics": True,
    }
    if arm_id in TREATMENT_ARMS:
        base.update({
            "kind": "reeagent_e1e3_allon", "zworld_p0_episodes": int(zworld_p0),
            "p0_warmup_episodes": int(p0_warm), "rl_episodes": int(rl_eps),
        })
        if arm_id == "shaped_rl":
            base.update({"shaping_coef": SHAPING_COEF})
        if arm_id == "demo_warmstart":
            base.update({"n_demo_episodes": int(n_demo)})
    else:
        base.update({"kind": "anchor"})
    return base


# ---------------------------------------------------------------------------------------
# Score / verdict
#
# WORST CELL, NEVER THE MEAN. Every gate below is a quantified claim over cells ("in every
# treatment cell", "on >= 2 of 3 seeds"), and the indexer RECOMPUTES `met` from the reported
# `measured` against `threshold`. So each precondition reports the EXTREMUM its own `met`
# tests, plus the `offending_cell` that produced it -- an in-band mean would recompute MET
# against a `met: False` and mask the cell that actually failed.
#
# UNMEASURABLE IS UNMET. A projected telemetry field that is None (the shared build's
# "absent is never zero" invariant) is scored at the failing end of its own bound, with the
# cell named. A cell that could not be measured must not certify itself.
# ---------------------------------------------------------------------------------------
def _treatment_cells(per_seed: Dict[str, List[Dict[str, Any]]]) -> List[Tuple[str, int, Dict[str, Any]]]:
    out: List[Tuple[str, int, Dict[str, Any]]] = []
    for arm in TREATMENT_ARMS:
        for row in per_seed[arm]:
            out.append((arm, int(row.get("seed", -1)), row))
    return out


def _worst(cells, value_fn, mode: str, unmeasurable: float) -> Tuple[float, Optional[str]]:
    """Extremum of `value_fn` over `cells`, with the cell id that produced it.

    `mode` is "min" for a FLOOR gate and "max" for a CEILING gate; `unmeasurable` is the
    value substituted when `value_fn` returns None, and must always be the FAILING end of
    the bound (see the section comment).
    """
    best: Optional[float] = None
    who: Optional[str] = None
    for arm, seed, row in cells:
        v = value_fn(row)
        v = unmeasurable if v is None else float(v)
        if best is None or (v < best if mode == "min" else v > best):
            best, who = v, f"{arm}@seed{seed}"
    if best is None:
        return float(unmeasurable), None
    return float(best), who


def _score(per_seed: Dict[str, List[Dict[str, Any]]], seeds: List[int]) -> Dict[str, Any]:
    def col(arm: str, key: str) -> List[float]:
        return [float(row[key]) for row in per_seed[arm]]

    oracle_v = col("greedy_oracle", "waypoints_visited_per_ep")
    random_v = col("random_walk", "waypoints_visited_per_ep")
    sparse_v = col("sparse_rl", "waypoints_visited_per_ep")
    shaped_v = col("shaped_rl", "waypoints_visited_per_ep")
    demo_v = col("demo_warmstart", "waypoints_visited_per_ep")

    n = len(seeds)
    oracle_mean = _mean(oracle_v)
    random_mean = _mean(random_v)
    cells = _treatment_cells(per_seed)

    # ---- C0 / C0b / C0c: carried verbatim from V3-EXQ-1039 -----------------------------
    c0_measured = min(oracle_v) if oracle_v else 0.0
    c0 = {
        "name": "greedy_oracle_clears_floor", "kind": "readiness",
        "description": (
            "greedy_oracle (env ground truth, never reads the field) visits/ep clears "
            "ORACLE_FLOOR on every seed at this budget/geometry -- the positive control "
            "that the env/geometry is navigable, same statistic C1 routes on."
        ),
        "control": "greedy_oracle waypoints_visited_per_ep, worst seed",
        "measured": round(float(c0_measured), 6), "threshold": float(ORACLE_FLOOR),
        "direction": "lower",
        "met": bool(c0_measured >= ORACLE_FLOOR),
    }

    zworld_frac = _mean([
        1.0 if row.get("zworld_encoder_trained") else 0.0 for _a, _s, row in cells
    ])
    c0b = {
        "name": "zworld_encoder_trained_majority_cells", "kind": "readiness",
        "description": (
            "Fraction of (seed, treatment-arm) cells whose SD-070 P0a warmup measurably "
            "moved the world encoder (assert_world_encoder_trained, strict=False)."
        ),
        "control": "latent_stack pre/post-warmup weight-delta audit",
        "measured": round(float(zworld_frac), 6), "threshold": 0.5,
        "direction": "lower",
        "met": bool(zworld_frac >= 0.5),
    }

    span = max(0.0, oracle_mean - random_mean)
    c0c = {
        "name": "achievable_span_clears_floor", "kind": "readiness",
        "description": (
            "(oracle - random) visits/ep span clears MIN_ACHIEVABLE_SPAN -- below this, "
            "LIFT_MARGIN_FRAC (a fraction OF the span) would be a near-zero, trivially-"
            "cleared bar rather than a meaningful one."
        ),
        "control": "greedy_oracle_mean - random_walk_mean, this run's own anchors",
        "measured": round(float(span), 6), "threshold": float(MIN_ACHIEVABLE_SPAN),
        "direction": "lower",
        "met": bool(span >= MIN_ACHIEVABLE_SPAN),
    }

    # ---- C0d: the 1039 boolean readout, REPLACED BY THE FLOAT DELTA (autopsy item (d)) --
    c0d_measured, c0d_cell = _worst(
        cells,
        lambda r: (abs(v) if (v := _num(r.get("p1abs_lpfc_bias_head_last_linear_weight_norm_delta"))) is not None else None),
        "min", 0.0,
    )
    c0d = {
        "name": "lpfc_bias_head_last_linear_weight_norm_delta_nonzero", "kind": "readiness",
        "description": (
            "Absolute post-minus-pre weight-norm delta of the lateral-PFC bias head's LAST "
            "LINEAR layer, as a FLOAT, in the WORST treatment cell. V3-EXQ-1039's C0d tested "
            "a boolean on a majority of cells and passed at 0.667 while the head's output "
            "stayed at chance; this reports the number instead. An unmeasurable cell "
            "(absent head) scores 0.0 and fails."
        ),
        "control": "allon_training absorption block, lpfc_bias_head norm delta, P1 phase",
        "measured": float(f"{c0d_measured:.12g}"), "threshold": float(HEAD_DELTA_FLOOR),
        "direction": "lower", "offending_cell": c0d_cell,
        "met": bool(c0d_measured >= HEAD_DELTA_FLOOR),
    }

    # ---- C0e: the COMPUTED candidate-summary degeneracy (autopsy item (c)) -------------
    c0e_measured, c0e_cell = _worst(
        cells, lambda r: _num(r.get("p1conv_candidate_summary_degenerate_frac")),
        "max", 1.0,
    )
    c0e_computed_everywhere = all(
        bool(row.get("p1conv_candidate_summary_degenerate_computed")) for _a, _s, row in cells
    )
    c0e = {
        "name": "candidate_summary_nondegenerate_computed", "kind": "readiness",
        "description": (
            "Fraction of P1 ticks whose CONSUMED candidate summaries were degenerate "
            "(post-centering norm <= floor x pre-centering norm), COMPUTED by the shared "
            "absorption/conversion build from the summaries the head actually read -- not "
            "LateralPFCAnalog's own candidate_summary_degenerate field, which is only "
            "computed when rule_readout_consumer is on and k >= 2 and is therefore vacuous "
            "on this recipe (the half of V3-EXQ-1039's C0d that passed on an __init__ "
            "value). WORST cell. Unmeasurable scores 1.0 and fails."
        ),
        "control": "allon_training conversion block, candidate_summary_degenerate_frac, P1",
        "measured": round(float(c0e_measured), 6),
        "threshold": float(SUMMARY_DEGENERATE_FRAC_CEILING),
        "direction": "upper", "offending_cell": c0e_cell,
        "computed_by_shared_build_in_every_cell": bool(c0e_computed_everywhere),
        "met": bool(c0e_measured <= SUMMARY_DEGENERATE_FRAC_CEILING),
    }

    # ---- A1: shaped-vs-sparse return separation (autopsy item 1) -----------------------
    a1_per_seed: List[Dict[str, Any]] = []
    for i, seed in enumerate(seeds):
        sp = per_seed["sparse_rl"][i]
        sh = per_seed["shaped_rl"][i]
        sp_mean = _num(sp.get("p1abs_p1_ep_return_mean"))
        sh_mean = _num(sh.get("p1abs_p1_ep_return_mean"))
        sp_sd = _num(sp.get("p1abs_p1_ep_return_sd"))
        sh_sd = _num(sh.get("p1abs_p1_ep_return_sd"))
        denom = max(sp_sd or 0.0, sh_sd or 0.0, RETURN_SD_EPS)
        if sp_mean is None or sh_mean is None:
            sep = None
            capped = False
        else:
            raw = (sh_mean - sp_mean) / denom
            capped = bool(raw > RETURN_SEPARATION_CAP)
            sep = min(raw, RETURN_SEPARATION_CAP)
        a1_per_seed.append({
            "seed": int(seed), "sparse_p1_return_mean": sp_mean,
            "shaped_p1_return_mean": sh_mean, "sparse_p1_return_sd": sp_sd,
            "shaped_p1_return_sd": sh_sd, "separation_denominator": denom,
            "both_arms_deterministic": bool((sp_sd or 0.0) <= 0.0
                                            and (sh_sd or 0.0) <= 0.0),
            "separation_sd": sep, "separation_capped": capped,
        })
    a1_vals = [(-1e18 if d["separation_sd"] is None else float(d["separation_sd"]))
               for d in a1_per_seed]
    a1_measured = min(a1_vals) if a1_vals else -1e18
    a1_cell = (f"shaped_vs_sparse@seed{a1_per_seed[a1_vals.index(a1_measured)]['seed']}"
               if a1_vals else None)
    a1 = {
        "name": "shaped_vs_sparse_return_separation", "kind": "readiness",
        "description": (
            "shaped_rl's P1 mean episode return exceeds sparse_rl's by at least "
            "RETURN_SEPARATION_SD standard deviations of sparse_rl's OWN per-episode return "
            "distribution (or shaped_rl's, whichever is larger -- see RETURN_SD_EPS), on "
            "EVERY seed (worst seed reported), reported capped at RETURN_SEPARATION_CAP. "
            "Self-calibrating to this run's own noise scale. ABSORPTION half, item 1: a "
            "shaped-vs-sparse "
            "pair must be shown to have received materially different returns before any "
            "flat null between them is admissible. Applies to this PAIR only -- "
            "demo_warmstart trains on the same unshaped env as sparse_rl by construction, "
            "so its absorption evidence is A3."
        ),
        "control": "allon_training absorption block, p1_ep_return mean/sd, both arms",
        "measured": float(f"{a1_measured:.6g}"), "threshold": float(RETURN_SEPARATION_SD),
        "direction": "lower", "offending_cell": a1_cell,
        "per_seed": a1_per_seed,
        "met": bool(a1_measured >= RETURN_SEPARATION_SD),
    }

    # ---- A2: non-skipped advantage fraction (autopsy item 1, THE key absorption number) -
    # USER-RATIFIED AMENDMENT, 2026-09-18 (AskUserQuestion, option B). The autopsy's literal
    # wording is "in every treatment cell"; this gate now quantifies over A2_GATED_ARMS
    # (shaped_rl only) and records the other arms as TELEMETRY. See A2_GATED_ARMS for the
    # reasoning and for why the unshaped arms are not exempted from scrutiny, only from the
    # gate.
    a2_cells = [c for c in cells if c[0] in A2_GATED_ARMS]
    a2_measured, a2_cell = _worst(
        a2_cells, lambda r: _num(r.get("p1abs_lpfc_adv_surviving_frac")), "min", 0.0,
    )
    a2_ungated = {}
    for _arm, _seed, _row in cells:
        if _arm in A2_GATED_ARMS:
            continue
        a2_ungated[f"{_arm}@seed{_seed}"] = _num(
            _row.get("p1abs_lpfc_adv_surviving_frac")
        )
    a2 = {
        "name": "adv_surviving_frac_clears_floor", "kind": "readiness",
        "description": (
            "Fraction of sampled REINFORCE outcome-buffer terms clearing "
            "abs(adv) >= ADV_MIN_THRESHOLD (0.005) on the lateral-PFC head, in the WORST "
            "cell of the arms whose MANIPULATION IS THE TRAINING SIGNAL (A2_GATED_ARMS = "
            "shaped_rl). ~0 there means the dense signal produced no usable gradient at "
            "all whatever its reward was -- the absorption failure V3-EXQ-1039 could not "
            "see. A cell that drew no REINFORCE term reports None and is scored 0.0 "
            "(fails). sparse_rl and demo_warmstart are recorded under "
            "`ungated_arms_surviving_frac` and deliberately NOT gated: see A2_GATED_ARMS."
        ),
        "control": "allon_training absorption block, lpfc_adv_surviving_frac, P1 phase",
        "measured": round(float(a2_measured), 6),
        "threshold": float(ADV_SURVIVING_FRAC_FLOOR),
        "direction": "lower", "offending_cell": a2_cell,
        "gated_arms": list(A2_GATED_ARMS),
        "ungated_arms_surviving_frac": a2_ungated,
        "scope_amendment": (
            "user-ratified 2026-09-18 (option B); the autopsy's own wording was "
            "'in every treatment cell'"
        ),
        "met": bool(a2_measured >= ADV_SURVIVING_FRAC_FLOOR),
    }

    # ---- A3: demo_warmstart CE below chance (autopsy item 1, third clause) -------------
    a3_per_seed: List[Dict[str, Any]] = []
    for row in per_seed["demo_warmstart"]:
        ce = _num(row.get("demo_ce_window_mean"))
        ref = _num(row.get("demo_ce_chance_reference"))
        ratio = (ce / ref) if (ce is not None and ref is not None and ref > 0.0) else None
        a3_per_seed.append({
            "seed": int(row.get("seed", -1)), "ce_window_mean": ce,
            "chance_reference_ln_k": ref, "ce_over_chance": ratio,
            "n_ce_updates": row.get("demo_n_ce_updates"),
        })
    a3_vals = [(1.0 if d["ce_over_chance"] is None else float(d["ce_over_chance"]))
               for d in a3_per_seed]
    a3_measured = max(a3_vals) if a3_vals else 1.0
    a3_cell = (f"demo_warmstart@seed{a3_per_seed[a3_vals.index(a3_measured)]['seed']}"
               if a3_vals else None)
    a3 = {
        "name": "demo_warmstart_ce_below_chance", "kind": "readiness",
        "description": (
            "demo_warmstart's phase-A cross-entropy, averaged over the final "
            "DEMO_CE_WINDOW_FRAC of its updates, as a RATIO to chance "
            "(ln(mean n_candidates) measured in that same loop), on the WORST seed. "
            "V3-EXQ-1039 measured exactly 1.0 x chance (3.465736 = ln(32)), the signature "
            "of an exactly-zero gradient. A seed with no measurable CE scores 1.0 (fails)."
        ),
        "control": "demo_warmstart phase-A CE trace vs ln(mean candidate-set size)",
        "measured": round(float(a3_measured), 6), "threshold": float(DEMO_CE_CHANCE_FRAC),
        "direction": "upper", "offending_cell": a3_cell,
        "per_seed": a3_per_seed,
        "met": bool(a3_measured <= DEMO_CE_CHANCE_FRAC),
    }

    # ---- B1: first-action diversity on a MAJORITY of ticks (autopsy item 2a) -----------
    b1_measured, b1_cell = _worst(
        cells,
        lambda r: _num((r.get("first_action_class_hist_p1") or {}).get("frac_ge_floor")),
        "min", 0.0,
    )
    b1 = {
        "name": "first_action_diversity_majority_ticks", "kind": "readiness",
        "description": (
            "Fraction of P1 E3 ticks offering >= FIRST_ACTION_CLASS_FLOOR (3, RATIFIED at "
            "the Step 8 user gate 2026-09-16T13:06:22Z over MECH-341's stricter 4.5) "
            "distinct first-action classes, in the WORST treatment cell; must be a strict "
            "MAJORITY. CONVERSION-CAPABILITY half: under proposer-pool collapse a fully "
            "absorbed manipulation still yields identical eval behaviour, so absorption "
            "alone would pass the same vacuous null behind a clean gate. A cell with no "
            "recorded ticks scores 0.0 (fails). P0 is recorded separately and is legitimately "
            "empty -- the shared build's conversion block is P1-gated at source."
        ),
        "control": "_FirstActionClassTap over allon_training._distinct_first_action_classes",
        "measured": round(float(b1_measured), 6),
        "threshold": float(FIRST_ACTION_MAJORITY_FRAC),
        "direction": "lower", "comparator": ">", "offending_cell": b1_cell,
        "met": bool(b1_measured > FIRST_ACTION_MAJORITY_FRAC),
    }

    # ---- B2: eval-side bias-head ablation sensitivity (autopsy item 2b) ----------------
    b2_per_arm: Dict[str, Any] = {}
    b2_counts: List[Tuple[float, str]] = []
    for arm in TREATMENT_ARMS:
        n_diff = 0
        n_measurable = 0
        detail = []
        for row in per_seed[arm]:
            ab = row.get("bias_ablation") or {}
            measurable = bool(ab.get("measurable"))
            differs = bool(ab.get("differs")) and measurable
            n_measurable += 1 if measurable else 0
            n_diff += 1 if differs else 0
            detail.append({"seed": int(row.get("seed", -1)), "measurable": measurable,
                           "differs": differs, "n_differing": ab.get("n_differing"),
                           "n_compared": ab.get("n_compared")})
        b2_per_arm[arm] = {"n_seeds_differing": n_diff, "n_seeds_measurable": n_measurable,
                           "per_seed": detail}
        b2_counts.append((float(n_diff), arm))
    b2_measured, b2_cell = (min(b2_counts) if b2_counts else (0.0, None))
    b2 = {
        "name": "bias_ablation_changes_committed_action", "kind": "readiness",
        "description": (
            "Number of seeds on which zeroing the lateral-PFC bias head changes at least "
            "one committed action in a paired, RNG-matched eval rollout -- in the WORST "
            "treatment arm. RATIFIED at the Step 8 user gate. Asserts only that two action "
            "sequences DIFFER within one process on one machine, never an exact committed "
            "action (torch.multinomial is not portable across machine classes). An "
            "unmeasurable cell counts as NOT differing."
        ),
        "control": "paired trained-vs-zeroed-head eval rollout, identical RNG and env seed",
        "measured": float(b2_measured), "threshold": float(ABLATION_MIN_SEEDS),
        "direction": "lower", "offending_cell": b2_cell,
        "per_arm": b2_per_arm,
        "met": bool(b2_measured >= ABLATION_MIN_SEEDS),
    }

    absorption_met = bool(a1["met"] and a2["met"] and a3["met"])
    conversion_met = bool(b1["met"] and b2["met"])
    readiness = [c0, c0b, c0c, c0d, c0e, a1, a2, a3, b1, b2]

    if not all(p["met"] for p in readiness):
        return {
            "label": "substrate_not_ready_requeue",
            "preconditions": readiness,
            "criteria_non_degenerate": {"C1": False},
            "combination_rule": (
                "C0 AND C0b AND C0c AND C0d AND C0e AND (A1 AND A2 AND A3 = ABSORPTION) "
                "AND (B1 AND B2 = CONVERSION CAPABILITY) -- all readiness, all conjunctive, "
                "all gate before any verdict criterion. Autopsy section 14: items 1 and 2 "
                "are a PAIR; shipping 1 without 2 reproduces the defect being repaired."
            ),
            "absorption_met": absorption_met, "conversion_met": conversion_met,
            "failed_preconditions": [p["name"] for p in readiness if not p["met"]],
            "oracle_mean": round(oracle_mean, 6), "random_mean": round(random_mean, 6),
        }

    # ---- R1 premise + C1 verdict: carried verbatim from V3-EXQ-1039 ---------------------
    sparse_mean = _mean(sparse_v)
    r1_threshold = random_mean + BASELINE_BLOCKED_FRAC * span
    r1 = {
        "name": "sparse_rl_stays_blocked", "kind": "premise",
        "description": (
            "sparse_rl (stock waypoint reward only) stays within BASELINE_BLOCKED_FRAC of "
            "the (oracle - random) span -- the premise this run's null/lift verdict depends "
            "on. Self-calibrated to THIS run's own achievable span."
        ),
        "measured": round(float(sparse_mean), 6), "threshold": round(float(r1_threshold), 6),
        "direction": "upper",
        "met": bool(sparse_mean <= r1_threshold),
    }
    if not r1["met"]:
        return {
            "label": "sparse_baseline_not_blocked",
            "preconditions": readiness + [r1],
            "criteria_non_degenerate": {"C1": False},
            "combination_rule": "R1 unmet -- the blocked-baseline premise is false",
            "absorption_met": absorption_met, "conversion_met": conversion_met,
            "oracle_mean": round(oracle_mean, 6), "random_mean": round(random_mean, 6),
            "sparse_mean": round(sparse_mean, 6),
        }

    def per_seed_lift(dense: List[float]) -> List[float]:
        return [d - s for d, s in zip(dense, sparse_v)]

    lift_margin = LIFT_MARGIN_FRAC * span
    shaped_lift = per_seed_lift(shaped_v)
    demo_lift = per_seed_lift(demo_v)
    shaped_clears = sum(1 for x in shaped_lift if x >= lift_margin)
    demo_clears = sum(1 for x in demo_lift if x >= lift_margin)
    shaped_negative = sum(1 for x in shaped_lift if x <= -lift_margin)
    demo_negative = sum(1 for x in demo_lift if x <= -lift_margin)

    r2_cells = [row["distinct_cells_per_ep"] for _a, _s, row in cells]
    r2 = {
        "name": "treatment_arms_move", "kind": "eval_protocol",
        "description": "Every treatment-arm cell visits >= MIN_DISTINCT_CELLS (not a "
                       "stationary argmax-into-wall eval policy).",
        "measured": round(float(min(r2_cells)) if r2_cells else 0.0, 6),
        "threshold": float(MIN_DISTINCT_CELLS),
        "direction": "lower",
        "met": bool(r2_cells and min(r2_cells) >= MIN_DISTINCT_CELLS),
    }

    c1_clear = (shaped_clears >= MIN_SEEDS) or (demo_clears >= MIN_SEEDS)
    c1_anomalous_negative = (shaped_negative >= MIN_SEEDS) or (demo_negative >= MIN_SEEDS)

    # Autopsy item 4: C1's non-degeneracy is ABSORPTION and CONVERSION-CAPABILITY and EVAL
    # MOBILITY -- not mobility alone, which is all V3-EXQ-1039 asserted.
    criteria_non_degenerate = {
        "C1": bool(r2["met"] and absorption_met and conversion_met),
    }
    preconditions = readiness + [r1, r2]

    if not r2["met"]:
        label = "degenerate_eval_protocol"
    elif c1_clear and not c1_anomalous_negative:
        label = "dense_or_warmstart_signal_converts_h1_supported"
    elif c1_anomalous_negative and not c1_clear:
        label = "dense_or_warmstart_signal_anomalous_negative_lift"
    elif c1_clear and c1_anomalous_negative:
        label = "dense_or_warmstart_signal_inconsistent_across_arms"
    else:
        label = "training_signal_does_not_convert_h1_not_supported"

    return {
        "label": label,
        "preconditions": preconditions,
        "criteria_non_degenerate": criteria_non_degenerate,
        "combination_rule": (
            "C1 = (>=MIN_SEEDS of N seeds clear LIFT_MARGIN_FRAC*span under shaped_rl) OR "
            "(>=MIN_SEEDS under demo_warmstart), evaluated ONLY after C0/C0b/C0c/C0d/C0e "
            "AND the paired absorption gate (A1 AND A2 AND A3) AND the paired conversion-"
            "capability gate (B1 AND B2) AND R1 AND R2 all hold"
        ),
        "note": (
            "The composite label is an OR/AND over BOTH arms and can read "
            "'inconsistent_across_arms' even when ONE arm gave a clean, clearly-directional "
            "result -- read shaped_clears_n_seeds / demo_clears_n_seeds and "
            "shaped_lift_per_seed / demo_lift_per_seed individually before trusting the "
            "composite label as the interpretable quantity."
        ) if label == "dense_or_warmstart_signal_inconsistent_across_arms" else None,
        "load_bearing": True,
        "absorption_met": absorption_met, "conversion_met": conversion_met,
        "oracle_mean": round(oracle_mean, 6), "random_mean": round(random_mean, 6),
        "sparse_mean": round(sparse_mean, 6),
        "shaped_mean": round(_mean(shaped_v), 6), "demo_mean": round(_mean(demo_v), 6),
        "shaped_lift_per_seed": [round(x, 6) for x in shaped_lift],
        "demo_lift_per_seed": [round(x, 6) for x in demo_lift],
        "shaped_clears_n_seeds": int(shaped_clears), "demo_clears_n_seeds": int(demo_clears),
        "min_seeds_required": int(MIN_SEEDS), "n_seeds": int(n),
    }


# ---------------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------------
def _flat_scalar(d: Dict[str, Any]) -> Dict[str, float]:
    """Flat numeric projection for the runpack converter / indexer.

    Two encoding rules, both load-bearing: booleans go out as 0/1 INTS (the indexer's
    `_is_number` excludes `bool`, so a raw True is recorded and invisible), and a
    non-finite or None value is DROPPED rather than emitted (a nan IS numeric to the
    indexer and would pollute a delta; an absent key correctly reads as unmeasured).
    """
    out: Dict[str, float] = {}
    for k, v in d.items():
        if isinstance(v, bool):
            out[k] = 1 if v else 0
        elif isinstance(v, (int, float)) and math.isfinite(float(v)):
            out[k] = float(v)
    return out


def main(dry_run: bool = False, probe: bool = False) -> Dict[str, Any]:
    if dry_run:
        seeds = DRY_SEEDS
        zworld_p0, p0_warm, rl_eps = DRY_ZWORLD_P0, DRY_P0_WARMUP, DRY_RL
        n_demo, eval_eps, steps = DRY_DEMO, DRY_EVAL, DRY_STEPS
        abl_eval = DRY_ABLATION_EVAL
    elif probe:
        seeds = [42]
        zworld_p0, p0_warm, rl_eps = 8, 10, 15
        n_demo, eval_eps, steps = 8, 5, 30
        abl_eval = 2
    else:
        seeds = SEEDS
        zworld_p0, p0_warm, rl_eps = ZWORLD_P0_EPISODES, P0_WARMUP_EPISODES, RL_EPISODES
        n_demo, eval_eps, steps = N_DEMO_EPISODES, EVAL_EPISODES, STEPS_PER_EPISODE
        abl_eval = ABLATION_EVAL_EPISODES

    t0 = __import__("time").perf_counter()
    per_seed: Dict[str, List[Dict[str, Any]]] = {arm: [] for arm in ARM_ORDER}
    arm_results: List[Dict[str, Any]] = []

    for seed in seeds:
        for arm_id in ARM_ORDER:
            print(f"Seed {seed} Condition {arm_id}", flush=True)
            cfg_slice = _arm_config_slice(arm_id, zworld_p0, p0_warm, rl_eps, n_demo,
                                          eval_eps, steps)
            with arm_cell(seed, config_slice=cfg_slice, script_path=Path(__file__)) as cell:
                if arm_id in TREATMENT_ARMS:
                    row = _run_treatment_cell(arm_id, seed, zworld_p0, p0_warm, rl_eps,
                                              n_demo, eval_eps, steps, abl_eval)
                else:
                    row = _run_anchor_cell(arm_id, seed, eval_eps, steps)
                cell.stamp(row)
            row["arm_id"] = arm_id
            row["seed"] = int(seed)
            per_seed[arm_id].append(row)
            arm_results.append(row)
            # Signals "this (seed, arm) cell's run completed", per the runner's progress
            # convention -- NOT a scientific per-cell verdict (there is no natural per-cell
            # pass/fail here; the actual verdict is the cross-arm comparison in _score,
            # printed once below). Matches V3-EXQ-1039/1030's identical convention.
            print("verdict: PASS", flush=True)

    verdict = _score(per_seed, seeds)
    elapsed = __import__("time").perf_counter() - t0

    non_degen = check_degeneracy({
        "waypoints_visited_per_ep_spread": {
            "values": [row["waypoints_visited_per_ep"] for row in arm_results],
        },
    })

    outcome = "PASS" if verdict["label"] not in (
        "substrate_not_ready_requeue", "sparse_baseline_not_blocked",
        "degenerate_eval_protocol",
    ) else "FAIL"

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{ts}_v3"

    readout = _flat_scalar({
        "oracle_mean_visits_per_ep": verdict.get("oracle_mean"),
        "random_mean_visits_per_ep": verdict.get("random_mean"),
        "sparse_mean_visits_per_ep": verdict.get("sparse_mean"),
        "shaped_mean_visits_per_ep": verdict.get("shaped_mean"),
        "demo_mean_visits_per_ep": verdict.get("demo_mean"),
        "shaped_clears_n_seeds": verdict.get("shaped_clears_n_seeds"),
        "demo_clears_n_seeds": verdict.get("demo_clears_n_seeds"),
        "min_seeds_required": verdict.get("min_seeds_required"),
        "absorption_gate_met": verdict.get("absorption_met"),
        "conversion_gate_met": verdict.get("conversion_met"),
        # Every readiness/premise gate's own measured value + bar, flat, so the verdict is
        # re-derivable from the scored pack without opening interpretation.preconditions.
        **{f"{p['name']}_measured": p.get("measured")
           for p in verdict.get("preconditions", [])},
        **{f"{p['name']}_threshold": p.get("threshold")
           for p in verdict.get("preconditions", [])},
        **{f"{p['name']}_met": p.get("met")
           for p in verdict.get("preconditions", [])},
    })

    manifest: Dict[str, Any] = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "supersedes": SUPERSEDES,
        "architecture_epoch": ARCHITECTURE_EPOCH,
        "claim_ids": CLAIM_IDS,
        "evidence_direction": "unknown",
        "evidence_direction_per_claim": {"INV-086": "non_contributory",
                                         "MECH-428": "non_contributory"},
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "outcome": outcome,
        "timestamp_utc": ts,
        "sleep_driver_pattern": "none",
        "interpretation": {
            "label": verdict["label"],
            "preconditions": verdict.get("preconditions", []),
            "criteria_non_degenerate": verdict.get("criteria_non_degenerate", {}),
        },
        "combination_rule": verdict.get("combination_rule"),
        "absorption_gate_met": verdict.get("absorption_met"),
        "conversion_gate_met": verdict.get("conversion_met"),
        "failed_preconditions": verdict.get("failed_preconditions"),
        "readout": readout,
        "arm_results": arm_results,
        "per_seed": per_seed,
        "n_seeds": len(seeds),
        "budget": {
            "zworld_p0_episodes": zworld_p0, "p0_warmup_episodes": p0_warm,
            "rl_episodes": rl_eps, "n_demo_episodes": n_demo,
            "eval_episodes": eval_eps, "steps_per_episode": steps,
            "ablation_eval_episodes": abl_eval,
        },
        "instrument_provenance": {
            "shared_build": (
                "sd-allon-training-signal-absorption-telemetry (ree-v3 996dec30) -- this "
                "driver CONSUMES the absorption/conversion blocks returned by "
                "_train_all_on_agent and recomputes none of them."
            ),
            "driver_side_addition": (
                "_FirstActionClassTap only: a per-tick histogram of distinct first-action "
                "classes, needed because the ratified gate is a MAJORITY (median/quantile) "
                "statement that n/mean/sd/min/max cannot express. Read-only pass-through, "
                "no RNG draw, module attribute restored in a finally."
            ),
        },
    }
    manifest.update(non_degen)

    out_path = write_flat_manifest(
        manifest, dry_run=dry_run,
        config={
            "seeds": seeds, "budget": manifest["budget"],
            "shaping_coef": SHAPING_COEF,
            "lift_margin_frac": LIFT_MARGIN_FRAC,
            "min_achievable_span": MIN_ACHIEVABLE_SPAN,
            "oracle_floor": ORACLE_FLOOR,
            "baseline_blocked_frac": BASELINE_BLOCKED_FRAC,
            "min_distinct_cells": MIN_DISTINCT_CELLS,
            "return_separation_sd": RETURN_SEPARATION_SD,
            "return_separation_cap": RETURN_SEPARATION_CAP,
            "return_sd_eps": RETURN_SD_EPS,
            "adv_surviving_frac_floor": ADV_SURVIVING_FRAC_FLOOR,
            "demo_ce_chance_frac": DEMO_CE_CHANCE_FRAC,
            "demo_ce_window_frac": DEMO_CE_WINDOW_FRAC,
            "first_action_class_floor": FIRST_ACTION_CLASS_FLOOR,
            "first_action_majority_frac": FIRST_ACTION_MAJORITY_FRAC,
            "ablation_min_seeds": ABLATION_MIN_SEEDS,
            "head_delta_floor": HEAD_DELTA_FLOOR,
            "summary_degenerate_frac_ceiling": SUMMARY_DEGENERATE_FRAC_CEILING,
            "lateral_pfc_capture_head_diagnostics": True,
            "alpha_world": ALPHA_WORLD, "alpha_self": ALPHA_SELF,
        },
        seeds=seeds, script_path=Path(__file__), started_at=t0, elapsed_seconds=elapsed,
        z_goal_stream_stats=_ZG.stats(),
    )

    print(f"label: {verdict['label']}", flush=True)
    if verdict.get("failed_preconditions"):
        print("failed preconditions: " + ", ".join(verdict["failed_preconditions"]),
              flush=True)
    print(f"absorption_gate_met: {verdict.get('absorption_met')}", flush=True)
    print(f"conversion_gate_met: {verdict.get('conversion_met')}", flush=True)
    print(f"outcome: {outcome}", flush=True)

    return {"outcome": outcome, "manifest_path": str(out_path)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--probe", action="store_true",
                        help="reachability check only -- NOT evidence, see queue note")
    args = parser.parse_args()
    _res = main(dry_run=args.dry_run, probe=args.probe)
    _o = str(_res["outcome"]).upper()
    emit_outcome(outcome=_o if _o in ("PASS", "FAIL") else "FAIL",
                 manifest_path=_res.get("manifest_path"), dry_run=args.dry_run)
