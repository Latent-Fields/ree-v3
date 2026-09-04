"""V3-EXQ-993a -- ARC-021/MECH-069 merged-channel ablation, ACTION-CONDITIONED
uncompressed harm readout. SUPERSEDES V3-EXQ-993.

Claims: ARC-021 (architectural_commitment -- "Three BG-like cortico-striatal
loops require distinct learning channels"), MECH-069 (mechanism_hypothesis --
"Sensory prediction error, motor-sensory error, and harm/goal error are
incommensurable and cannot be collapsed"). Proposal EXP-0528 / EVB-1248,
dispatch_mode=targeted_probe.

EXT-003 IS DELIBERATELY NOT TAGGED (change from V3-EXQ-993, flagged to
governance for ratification). EXT-003 is a reward-hacking rider -- "scalar
reward conflates incommensurable error signals" so an agent exploits the
conflation. This design has NO reward, NO policy optimisation and NO agent
acting to maximise anything: it is a supervised prediction study under a
uniform-random policy. The MERGED arm does instantiate the *mechanism* half
(one scalar objective over three error terms), but nothing here can hack
anything, so an EXT-003 evidence tag would let a claim-scoring consumer read
"reward hacking tested" off a design with no reward. 993 tagged it and got
`non_contributory` on all three, so dropping it costs EXT-003 nothing.

=== WHY 993 FAILED, AND WHY THE CHIP'S PROPOSED FIX IS NECESSARY BUT NOT
SUFFICIENT (measured this session, 2026-09-04) ===

993 FAILed `separated_baseline_signal_absent_both_conditions`: the SEPARATED
control arm produced max |calibration_gap| 0.00152 over 12 cells against
SEPARATED_SIGNAL_FLOOR 0.02, a 13.1x shortfall, negative in 4 of 6 baseline
cells. The confirmed cluster autopsy
(evidence/planning/failure_autopsy_ext-claim-probe-cluster_2026-09-03.md,
section 4, routing section 6) localised the compression to the DRIVER-LOCAL
harm head -- "the harm head compresses two well-separated predicted latents
onto near-identical sigmoids" -- and governance-20260903T2013 amendment 2
confirmed no substrate entry exists or should. The prescribed fix was to read
the PRE-SIGMOID logit instead.

That fix was measured against 993's own configuration (probe:
_scratch/campaign_c2a_993a/, SEPARATED arm only, full scale 80+80 episodes x
120 steps, PROBE_RESETS=15, 4 seeds x 2 conditions). It is real but it is not
enough:

  readout                              max|gap|   range    control cells +ve
  sigmoid (993's own)                  0.00152    0.00246  4 of 8
  pre-sigmoid logit (the chip's fix)   0.11871    0.17914  2 of 8
  action-conditioned logit             0.21466    0.13294  16 of 16
  ... + the three red-team repairs      0.71546    0.48689  24 of 24

(The last row is the design as queued, re-measured through the driver's own
production path; the row above it is the same readout before the F5/F7/F8
repairs and before that provenance correction, retained to show what each step
bought. Per-cell |t| for the pre-sigmoid row: 0.31, 0.65, 0.89, 0.89, 0.99,
1.71, 1.93, 2.42 -- six of eight inside 2 SE of zero.)

The probe reproduces 993's reported max |calibration_gap| of 0.00152 EXACTLY
in sigmoid space, which is the construct check that the probe measures the
same quantity the run did. Reading the pre-sigmoid logit multiplies the
magnitude by 78x -- and reveals that the underlying quantity is NOISE: 6 of 8
control cells sit within 2 standard errors of zero (|t| = 0.31, 0.65, 0.89,
0.89, 0.99, 1.71), 6 of 8 are negative, i.e. anti-calibrated. A bar set inside
that "range" would be a bar set inside sampling noise.

THE ROOT CAUSE IS DEEPER THAN THE SIGMOID, and it is a training-target defect,
not a readout one. 993 trained `harm_head(z_t1) -> harm_label` -- the harm
label attached to the state AFTER the transition. But
`ree_core/environment/causal_grid_world.py` line 2745 executes
`self.grid[new_x, new_y] = self.ENTITY_TYPES["agent"]` on every committed
move, so when the agent steps onto a hazard the hazard entity is OVERWRITTEN
in the grid, and `SmallViewEnv.world_state` carries only the local entity view
plus the contamination view (never body state, never health). The post-harm
observation therefore has the harm cue ERASED. Measured consequence: the
harm head's ability to separate real-harm z_t1 from non-harm z_t1
(`harm_discrim_logit`) is 0.042 and -0.029 in DENSE -- undiscriminating, and
on one seed pointing the wrong way. 993's head was taught that post-harm
states look safe, so no readout applied to it can carry a forward-looking
causal signature.

A second, independent break was measured alongside: 993 evaluated the harm
head on `predict_forward(z, a)`, an MSE-regressed prediction whose error
(`pred_err_l2`) is 0.67-0.87x the entire spread of real latents
(`reach_ratio` 1.15-1.49), i.e. the prediction lands about as far from the
true next latent as two random latents are apart. In SPARSE, where the harm
head DOES discriminate strongly (harm_discrim_logit 1.51-2.98, because SPARSE
harm labels are dominated by CONTAMINATION events whose cue survives in the
contamination view), the causal signature is STILL null -- which localises
that second break to the forward-prediction stage rather than to the readout.

Two repairs were measured and REJECTED before the one below was adopted:
  (a) pos_weight label balancing on the 993 harm head: inert. Per-sample Adam
      normalises the update, so scaling the loss on a ~2% minority of steps
      moved the gap by <1e-4 (0.00047594070 -> 0.00047594220 on DENSE/101).
  (b) a naive env-lookahead (move the agent pointer to the destination cell
      and read the harm logit there): large (max|gap| 1.26, |t| up to 11.9)
      but CONSTRUCT-INVALID and consistently ANTI-calibrated, precisely
      because it reads a view with `hazard` at the centre -- a token the head
      never sees in training, since step() overwrote it. This is the
      V3-EXQ-991 out-of-distribution failure class and was discarded on that
      basis, not on its effect size.

=== THE FIX ADOPTED: AN ACTION-CONDITIONED HARM HEAD, READ IN DISTRIBUTION ===

`harm_logit(z_t, a_t) -> logit P(harm on this transition)`, trained on exactly
the distribution it is read on, and read PRE-SIGMOID:

    causal_sig      = harm_logit(z_t, a_actual) - harm_logit(z_t, a_cf)
    calibration_gap = mean(causal_sig | near_hazard) - mean(causal_sig | safe)

This keeps the SD-003 counterfactual shape of V3-EXQ-007-010 and of 993 (same
state, actual action versus a distinct counterfactual one) while removing both
measured breaks: the label now attaches to the PRE-transition state, where the
hazard cue is still visible, and the forward head is no longer in the readout
path so its MSE smoothing cannot annihilate the contrast.

Three further repairs came out of the Step 4.5 red-team pass and are load-
bearing, not cosmetic -- together they raised the control arm's signal 3.3x:

  F7 IN-DISTRIBUTION PROBES. `step()` vacates the old cell and writes
     grid[new] = agent (causal_grid_world.py:2453 and :2745), so every TRAINING
     view has the agent token at the 3x3 centre. 993's probe moved only
     agent_x/agent_y, leaving the centre reading empty/resource and the stale
     agent token elsewhere in the view -- out of distribution at exactly the
     position the encoder is most sensitive to, which is the same objection
     this driver uses to reject the env-lookahead readout. `_eval_probes` now
     reproduces the grid state `step()` would have produced and restores it in
     a `finally`, so probe order stays irrelevant.
  F8 MOVEMENT-ONLY COUNTERFACTUAL. 993 drew the counterfactual from all five
     actions including STAY while the actual was always a movement, so with
     p=1/4 the contrast was P(harm|move) - P(harm|STAY). Measured effect: in
     SPARSE the near-hazard mean causal_sig was itself <= 0 on 3 of 4 seeds,
     i.e. the positive gap was carried by the safe population's STAY
     counterfactual rather than by hazard-approach discrimination.
  F5 DEPTH- AND CLIP-MATCHED ARMS. MERGED's every head reads a 2-hidden-layer
     trunk plus a linear out; 993 gave SEPARATED 2-layer heads and clipped each
     head separately at 1.0 against MERGED's single union clip. The arms
     therefore differed in depth, capacity and clipping as well as in sharing,
     so a "supports" could have been joint-clip starvation and a "weakens"
     could have been extra MERGED capacity. SEPARATED now has matched depth and
     width and one union clip; parameter SHARING and loss SUMMATION are the only
     remaining differences, which is exactly ARC-021's content. Input dimensions
     still differ where the ROLES differ (SEPARATED's sensory head does not see
     the action) -- that asymmetry is the separate-channels premise, not a
     confound.

=== THE CONTROL-ARM MEASUREMENT THE BARS ARE DERIVED FROM ===

Measured through THIS DRIVER'S OWN production path -- `_run_cell(condition,
"SEPARATED", seed)`, `arm_cell`/`reset_all_rng` included -- not through a probe
subclass. (The design probe's subclass consumed the torch RNG differently before
head construction, so its numbers were a different head-init draw; red-team F3.
The MERGED arm was deliberately never run.)

24 control cells = 3 independent head-init draws x 4 seeds x 2 conditions:
  calibration_gap  min 0.22857  max 0.71546  RANGE 0.48689  mean 0.46590
                   sd 0.14184   POSITIVE IN 24 OF 24
  per-draw condition means 0.3986 .. 0.5312 (8 condition-means, all > 0.39)
  harm-head action sensitivity, worst cell 0.11382

PAIRED NULL (same seed/condition/P0; ONLY the head init differs), all three
draw-pairs, n=24: mean +0.0216, sd 0.1659, max |d| 0.28432. The three pairwise
means are +0.0857, +0.0324, -0.0533 -- the sign is NOT consistent, so the null
is centred on zero. (This corrects the design probe's apparent +0.02248 / 7-of-8
directional bias, which red-team F4 flagged as having no structural mechanism
and which was indeed an artefact of that probe's asymmetric reseeding.)

=== PRE-REGISTERED THRESHOLDS, EACH DERIVED FROM THAT MEASUREMENT ===

MARGIN = 0.15. Sized against the measured paired null by Monte Carlo of the
FULL criterion (mean of 4 paired diffs <= -MARGIN AND all 4 <= 0, per-cell noise
Normal(0, 0.1659)): per-condition false-positive rate 2.1%, and because a PASS
requires BOTH conditions, 0.045% overall. It is 31% of the control arm's 0.48689
demonstrated range and 66% of its 0.22857 floor headroom.

SEPARATED_SIGNAL_FLOOR = 0.20, on the control arm's per-condition MEAN. Half the
weakest measured per-draw condition mean (0.3986). Deliberately ABOVE MARGIN so
it stays INDEPENDENT of H1 (red-team F10: with FLOOR below MARGIN, H1 passing
implied non-degeneracy passing, so the floor could only ever fail on coverage,
never on weak signal). At 0.20 a control arm with every cell at 0.16 passes H1
and correctly fails non-degeneracy.

HARM_ACTION_SENSITIVITY_FLOOR = 0.05. Replaces 993's FORWARD_SENSITIVITY_FLOOR:
the forward head is no longer on the readout path, so a precondition on it would
certify a component the criteria never read (the V3-EXQ-643 same-statistic
defect). Worst measured control cell 0.1138 -> 2.3x.

Joint satisfiability, checked on paper (red-team family 2): non-degeneracy needs
mean SEP >= 0.20 and the criterion needs mean MERGED <= mean SEP - 0.15, i.e.
<= +0.05 at the floor and <= +0.32 at the measured mean. Both reachable
everywhere in the measured range; no arithmetic contradiction.

=== DV_HEADROOM PRECONDITIONS (the thing 993 lacked; ree-v3 8e133d26ed) ===

Built with `_metrics.dv_headroom_check(...)` and routed through
`p0_readiness_gate`, so an unmet one self-routes to `substrate_not_ready_requeue`
and never to a verdict on ARC-021/MECH-069. Both are measured at RUNTIME from the
control arm's own realised values; the design measurement only sets expectations.
They are preceded by `control_arm_coverage_complete` (red-team F6), which
requires every INTENDED control cell to have produced a usable calibration_gap --
otherwise the headroom checks would certify over the realised n and a short
control arm could pass H1, train the whole MERGED arm, and only then fail
non-degeneracy on the short condition.

  H1 dv_headroom_margin_room_below_control
     dv_name=calibration_gap, criterion_threshold=MARGIN, margin=1.0,
     statistic="floor_headroom", dv_bounds=(0.0, 1.0).
     floor_headroom = min(control) - 0.0 is exactly the room the MERGED arm has
     to FALL, which is what a suppression-direction criterion consumes. 0.0 is
     used as the bottom because "no hazard-conditional harm calibration at all"
     is the meaningful floor; the logit difference is formally unbounded below,
     so this UNDERSTATES the room and is conservative.
     Measured on the three draws: 1.52x, 2.37x, 1.71x.
  H2 dv_headroom_control_signal_floor_reachable
     dv_name=calibration_gap, criterion_threshold=SEPARATED_SIGNAL_FLOOR,
     margin=2.0, statistic="max_abs". The 2026-09-03 cluster autopsy's own table
     pairs 993's max |calibration_gap| against SEPARATED_SIGNAL_FLOOR, so
     max_abs is the sanctioned statistic here.
     Measured on the three draws: 1.79x, 1.73x, 1.70x.

The gate runs over the SEPARATED cells BEFORE any MERGED cell is trained, so an
unmet gate costs half the grid rather than all of it.

=== HOW A NOT-READY RESULT IS RECORDED (red-team F1 -- the important one) ===

`build_experiment_indexes.py` computes `direction_explicitly_set =
bool(manifest["evidence_direction_note"])` (line 1871) and then REWRITES any
non-explicit `"unknown"` on a FAIL to `"weakens"` (lines 3519-3520).
`non_degenerate`/`scoring_excluded` does not rescue this -- the gap register's
conflict_ratio counts entries regardless of scoring_excluded. V3-EXQ-993 emitted
a bare `"unknown"` and its autopsy had to hand-correct the record to
`non_contributory`. So every branch here that makes NO comparison emits
`evidence_direction: "non_contributory"` (an allowed value the indexer does not
rewrite) AND an `evidence_direction_note`, which are two independent reasons an
instrument-not-ready run cannot be logged as evidence against either claim.
Verified by simulating the indexer rule on the smoke manifest.

=== PRE-ROUTING CHECK C1 (the autopsy required this verdict be recorded) ===

Three ARC-021/MECH-069-family drivers exist on disk. Verdict: none covers this
question, and two carry exactly the defect this redesign repairs.
  - `v3_exq_004_arc021_incommensurability.py` -- same probe-based
    calibration_gap, separate-vs-collapsed optimizers, but the SAME
    sigmoid-compressed post-transition readout and a pre-registered C1 bar of
    0.05 chosen with no headroom measurement. No manifest. Running it as written
    would reproduce 993's failure with a bar 20x the measured sigmoid-space
    range. NOT a substitute.
  - `v3_exq_005_mech069_error_scale.py` -- measures loss-channel SCALE ratios and
    pairwise correlations. A different question (are the three error signals
    incommensurable in magnitude and covariance) from this one (does MERGING them
    degrade harm calibration). No manifest. NOT a substitute.
  - `v3_spark_arc021_three_loop_scale.py` -- closest in spirit
    (`e3_discrim_separate > e3_discrim_merged + 0.02` at world_dim=128) but reads
    a discrimination statistic rather than the SD-003 causal signature ARC-021's
    `what_would_answer` names, and its 0.02 bar is likewise unmeasured. No
    manifest. NOT a substitute.

=== GOV-REUSE-1 (Step 2.4) ===

Decisive readout: `calibration_gap` of an ACTION-CONDITIONED pre-sigmoid harm
logit under a shared-trunk/single-optimizer merge. Checked
`v3_exq_035_mech069_optimizer_merge_20260318T202626Z_v3` -- the one recorded
manifest in this family, MECH-069 + SD-003, FAIL/mixed, which DOES carry
`calibration_gap_approach_separated` -0.02819 vs `_merged` -0.09331. Not
recoverable for three independent reasons: (1) it predates the Experimental
Recording Standard and carries NO `substrate_hash` at any level, so substrate
compatibility cannot be confirmed and the skill directs treating that as
not-recoverable; (2) its readout is the agent-substrate post-transition sigmoid
form, the exact quantity measured here to be noise; (3) both its arms are
NEGATIVE, i.e. anti-calibrated, so it carries no usable control-arm signal. Also
checked `v3_exq_004/005/spark` -- no manifests at all. Not recoverable -> run.

=== SUBSTRATE READINESS (Step 2.5 / 2.5a / 2.5b / 2.5c) ===

2.5/2.5a. Substrate footprint is `ree_core/environment/causal_grid_world.py`
ONLY. As in 993, this driver imports no `ree_core.agent`/`REEAgent` and no
E1/E2/E3 predictor module: the manipulation under test is OPTIMIZER AND
PARAMETER-SHARING TOPOLOGY, not any specific existing predictor implementation,
so the three channel heads are legitimately experiment-local (as `SmallViewEnv`
is experiment-local env code). `ree_core/latent/stack.py` is not imported. The
empirical confirmation for 2.5a is unusually strong: the full design was measured
end-to-end at production scale before authoring, including the
`grid[new,new] = agent` overwrite at causal_grid_world.py:2745 that the new
training target exists to route around, and the :2453/:2421 vacate behaviour the
F7 probe repair reproduces.

2.5b re-derive brake: 0 for ARC-021, 0 for MECH-069. The 993 autopsy declares
`recommended_epistemic_category_per_claim` = "standard" for both (explicitly: "a
readout dynamic-range ceiling is measurement debt, not a substrate ceiling on the
claims"), so it does not count toward the brake. Brake does not fire.

2.5c substrate-path overlap. Three open entries name `causal_grid_world.py`.
`SD-MECH303-THRESHOLD-SOURCING` (degrading) -- inert, this driver never reads
`contextual_safety_harm_threshold` or any MECH-303 field.
`mech357-freeze-incompatible-pressure-mechanism` (degrading) -- inert, no Stage-H
onboarding or hazard-pursuit machinery here. `waypoint-proximity-field-observable`
(severity CORRUPTING, status `implemented_pending_validation`, therefore OPEN) is
scoped to `CausalGridWorld._get_observation_dict`, which `SmallViewEnv` overrides
in full; `self._get_observation_dict()` dispatches to that override from both
`reset()` and `step()`, and this driver never enables `subgoal_mode`, constructs
no waypoints and has no navigation DV -- so the defect ("waypoints invisible
beyond the local view, every subgoal_mode navigation DV pinned at random-walk
level") has nothing in this design to corrupt. Source-verified non-reach, not a
hand-wave.

2.6 ethics preflight: all involvement flags false, decision `allow`. No
negative-valence drive, no suffering accumulator, no self-model, no replay -- a
supervised prediction study under a random policy (SENT-0).

=== DESIGN ===

Two hazard-density conditions (DENSE: 12x12 grid, 15 hazards, 3x3 local view;
SPARSE: identical 12x12/3x3, 4 hazards), because ARC-021's falsifying-signature
text calls for "both dense- and sparse-hazard conditions". Two arms, identical in
every respect except optimizer/parameter topology:

  ARM_SEPARATED: three independent heads (SensoryHead, ForwardHead,
    action-conditioned HarmHead), three independent Adam instances, three
    independent losses, three disjoint parameter sets. No gradient from any loss
    reaches another channel's parameters. Depth-, width- and clip-matched to
    MERGED (F5 above).
  ARM_MERGED: one shared MLP trunk consuming (z_world, action_onehot); all three
    heads read off its hidden representation; ONE Adam over trunk+heads; ONE
    combined loss backpropagated jointly every step.

Both arms share the same world_encoder/world_decoder architecture, warmed up
identically in P0 (phased training, MANDATORY: P0 encoder warmup -> P1
frozen-encoder head training on .detach()ed z_world -> P2 eval probes), the same
env config per (condition, seed), and a full `reset_all_rng(seed)` at cell entry
via `arm_cell`, so P0 is bit-identical across arms at a given (condition, seed)
and the only degree of freedom is P1's head/optimizer topology. Cells are
order-independent and `reuse_eligible`.

SEEDS = 4 (101, 202, 303, 404) -- up from 993's 3. The criterion requires
sign-consistency across ALL seeds, so a 4th seed makes it strictly harder to
pass, and these are the exact seeds the control-arm range above was measured on.

DV-SYMMETRY (Step 3 mandatory declaration, per arm). The DV `calibration_gap` is
a continuous mean-difference of an unbounded real scalar (a logit difference),
computed over two geometrically distinct probe populations that are never pooled
then split. For BOTH arms: the manipulation is not a uniform additive constant
broadcast across candidates (nothing computes an argmax or softmax-sample over
candidates); it is not a monotone rescaling of an existing ranking (the DV is
never ranked or thresholded before being averaged); and it is not a permutation
of interchangeable units (each cell's heads are trained from scratch, and
MERGED's shared trunk is not a permutation-symmetric function of anything
SEPARATED also computes -- different parameter topology, different gradient
graph). The manipulation changes which parameters receive gradient from which
loss term, which propagates into `harm_logit`'s learned function and can move
calibration_gap in either direction. ARM_SEPARATED is the control and its
non-invariance is the same argument. Neither arm is scoped out.

PRECONDITIONS (positive controls, Step 2.5a / P0 readiness-assert), all five
readiness-kind, all on the SAME machinery the load-bearing criteria route
through:
  0. control_arm_coverage_complete -- every intended control cell produced a
     usable calibration_gap (red-team F6).
  1. harm_head_action_sensitivity_present -- worst cell across the CONTROL arm,
     mean |harm_logit(z,a1) - harm_logit(z,a2)| over a fixed probe batch. Scoped
     to the control arm on purpose (red-team F2): a MERGED harm head gone
     action-blind is the EXTREME FORM of the degradation ARC-021 predicts, so
     gating on it would convert the strongest possible positive result into
     "substrate not ready". MERGED's own sensitivity is recorded as a diagnostic.
  2. p1_harm_events_observed_per_condition -- MIN across the two conditions (993
     red-team fix F7: a pooled floor let SPARSE train on near-zero labelled data
     while DENSE's volume masked it).
  3./4. H1 and H2 above.

LOAD-BEARING CRITERIA (pre-registered, per condition, PAIRED per seed).
C1_dense / C2_sparse: within a condition, for every seed present in BOTH arms,
diff = MERGED - SEPARATED; the criterion passes only if the condition is
non-degenerate (all 4 seeds paired AND mean SEPARATED calibration_gap >=
SEPARATED_SIGNAL_FLOOR) AND mean(diffs) <= -MARGIN AND every seed's diff <= 0
(sign-consistency, so one outlier seed cannot carry the mean).

COMBINATION RULE (pre-registered): PASS iff C1_dense AND C2_sparse when BOTH
conditions clear non-degeneracy ("supports"; "mixed" if exactly one criterion
fires, "weakens" if neither). If NEITHER condition is non-degenerate the run is
`non_degenerate: false` / `non_contributory`. If exactly ONE is, the outcome is
that condition's criterion alone, labelled
`single_condition_only_partial_evidence_<condition>`, with evidence_direction
capped at "mixed" (993 red-team second-pass finding 3).

WHAT A NULL DOES AND DOES NOT EXCLUDE (red-team F9, stated rather than papered
over). With H1/H2 met the control arm demonstrably has room for the MERGED arm to
fall by MARGIN, so `merged_channel_hypothesis_not_supported` is a real `weakens`
for ARC-021 rather than a readout ceiling. But MARGIN is 0.15 against a control
mean of ~0.47, so a `weakens` excludes a degradation of >= 0.15; it does NOT
exclude a smaller, real, sign-consistent one. The per-condition mean paired diff
and every per-seed diff are recorded in `interpretation.condition_detail` so a
sub-margin effect is visible to a later reader rather than being erased by the
label. ARC-021 stays `provisional` and MECH-069 stays `stable` regardless: this
run tests the collapse-degrades-E3 consequence, not MECH-069's scale/temporal
content (that is v3_exq_005's question), so it cannot on its own demote a stable
claim.

RED-TEAM VERDICT (Step 4.5, model fable, 2026-09-04): CONTESTED, 10 findings,
every one dispositioned. FIXED after verifying the citation at source: F1
(indexer rewrites a bare unknown+FAIL to `weakens`; confirmed at
build_experiment_indexes.py:1871 and :3519-3520 -- now emits `non_contributory`
plus an `evidence_direction_note`), F2 (precondition scoped to the control arm),
F5 (depth/clip-matched arms), F6 (coverage-complete gate), F7 (in-distribution
probes), F8 (movement-only counterfactual), F10 (FLOOR raised above MARGIN so
non-degeneracy is independent of H1). MEASURED AND RESOLVED: F3 (the design
probe's head-init draw differed from the driver's -- the whole control-arm
measurement was redone through the production path; the reviewer's arithmetic
that the anchor cell's |t| was 1.84 rather than >= 2.80 was correct and the old
figure is withdrawn), F4 (the paired null's apparent directional bias had no
mechanism -- a third draw showed the sign flips, so it was an artefact of the old
probe and the null is centred). RECORDED, NOT FIXED: F9 (a pre-registered margin
necessarily absorbs sub-margin degradation -- stated above, with the per-seed
diffs recorded so it is visible). The three arm/DV repairs raised the control
arm's demonstrated signal 3.3x (0.129 -> 0.466 mean), which is why the bars above
are re-derived rather than carried over.

SLEEP DRIVER: not applicable -- no sleep loop used in this probe.
"""
from __future__ import annotations

import argparse
import random
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ree_core.environment.causal_grid_world import CausalGridWorld  # noqa: E402
from experiment_protocol import emit_outcome  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from experiments._metrics import (  # noqa: E402
    P0NotReady,
    dv_headroom_check,
    p0_readiness_gate,
)
from experiments.pack_writer import write_flat_manifest  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_993a_arc021_merged_channel_action_conditioned_harm"
EXPERIMENT_PURPOSE = "evidence"
# EXT-003 deliberately dropped -- see the docstring. Flagged to governance.
CLAIM_IDS: List[str] = ["ARC-021", "MECH-069"]
SUPERSEDES = "V3-EXQ-993"

SEEDS = [101, 202, 303, 404]
CONDITIONS = ["DENSE", "SPARSE"]
ARMS = ["SEPARATED", "MERGED"]
NUM_HAZARDS = {"DENSE": 15, "SPARSE": 4}
GRID_SIZE = 12
OBS_RADIUS = 1
OBS_VIEW_SIZE = 2 * OBS_RADIUS + 1  # 3x3

WORLD_DIM = 32
HIDDEN_DIM = 64
P0_EPISODES = 80
P1_EPISODES = 80
STEPS_PER_EPISODE = 120
TOTAL_EPISODES_PER_CELL = P0_EPISODES + P1_EPISODES
PROBE_RESETS = 15
STAY_ACTION_IDX = 4  # env.ACTIONS[4] == (0, 0)

LR_HEAD = 1e-3
LR_ENCODER = 1e-3

MARGIN = 0.15                       # C1/C2: MERGED must be at least this much LOWER.
                                    # Sized against the MEASURED paired null (head
                                    # init only; 3 draws x 4 seeds x 2 conditions,
                                    # mean +0.0216, sd 0.1659, sign NOT consistent
                                    # across draw pairs -> centred on zero). Monte
                                    # Carlo of the full criterion under that null
                                    # (mean of 4 <= -MARGIN AND all 4 <= 0):
                                    # per-condition false-positive 2.1%, and since a
                                    # PASS needs BOTH conditions, 0.045% overall.
                                    # 31% of the control arm's 0.48689 range and 66%
                                    # of its 0.22857 floor headroom.
SEPARATED_SIGNAL_FLOOR = 0.20       # non-degeneracy per condition, on the control
                                    # arm's MEAN calibration_gap. Half the weakest
                                    # measured per-draw condition mean (0.3986).
                                    # Deliberately > MARGIN so this stays INDEPENDENT
                                    # of H1 (red-team F10: at the previous values H1
                                    # implied non-degeneracy, so the floor could only
                                    # ever fail on coverage, never on weak signal).
HARM_ACTION_SENSITIVITY_FLOOR = 0.05  # precondition 1. Replaces 993's
                                    # FORWARD_SENSITIVITY_FLOOR: the forward head is no
                                    # longer on the readout path, so a precondition on
                                    # it would certify a component the criteria never
                                    # read (the V3-EXQ-643 same-statistic defect).
                                    # Worst measured control cell 0.1138 -> 2.3x.
HARM_EVENTS_FLOOR_PER_CONDITION = 10  # precondition 2, MIN across the two conditions (993 fix F7)
MIN_PROBE_COVERAGE = 5              # per-cell minimum n_near / n_safe to count (993 fix F5)

# SMOKE POSITIVE CONTROL, dry-run only. At --dry-run scale the readiness gate
# ALWAYS fires (2 episodes produce no harm events and a near-zero DV), so a
# plain dry-run exercises only the gate-blocked branch and leaves the entire
# MERGED arm -- new harm_logit signature, new train_step, the paired verdict
# logic -- unexecuted. That is exactly the short-circuit-hides-a-crash failure
# mode. `--smoke-force-gate` forces the gate green so the smoke covers the full
# path as a POSITIVE control. It is refused outside --dry-run, and the runner
# never passes it (the queue entry invokes the script with no such argument).
_SMOKE_FORCE_GATE = False


class SmallViewEnv(CausalGridWorld):
    """CausalGridWorld with a 3x3 local observation (radius 1) instead of
    the 5x5 default. Ported from V3-EXQ-010's `SmallViewEnv` (the one
    config with positive-direction SD-003 calibration_gap signal), with
    `num_hazards` taken from the constructor kwargs rather than a module
    constant, so it can serve both the DENSE and SPARSE conditions.
    Experiment-local -- does not modify the base class.
    """

    def __init__(self, **kwargs):
        view = OBS_VIEW_SIZE
        self._body_obs_dim = 10  # same as base
        self._world_obs_dim = view * view * 7 + view * view  # placeholder pre-super()
        super().__init__(**kwargs)
        self._world_obs_dim = view * view * self.NUM_ENTITY_TYPES + view * view

    @property
    def world_obs_dim(self) -> int:
        return self._world_obs_dim

    @property
    def body_obs_dim(self) -> int:
        return self._body_obs_dim

    def _get_observation_dict(self) -> Dict[str, torch.Tensor]:
        ax, ay = self.agent_x, self.agent_y
        r = OBS_RADIUS
        view = OBS_VIEW_SIZE

        body = torch.zeros(self._body_obs_dim)
        body[0] = ax / self.size
        body[1] = ay / self.size
        body[2] = self.agent_health
        body[3] = self.agent_energy
        max_vis = max(1, self.footprint_grid.max())
        body[4] = float(self.footprint_grid[ax, ay]) / max_vis
        action_enc = self._last_action if self._last_action < 4 else 0
        body[5 + action_enc] = 1.0
        body[9] = min(1.0, self.steps / 500.0)

        local_view = torch.zeros(view, view, self.NUM_ENTITY_TYPES)
        for di in range(-r, r + 1):
            for dj in range(-r, r + 1):
                ni, nj = ax + di, ay + dj
                if 0 <= ni < self.size and 0 <= nj < self.size:
                    etype = self.grid[ni, nj]
                else:
                    etype = self.ENTITY_TYPES["wall"]
                local_view[di + r, dj + r, etype] = 1.0
        local_view_flat = local_view.reshape(-1)

        cont_view = torch.zeros(view, view)
        for di in range(-r, r + 1):
            for dj in range(-r, r + 1):
                ni, nj = ax + di, ay + dj
                if 0 <= ni < self.size and 0 <= nj < self.size:
                    cont_view[di + r, dj + r] = float(self.contamination_grid[ni, nj])
        cont_view_flat = (cont_view / (self.contamination_threshold + 1e-6)).reshape(-1)

        world_state = torch.cat([local_view_flat, cont_view_flat])

        return {
            "body_state": body.float(),
            "world_state": world_state.float(),
            "contamination_view": cont_view_flat.float(),
        }


def _action_onehot(action_idx: int, num_actions: int) -> torch.Tensor:
    v = torch.zeros(1, num_actions)
    v[0, action_idx] = 1.0
    return v


def _random_cf_action(actual_idx: int, num_actions: int) -> int:
    """A counterfactual MOVEMENT action, never STAY (red-team F8).

    993 drew the counterfactual from all actions including STAY (index 4) while
    the ACTUAL action was always a movement -- near-hazard probes step into the
    hazard, safe probes draw randint(0, action_dim-2). With p=1/4 the contrast
    was therefore P(harm | move) - P(harm | STAY) rather than a
    movement-versus-movement counterfactual, and STAY on a cell whose
    contamination has crossed threshold is itself a harm transition
    (causal_grid_world.py:2421 retypes the vacated cell, :2497 then fires).
    Measured consequence in 993a's own design probe: in SPARSE the near-hazard
    mean causal_sig was itself <= 0 on 3 of 4 seeds, so the positive gap was
    being carried by the safe population's STAY counterfactual rather than by
    hazard-approach discrimination. Restricting the counterfactual to movement
    matches the actual-action distribution and makes the contrast the one the
    claim is about."""
    choices = [a for a in range(num_actions) if a != actual_idx and a != STAY_ACTION_IDX]
    return random.choice(choices) if choices else 0


def _make_world_codec(world_obs_dim: int) -> Tuple[nn.Module, nn.Module]:
    encoder = nn.Sequential(
        nn.Linear(world_obs_dim, HIDDEN_DIM), nn.ReLU(),
        nn.Linear(HIDDEN_DIM, WORLD_DIM),
    )
    decoder = nn.Sequential(
        nn.Linear(WORLD_DIM, HIDDEN_DIM), nn.ReLU(),
        nn.Linear(HIDDEN_DIM, world_obs_dim),
    )
    return encoder, decoder


class SeparatedChannels:
    """ARM_SEPARATED: three independent heads, three independent
    optimizers, three independent losses. No shared parameters."""

    def __init__(self, action_dim: int):
        self.action_dim = action_dim
        # DEPTH-MATCHED to MERGED (red-team F5). MERGED's every head reads a
        # 2-hidden-layer trunk plus a linear out = 3 weight layers; 993 gave
        # SEPARATED 2-layer heads, so the arms differed in DEPTH and CAPACITY as
        # well as in sharing. A "weakens" could then have been extra MERGED
        # capacity compensating for contamination, and a "supports" could have
        # been SEPARATED's shallower heads underfitting -- neither is ARC-021's
        # content. Matching depth and width leaves parameter SHARING and loss
        # SUMMATION as the only differences, which is exactly the manipulation.
        # Input dimensions still differ where the roles differ (SEPARATED's
        # sensory head does not see the action; MERGED's shared trunk must) --
        # that asymmetry IS the separate-channels premise and is not a confound.
        self.sensory_head = nn.Sequential(
            nn.Linear(WORLD_DIM, HIDDEN_DIM), nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.ReLU(),
            nn.Linear(HIDDEN_DIM, WORLD_DIM)
        )
        self.forward_head = nn.Sequential(
            nn.Linear(WORLD_DIM + action_dim, HIDDEN_DIM), nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.ReLU(),
            nn.Linear(HIDDEN_DIM, WORLD_DIM)
        )
        # ACTION-CONDITIONED (the 993a repair): h(z_t, a_t) -> logit P(harm on
        # this transition). 993 trained h(z_t1) -> harm_label, but
        # causal_grid_world.py:2745 overwrites the hazard cell with the agent
        # entity on entry, so z_t1 for a harm transition has the harm cue ERASED
        # and the head learns that post-harm states look safe. Conditioning on
        # (z_t, a_t) puts the label on the PRE-transition state, where the cue is
        # still visible, and matches the distribution the probes read.
        self.harm_head = nn.Sequential(
            nn.Linear(WORLD_DIM + action_dim, HIDDEN_DIM), nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.ReLU(),
            nn.Linear(HIDDEN_DIM, 1)
        )
        self.opt_sensory = optim.Adam(self.sensory_head.parameters(), lr=LR_HEAD)
        self.opt_forward = optim.Adam(self.forward_head.parameters(), lr=LR_HEAD)
        self.opt_harm = optim.Adam(self.harm_head.parameters(), lr=LR_HEAD)
        self._all_params = (
            list(self.sensory_head.parameters())
            + list(self.forward_head.parameters())
            + list(self.harm_head.parameters())
        )

    def predict_forward(self, z: torch.Tensor, action_onehot: torch.Tensor) -> torch.Tensor:
        return self.forward_head(torch.cat([z, action_onehot], dim=-1))

    def predict_sensory(self, z: torch.Tensor) -> torch.Tensor:
        return self.sensory_head(z)

    def harm_logit(self, z: torch.Tensor, action_onehot: torch.Tensor) -> torch.Tensor:
        return self.harm_head(torch.cat([z, action_onehot], dim=-1))

    def train_step(
        self, z_t: torch.Tensor, action_onehot: torch.Tensor, z_t1: torch.Tensor,
        harm_label: torch.Tensor,
    ) -> Dict[str, float]:
        # Sensory target is self-reconstruction (z_t -> z_t), NOT z_t1: z_t1 was
        # reached via a random ACTUAL action, so training an action-free head
        # against it would silently make "sensory prediction" a degenerate
        # proxy for the (already-separate) forward-prediction task. A stable
        # persistence/consistency objective is the coherent action-free
        # analogue (red-team fix F1; see module docstring DESIGN section).
        # CLIP-MATCHED to MERGED (red-team F5). MERGED applies ONE
        # clip_grad_norm_ at 1.0 over the union trunk+heads; 993 clipped each
        # SEPARATED head at 1.0 individually, which is a strictly weaker
        # constraint on the union. A "supports" could then have been joint-clip
        # starvation of MERGED's harm gradient rather than representational
        # contamination. All three gradients are computed first, then clipped
        # as one union, then the three (still independent) optimizers step --
        # so the channels remain genuinely separate in parameters and losses
        # while the clipping constraint is identical across arms.
        sensory_loss = F.mse_loss(self.sensory_head(z_t), z_t)
        forward_loss = F.mse_loss(self.predict_forward(z_t, action_onehot), z_t1)

        # harm_label is ALWAYS 0.0 or 1.0 (never skipped) -- red-team fix F2:
        # training on positive-only labels leaves harm_head undiscriminating
        # in any cell with sparse harm events (worse, at pure random init).
        # Trained on (z_t, a_t) -- the PRE-transition state and the action taken
        # from it -- so the head is read at exactly the point it is trained.
        harm_loss = F.binary_cross_entropy_with_logits(
            self.harm_logit(z_t, action_onehot), harm_label)

        self.opt_sensory.zero_grad()
        self.opt_forward.zero_grad()
        self.opt_harm.zero_grad()
        # THREE separate backward passes over THREE disjoint parameter sets --
        # no gradient from any loss reaches another channel's parameters. This
        # is the separate-channels premise; only the clip below is unioned.
        sensory_loss.backward()
        forward_loss.backward()
        harm_loss.backward()
        torch.nn.utils.clip_grad_norm_(self._all_params, 1.0)
        self.opt_sensory.step()
        self.opt_forward.step()
        self.opt_harm.step()

        return {
            "sensory_loss": float(sensory_loss.item()),
            "forward_loss": float(forward_loss.item()),
            "harm_loss": float(harm_loss.item()),
        }


class MergedChannels:
    """ARM_MERGED: one shared trunk, one optimizer over trunk+heads, one
    combined loss backpropagated jointly. The trunk consumes
    (z_world, action_onehot); STAY (index 4) is used as the "evaluate this
    state" action for the sensory and harm heads (state-only passes),
    while the actually-taken/counterfactual action drives the forward
    head's action-conditioned pass."""

    def __init__(self, action_dim: int):
        self.action_dim = action_dim
        self.trunk = nn.Sequential(
            nn.Linear(WORLD_DIM + action_dim, HIDDEN_DIM), nn.ReLU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.ReLU(),
        )
        self.sensory_out = nn.Linear(HIDDEN_DIM, WORLD_DIM)
        self.forward_out = nn.Linear(HIDDEN_DIM, WORLD_DIM)
        self.harm_out = nn.Linear(HIDDEN_DIM, 1)
        params = (
            list(self.trunk.parameters())
            + list(self.sensory_out.parameters())
            + list(self.forward_out.parameters())
            + list(self.harm_out.parameters())
        )
        self.opt = optim.Adam(params, lr=LR_HEAD)
        self._stay = _action_onehot(STAY_ACTION_IDX, action_dim)

    def _stay_onehot(self, batch_size: int) -> torch.Tensor:
        return self._stay.expand(batch_size, -1)

    def predict_forward(self, z: torch.Tensor, action_onehot: torch.Tensor) -> torch.Tensor:
        repr_ = self.trunk(torch.cat([z, action_onehot], dim=-1))
        return self.forward_out(repr_)

    def predict_sensory(self, z: torch.Tensor) -> torch.Tensor:
        repr_ = self.trunk(torch.cat([z, self._stay_onehot(z.shape[0])], dim=-1))
        return self.sensory_out(repr_)

    def harm_logit(self, z: torch.Tensor, action_onehot: torch.Tensor) -> torch.Tensor:
        # Action-conditioned, exactly as SEPARATED -- and deliberately through the
        # SAME shared trunk that carries the sensory and forward gradients. This is
        # where the ARC-021 manipulation bites: the harm readout's representation
        # is contaminated by the other two losses.
        repr_ = self.trunk(torch.cat([z, action_onehot], dim=-1))
        return self.harm_out(repr_)

    def train_step(
        self, z_t: torch.Tensor, action_onehot: torch.Tensor, z_t1: torch.Tensor,
        harm_label: torch.Tensor,
    ) -> Dict[str, float]:
        # Sensory target is self-reconstruction (z_t -> z_t) through the SAME
        # trunk(z, STAY) pass the harm head reads -- both are now genuinely
        # "evaluate/represent this state" objectives, coherent with each
        # other and with STAY as the "no action" convention (red-team fix
        # F1: the prior z_t1 target was reached via a random ACTUAL action,
        # contradicting the STAY-conditioned trunk pass on every step).
        sensory_loss = F.mse_loss(self.predict_sensory(z_t), z_t)
        forward_loss = F.mse_loss(self.predict_forward(z_t, action_onehot), z_t1)
        # harm_label is ALWAYS 0.0 or 1.0 (never skipped) -- red-team fix F2.
        harm_loss = F.binary_cross_entropy_with_logits(
            self.harm_logit(z_t, action_onehot), harm_label)
        total = sensory_loss + forward_loss + harm_loss
        self.opt.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.trunk.parameters()) + list(self.sensory_out.parameters())
            + list(self.forward_out.parameters()) + list(self.harm_out.parameters()),
            1.0,
        )
        self.opt.step()
        return {
            "sensory_loss": float(sensory_loss.item()),
            "forward_loss": float(forward_loss.item()),
            "harm_loss": float(harm_loss.item()),
        }


def _make_channels(arm: str, action_dim: int):
    return SeparatedChannels(action_dim) if arm == "SEPARATED" else MergedChannels(action_dim)


def _collect_random_episode(env: SmallViewEnv) -> List[Tuple[torch.Tensor, int, torch.Tensor, float]]:
    """One episode of uniform-random-policy rollout. Returns a list of
    (world_obs_t, action_idx_t, world_obs_t1, harm_signal_t) tuples."""
    _, obs_dict = env.reset()
    transitions = []
    for _step in range(STEPS_PER_EPISODE):
        world_obs_t = obs_dict["world_state"]
        action_idx = random.randint(0, env.action_dim - 1)
        _, harm_signal, done, _info, obs_dict = env.step(action_idx)
        world_obs_t1 = obs_dict["world_state"]
        transitions.append((world_obs_t, action_idx, world_obs_t1, float(harm_signal)))
        if done:
            break
    return transitions


def _run_p0(env: SmallViewEnv, encoder: nn.Module, decoder: nn.Module, opt: optim.Optimizer) -> Dict[str, float]:
    encoder.train()
    decoder.train()
    total_recon = 0.0
    n = 0
    for ep in range(P0_EPISODES):
        transitions = _collect_random_episode(env)
        for world_obs_t, _a, _world_obs_t1, _h in transitions:
            w = world_obs_t.unsqueeze(0)
            z = encoder(w)
            recon = decoder(z)
            loss = F.mse_loss(recon, w)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(decoder.parameters()), 1.0)
            opt.step()
            total_recon += float(loss.item())
            n += 1
        if (ep + 1) % 20 == 0 or ep == P0_EPISODES - 1:
            print(
                f"  [train] p0 ep {ep + 1}/{TOTAL_EPISODES_PER_CELL} "
                f"mean_recon={total_recon / max(1, n):.5f}",
                flush=True,
            )
    return {"mean_recon_loss": total_recon / max(1, n)}


def _run_p1(
    env: SmallViewEnv, encoder: nn.Module, channels, action_dim: int,
) -> Dict[str, Any]:
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    total_harm_events = 0
    for ep in range(P1_EPISODES):
        transitions = _collect_random_episode(env)
        for world_obs_t, action_idx, world_obs_t1, harm_signal in transitions:
            with torch.no_grad():
                z_t = encoder(world_obs_t.unsqueeze(0))
                z_t1 = encoder(world_obs_t1.unsqueeze(0))
            action_onehot = _action_onehot(action_idx, action_dim)
            # harm_label is ALWAYS 0.0 or 1.0 -- red-team fix F2 (positive-only
            # labels leave harm_head undiscriminating, worse in sparse-harm cells).
            is_harm = harm_signal < 0
            if is_harm:
                total_harm_events += 1
            harm_label = torch.ones(1, 1) if is_harm else torch.zeros(1, 1)
            channels.train_step(z_t, action_onehot, z_t1, harm_label)
        if (ep + 1) % 20 == 0 or ep == P1_EPISODES - 1:
            print(
                f"  [train] p1 ep {P0_EPISODES + ep + 1}/{TOTAL_EPISODES_PER_CELL} "
                f"harm_events={total_harm_events}",
                flush=True,
            )
    return {"p1_harm_events": total_harm_events}


def _harm_action_sensitivity(encoder: nn.Module, channels, env: SmallViewEnv, action_dim: int) -> float:
    """Positive control on the READOUT ITSELF: mean
    |harm_logit(z,a1) - harm_logit(z,a2)| over a fixed probe batch and two
    distinct movement actions, measured immediately after P1.

    This asserts the SAME statistic the load-bearing criteria consume (a
    difference of two harm logits at one state under two actions), not a
    magnitude proxy on a different component. 993 asserted the FORWARD head's
    action sensitivity, which was 17.9x over its floor while the readout the
    criteria actually read was flat -- the precondition certified a component
    the verdict never touched. An action-blind harm head makes causal_sig
    structurally zero in BOTH arms, so a below-floor reading means the
    substrate is not ready, never that ARC-021 was falsified."""
    with torch.no_grad():
        probe_worlds = []
        for _ in range(16):
            _, obs_dict = env.reset()
            probe_worlds.append(obs_dict["world_state"].unsqueeze(0))
        z = encoder(torch.cat(probe_worlds, dim=0))
        a1 = _action_onehot(0, action_dim).expand(z.shape[0], -1)
        a2 = _action_onehot(1, action_dim).expand(z.shape[0], -1)
        diff = channels.harm_logit(z, a1) - channels.harm_logit(z, a2)
        return float(diff.abs().mean().item())


def _eval_probes(env: SmallViewEnv, encoder: nn.Module, channels, action_dim: int) -> Dict[str, Any]:
    encoder.eval()
    near_sigs: List[float] = []
    safe_sigs: List[float] = []
    fatal_errors = 0

    wall_type = env.ENTITY_TYPES["wall"]
    hazard_type = env.ENTITY_TYPES["hazard"]

    def _run_probe(ax: int, ay: int, actual_idx: int) -> float:
        """causal_sig = harm_logit(z_t, a_actual) - harm_logit(z_t, a_cf).

        Two deliberate departures from 993, both measured (module docstring):
        (1) PRE-SIGMOID. The sigmoid compressed the DV by 78x; at a base rate of
            ~2-4% its derivative p(1-p) is ~0.02-0.04, so the whole contrast
            collapsed into ~0.2% of [-1,+1].
        (2) The forward head is NOT on this path. 993 evaluated the harm head on
            predict_forward(z, a), whose MSE-regressed output lands about as far
            from the true next latent as two random latents are apart
            (reach_ratio 1.15-1.49) -- so the counterfactual contrast was
            annihilated before the harm head ever saw it.
        The counterfactual structure of SD-003 / V3-EXQ-007-010 is preserved
        exactly: same state, actual action versus a distinct counterfactual one.
        """
        # RED-TEAM F7: reproduce the grid state step() would have produced, not
        # just the agent POINTER. causal_grid_world.step() vacates the old cell
        # (grid[old] = empty, or contaminated above threshold -- line 2453/2421)
        # and writes grid[new] = agent (line 2745), so EVERY training
        # observation has the agent token at the 3x3 centre. 993 moved only
        # agent_x/agent_y, leaving the centre reading empty/resource and the
        # stale agent token sitting elsewhere inside the 3x3 view -- an
        # out-of-distribution input at exactly the position the encoder is most
        # sensitive to. That is the same defect this driver cites to reject the
        # env-lookahead readout, so it cannot be tolerated here either. The
        # mutation is fully restored in the finally block, so probe order stays
        # irrelevant and the env is unchanged on exit.
        prev_x, prev_y = env.agent_x, env.agent_y
        saved_center = int(env.grid[ax, ay])
        saved_prev = int(env.grid[prev_x, prev_y])
        env.grid[prev_x, prev_y] = (
            env.ENTITY_TYPES["contaminated"]
            if float(env.contamination_grid[prev_x, prev_y]) >= env.contamination_threshold
            else env.ENTITY_TYPES["empty"]
        )
        env.grid[ax, ay] = env.ENTITY_TYPES["agent"]
        env.agent_x = ax
        env.agent_y = ay
        try:
            obs_dict = env._get_observation_dict()
            with torch.no_grad():
                z = encoder(obs_dict["world_state"].unsqueeze(0))
                cf_idx = _random_cf_action(actual_idx, action_dim)
                a_act = _action_onehot(actual_idx, action_dim)
                a_cf = _action_onehot(cf_idx, action_dim)
                h_act = channels.harm_logit(z, a_act)
                h_cf = channels.harm_logit(z, a_cf)
                return float((h_act - h_cf).item())
        finally:
            env.grid[ax, ay] = saved_center
            env.grid[prev_x, prev_y] = saved_prev
            env.agent_x, env.agent_y = prev_x, prev_y

    try:
        for _ in range(PROBE_RESETS):
            env.reset()
            for hx, hy in env.hazards:
                for action_idx, (dx, dy) in env.ACTIONS.items():
                    if action_idx == STAY_ACTION_IDX:
                        continue
                    ax, ay = hx - dx, hy - dy
                    if 0 <= ax < env.size and 0 <= ay < env.size:
                        cell = int(env.grid[ax, ay])
                        if cell not in (wall_type, hazard_type):
                            near_sigs.append(_run_probe(ax, ay, action_idx))

            for px in range(env.size):
                for py in range(env.size):
                    if int(env.grid[px, py]) in (wall_type, hazard_type):
                        continue
                    min_dist = min(abs(px - hx) + abs(py - hy) for hx, hy in env.hazards)
                    if min_dist > 3:
                        safe_sigs.append(_run_probe(px, py, random.randint(0, action_dim - 2)))
    except Exception:
        import traceback
        fatal_errors += 1
        print(f"  FATAL: {traceback.format_exc()}", flush=True)

    # Red-team fix F5: an empty probe list must NOT silently substitute 0.0
    # (a different statistic, "no safe probes" != "zero causal signature").
    # Coverage insufficiency propagates as None and excludes the cell from
    # aggregation, rather than corrupting the mean with a fabricated value.
    coverage_ok = (
        fatal_errors == 0
        and len(near_sigs) >= MIN_PROBE_COVERAGE
        and len(safe_sigs) >= MIN_PROBE_COVERAGE
    )
    mean_near = float(sum(near_sigs) / len(near_sigs)) if near_sigs else None
    mean_safe = float(sum(safe_sigs) / len(safe_sigs)) if safe_sigs else None
    calibration_gap = (
        (mean_near - mean_safe) if (coverage_ok and mean_near is not None and mean_safe is not None) else None
    )

    return {
        "calibration_gap": calibration_gap,
        "mean_causal_sig_near_hazard": mean_near,
        "mean_causal_sig_safe": mean_safe,
        "n_near_hazard_probes": len(near_sigs),
        "n_safe_probes": len(safe_sigs),
        "fatal_errors": fatal_errors,
        "coverage_ok": coverage_ok,
    }


def _run_cell(condition: str, arm: str, seed: int) -> Dict[str, Any]:
    print(f"Seed {seed} Condition {condition}_{arm}", flush=True)
    config_slice = {
        "condition": condition,
        "arm": arm,
        "num_hazards": NUM_HAZARDS[condition],
        "grid_size": GRID_SIZE,
        "obs_view_size": OBS_VIEW_SIZE,
        "world_dim": WORLD_DIM,
        "hidden_dim": HIDDEN_DIM,
        "p0_episodes": P0_EPISODES,
        "p1_episodes": P1_EPISODES,
        "steps_per_episode": STEPS_PER_EPISODE,
    }
    with arm_cell(seed, config_slice=config_slice, script_path=Path(__file__)) as cell:
        env = SmallViewEnv(size=GRID_SIZE, num_hazards=NUM_HAZARDS[condition], seed=seed)
        action_dim = env.action_dim
        encoder, decoder = _make_world_codec(env.world_obs_dim)
        codec_opt = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=LR_ENCODER)

        p0_stats = _run_p0(env, encoder, decoder, codec_opt)
        channels = _make_channels(arm, action_dim)
        p1_stats = _run_p1(env, encoder, channels, action_dim)

        harm_sensitivity = _harm_action_sensitivity(encoder, channels, env, action_dim)
        probe_stats = _eval_probes(env, encoder, channels, action_dim)

        # Red-team fix N1 (2nd pass): calibration_gap is None when coverage_ok
        # is False -- f"{None:.4f}" raises TypeError, which would crash the
        # whole run and make the F5 exclusion path unreachable dead code.
        _gap = probe_stats["calibration_gap"]
        _gap_str = f"{_gap:.4f}" if _gap is not None else "None(coverage_insufficient)"
        print(
            f"  [{condition}_{arm} seed={seed}] gap={_gap_str} "
            f"n_near={probe_stats['n_near_hazard_probes']} n_safe={probe_stats['n_safe_probes']} "
            f"harm_action_sensitivity={harm_sensitivity:.5f} harm_events={p1_stats['p1_harm_events']}",
            flush=True,
        )

        row = {
            "condition": condition,
            "arm": arm,
            "seed": seed,
            "mean_recon_loss": p0_stats["mean_recon_loss"],
            "p1_harm_events": p1_stats["p1_harm_events"],
            "harm_action_sensitivity": harm_sensitivity,
            **probe_stats,
        }
        cell.stamp(row)

    passed = bool(row["coverage_ok"])
    print(f"verdict: {'PASS' if passed else 'FAIL'}", flush=True)
    return row


def _build_preconditions(rows: List[Dict[str, Any]], stage: str) -> List[Dict[str, Any]]:
    """Build all four readiness checks over `rows`.

    Called TWICE: once over the SEPARATED (control) cells only, before any
    MERGED cell is trained, and once over the full grid for the manifest. The
    early call is the whole point of the dv_headroom class -- V3-EXQ-993 burned
    its entire 12-cell grid before discovering its control arm produced no
    signal, and a headroom check can only be computed once control values
    exist, so control-first ordering is what makes the gate cheap."""
    # RED-TEAM F2: the gate ranges over the CONTROL arm ONLY. A MERGED harm head
    # gone action-blind is the EXTREME FORM of the degradation ARC-021 predicts;
    # gating on it would convert the strongest possible positive result into
    # "substrate not ready". The docstring's justification for this precondition
    # ("structurally zero in BOTH arms") is true of the control arm, which is
    # what certifies that the instrument can read anything at all. MERGED's own
    # sensitivity is still RECORDED, as a diagnostic, never as a gate.
    control_rows = [r for r in rows if r["arm"] == "SEPARATED"]
    worst_harm_sensitivity = min(r["harm_action_sensitivity"] for r in control_rows)
    worst_cell = min(control_rows, key=lambda r: r["harm_action_sensitivity"])
    merged_rows = [r for r in rows if r["arm"] == "MERGED"]
    worst_merged_sensitivity = (
        min(r["harm_action_sensitivity"] for r in merged_rows) if merged_rows else None
    )
    dense_harm_events = sum(r["p1_harm_events"] for r in rows if r["condition"] == "DENSE")
    sparse_harm_events = sum(r["p1_harm_events"] for r in rows if r["condition"] == "SPARSE")
    min_condition_harm_events = min(dense_harm_events, sparse_harm_events)

    # Control-arm realised DV values -- the input to BOTH dv_headroom checks.
    # Only SEPARATED cells with usable probe coverage contribute; a None gap is
    # a coverage exclusion (993 fix F5), not a zero.
    control_gaps = [
        r["calibration_gap"] for r in rows
        if r["arm"] == "SEPARATED" and r["calibration_gap"] is not None
    ]
    # RED-TEAM F6: the headroom checks would otherwise certify over the REALIZED
    # control cells rather than the INTENDED ones -- a 7-of-8 control arm could
    # pass H1, train the whole MERGED arm, and only then fail non-degeneracy on
    # the short condition (which requires all seeds paired). Denominating the
    # gate on the intended n is the difference between refusing early and
    # discovering the shortfall after the compute is spent.
    n_intended_control = len(SEEDS) * len(CONDITIONS)
    precondition_0 = {
        "name": "control_arm_coverage_complete",
        "description": (
            "Every intended SEPARATED (control) cell produced a usable "
            "calibration_gap. The dv_headroom checks below are denominated on "
            "this same set, so a short control arm must refuse here rather than "
            "silently certify headroom over fewer cells than the criteria need."
        ),
        "kind": "readiness",
        "measured": float(len(control_gaps)),
        "threshold": float(n_intended_control),
        "direction": "lower",
        "control": f"intended = len(SEEDS) x len(CONDITIONS) = {n_intended_control}",
        "met": bool(len(control_gaps) >= n_intended_control),
    }

    precondition_1 = {
        "name": "harm_head_action_sensitivity_present",
        "description": (
            "Worst-cell mean |harm_logit(z,a1)-harm_logit(z,a2)| over a fixed probe "
            "batch and two distinct movement actions, measured after P1 training. "
            "This is the SAME statistic the load-bearing criteria route on (a "
            "difference of two harm logits at one state under two actions), not a "
            "magnitude proxy on a different component -- 993 asserted the FORWARD "
            "head's sensitivity at 17.9x over floor while the readout the criteria "
            "actually read was flat. An action-blind harm head makes causal_sig "
            "structurally zero in BOTH arms, so below-floor means substrate-not-ready."
        ),
        "kind": "readiness",
        "measured": worst_harm_sensitivity,
        "threshold": HARM_ACTION_SENSITIVITY_FLOOR,
        "direction": "lower",
        "control": (
            f"16-sample probe batch, actions 0 vs 1, worst cell across the "
            f"{len(control_rows)} SEPARATED (control) cells of the {stage} set"
        ),
        "worst_merged_harm_action_sensitivity_diagnostic": worst_merged_sensitivity,
        "offending_cell": f"{worst_cell['condition']}_{worst_cell['arm']}_seed{worst_cell['seed']}",
        "met": bool(worst_harm_sensitivity >= HARM_ACTION_SENSITIVITY_FLOOR),
    }
    precondition_2 = {
        "name": "p1_harm_events_observed_per_condition",
        "description": (
            "MIN(total P1 harm events summed over arms+seeds) across the two hazard-"
            "density conditions -- 993 red-team fix F7: a pooled floor let SPARSE's "
            "harm head train on near-zero labeled data while DENSE's volume masked "
            "it. Taking the min across conditions requires BOTH to have real coverage."
        ),
        "kind": "readiness",
        "measured": float(min_condition_harm_events),
        "threshold": float(HARM_EVENTS_FLOOR_PER_CONDITION),
        "direction": "lower",
        "control": f"dense={dense_harm_events}, sparse={sparse_harm_events} ({stage} cells)",
        "met": bool(min_condition_harm_events >= HARM_EVENTS_FLOOR_PER_CONDITION),
    }

    # H1/H2 -- the dv_headroom class minted by the 2026-09-03 cluster autopsy
    # (ree-v3 8e133d26ed). criterion_threshold is passed as the module constant
    # the criterion itself reads, never a re-typed literal, so the gate cannot
    # drift away from the science it guards.
    if control_gaps:
        h1 = dv_headroom_check(
            "dv_headroom_margin_room_below_control",
            dv_name="calibration_gap",
            criterion_threshold=MARGIN,
            control_values=control_gaps,
            statistic="floor_headroom",
            dv_bounds=(0.0, 1.0),
            margin=1.0,
            kind_note=(
                "floor_headroom = min(control calibration_gap) - 0.0 is exactly the "
                "room the MERGED arm has to FALL, which is what this suppression-"
                "direction criterion consumes. 0.0 is used as the floor because 'no "
                "hazard-conditional harm calibration at all' is the meaningful bottom; "
                "the logit difference is formally unbounded below, so this UNDERSTATES "
                "the available room and is therefore conservative. Design-time probe "
                "expectation: ~0.082 measured vs 0.06 required (~1.36x)."
            ),
        )
        h2 = dv_headroom_check(
            "dv_headroom_control_signal_floor_reachable",
            dv_name="calibration_gap",
            criterion_threshold=SEPARATED_SIGNAL_FLOOR,
            control_values=control_gaps,
            statistic="max_abs",
            margin=2.0,
            kind_note=(
                "The 2026-09-03 cluster autopsy's own table pairs 993's max "
                "|calibration_gap| (0.00152) against SEPARATED_SIGNAL_FLOOR (0.02) for "
                "a 13.1x shortfall, so max_abs is the sanctioned statistic for this "
                "non-degeneracy floor. Design-time probe expectation: ~0.215 measured "
                "vs 0.08 required (~2.68x)."
            ),
        )
        headroom = [h1, h2]
    else:
        # No usable control value at all -- report as UNMET rather than skipping,
        # so an empty control arm can never certify itself by absence.
        headroom = [{
            "name": "dv_headroom_margin_room_below_control",
            "kind": "dv_headroom",
            "dv_name": "calibration_gap",
            "achievable_statistic": "floor_headroom",
            "measured": float("nan"),
            "threshold": MARGIN,
            "direction": "lower",
            "criterion_threshold": MARGIN,
            "headroom_margin": 1.0,
            "n_control_values": 0,
            "met": False,
            "control": "no SEPARATED cell produced a usable calibration_gap",
        }]
    return [precondition_0, precondition_1, precondition_2] + headroom


def _mean_or_none(rows: List[Dict[str, Any]], condition: str, arm: str) -> Optional[float]:
    vals = [
        r["calibration_gap"] for r in rows
        if r["condition"] == condition and r["arm"] == arm and r["calibration_gap"] is not None
    ]
    return statistics.fmean(vals) if vals else None


def run_experiment() -> Dict[str, Any]:
    # STAGE A -- every SEPARATED (control) cell, both conditions, first.
    rows: List[Dict[str, Any]] = []
    for condition in CONDITIONS:
        for seed in SEEDS:
            rows.append(_run_cell(condition, "SEPARATED", seed))

    control_preconditions = _build_preconditions(rows, "control-arm")
    gate_blocked = False
    try:
        p0_readiness_gate(control_preconditions)
    except P0NotReady:
        gate_blocked = True

    if gate_blocked and _SMOKE_FORCE_GATE:
        print(
            "  [smoke] readiness gate UNMET but --smoke-force-gate is set; "
            "continuing to the MERGED arm as a positive control. This path is "
            "dry-run-only and can never be reached in a real run.",
            flush=True,
        )
        gate_blocked = False

    if gate_blocked:
        # Self-route to substrate_not_ready_requeue WITHOUT training a single
        # MERGED cell. This is never a verdict on ARC-021 or MECH-069.
        # NOTE: no extra `verdict:` prints here -- _run_cell already emitted one
        # per control cell, and the runner counts verdict lines to advance its
        # progress bar. The MERGED cells simply never run, so this run emits
        # len(SEEDS) x len(CONDITIONS) verdicts instead of twice that.
        return {
            "outcome": "FAIL",
            "evidence_direction": "non_contributory",
            "evidence_direction_per_claim": {c: "non_contributory" for c in CLAIM_IDS},
            "evidence_direction_note": (
                "NO COMPARISON WAS MADE. Set explicitly so build_experiment_indexes.py "
                "does NOT rewrite this run's direction: the indexer computes "
                "direction_explicitly_set = bool(manifest['evidence_direction_note']) "
                "(line 1871) and then rewrites any non-explicit 'unknown' on a FAIL "
                "to 'weakens' (lines 3519-3520). V3-EXQ-993 emitted a bare 'unknown' "
                "and its 2026-09-03 autopsy had to hand-correct the record to "
                "'non_contributory'. non_degenerate/scoring_excluded does NOT rescue "
                "this: the gap register's conflict_ratio counts entries regardless of "
                "scoring_excluded. So the direction is stated as non_contributory AND "
                "this note is present, which are the two independent things that keep "
                "an instrument-not-ready run from being logged as evidence against "
                "ARC-021 or MECH-069. ""Trigger: the control-arm readiness gate was unmet "
                "before any MERGED cell was trained."
            ),
            "non_degenerate": False,
            "degeneracy_reason": (
                "control-arm readiness gate unmet before the MERGED arm was trained "
                "(see interpretation.preconditions); no comparison was made"
            ),
            "metrics": {
                "mean_calibration_gap_dense_separated": _mean_or_none(rows, "DENSE", "SEPARATED"),
                "mean_calibration_gap_dense_merged": None,
                "mean_calibration_gap_sparse_separated": _mean_or_none(rows, "SPARSE", "SEPARATED"),
                "mean_calibration_gap_sparse_merged": None,
                "dense_n_seed_pairs": 0,
                "sparse_n_seed_pairs": 0,
                "worst_harm_action_sensitivity": control_preconditions[0]["measured"],
                "dense_harm_events": sum(r["p1_harm_events"] for r in rows if r["condition"] == "DENSE"),
                "sparse_harm_events": sum(r["p1_harm_events"] for r in rows if r["condition"] == "SPARSE"),
                "n_finite_violations_total": sum(r["fatal_errors"] for r in rows),
                "margin": MARGIN,
                "separated_signal_floor": SEPARATED_SIGNAL_FLOOR,
                "control_only_stage": True,
                "n_control_cells": len(rows),
            },
            "per_seed_rows": rows,
            "arm_results": rows,
            "interpretation": {
                "label": "substrate_not_ready_requeue",
                "preconditions": control_preconditions,
                "combination_rule": (
                    "Control-arm-first staging: the readiness gate (incl. both "
                    "dv_headroom checks) is evaluated over the SEPARATED cells before "
                    "any MERGED cell is trained. An unmet gate stops the run here."
                ),
                "criteria": [
                    {"name": "C1_dense_merged_degrades_vs_separated", "load_bearing": True, "passed": False},
                    {"name": "C2_sparse_merged_degrades_vs_separated", "load_bearing": True, "passed": False},
                ],
                "criteria_non_degenerate": {
                    "C1_dense_merged_degrades_vs_separated": False,
                    "C2_sparse_merged_degrades_vs_separated": False,
                },
            },
        }

    # STAGE B -- the MERGED arm, only once the control arm has demonstrated the
    # DV has room for the criterion to fire.
    for condition in CONDITIONS:
        for seed in SEEDS:
            rows.append(_run_cell(condition, "MERGED", seed))

    n_finite_violations_total = sum(r["fatal_errors"] for r in rows)
    preconditions = _build_preconditions(rows, "full-grid")
    # Evaluate the full-grid gate through the SAME tested helper stage A uses,
    # rather than re-implementing its met/direction/NaN semantics here. A
    # hand-rolled duplicate is exactly how a gate drifts away from the thing it
    # is supposed to enforce.
    try:
        p0_readiness_gate(preconditions)
        readiness_met = True
    except P0NotReady:
        readiness_met = False
    worst_harm_sensitivity = preconditions[0]["measured"]
    dense_harm_events = sum(r["p1_harm_events"] for r in rows if r["condition"] == "DENSE")
    sparse_harm_events = sum(r["p1_harm_events"] for r in rows if r["condition"] == "SPARSE")

    def _condition_verdict(condition: str) -> Dict[str, Any]:
        """993 red-team fixes F3+F4, carried forward: a PAIRED per-seed comparison (same seed drives
        bit-identical P0 in both arms, so pairing removes cross-seed P0
        variance) with a sign-consistency requirement (all seeds must agree
        in direction, not just the pooled mean), and per-CONDITION
        non-degeneracy gating (a degenerate SPARSE baseline can no longer
        silently count toward the verdict via C2)."""
        sep_by_seed = {
            r["seed"]: r["calibration_gap"] for r in rows
            if r["condition"] == condition and r["arm"] == "SEPARATED" and r["calibration_gap"] is not None
        }
        mer_by_seed = {
            r["seed"]: r["calibration_gap"] for r in rows
            if r["condition"] == condition and r["arm"] == "MERGED" and r["calibration_gap"] is not None
        }
        paired_seeds = sorted(set(sep_by_seed) & set(mer_by_seed))
        diffs = [mer_by_seed[s] - sep_by_seed[s] for s in paired_seeds]
        mean_sep = statistics.fmean(sep_by_seed.values()) if sep_by_seed else None
        mean_merged = statistics.fmean(mer_by_seed.values()) if mer_by_seed else None
        # Non-degenerate requires EVERY seed to have a valid paired cell (no
        # silent power loss from an excluded cell) AND the separated baseline
        # to clear the (decoupled) signal floor.
        non_degenerate = bool(
            len(paired_seeds) == len(SEEDS) and mean_sep is not None and mean_sep >= SEPARATED_SIGNAL_FLOOR
        )
        criterion_passed = bool(
            non_degenerate and diffs
            and statistics.fmean(diffs) <= -MARGIN
            and all(d <= 0 for d in diffs)
        )
        return {
            "condition": condition,
            "mean_gap_separated": mean_sep,
            "mean_gap_merged": mean_merged,
            "n_seed_pairs": len(paired_seeds),
            "per_seed_diffs": diffs,
            "non_degenerate": non_degenerate,
            "criterion_passed": criterion_passed,
        }

    dense_v = _condition_verdict("DENSE")
    sparse_v = _condition_verdict("SPARSE")

    both_nd = dense_v["non_degenerate"] and sparse_v["non_degenerate"]
    only_dense_nd = dense_v["non_degenerate"] and not sparse_v["non_degenerate"]
    only_sparse_nd = sparse_v["non_degenerate"] and not dense_v["non_degenerate"]
    neither_nd = not dense_v["non_degenerate"] and not sparse_v["non_degenerate"]

    non_degenerate: bool
    degeneracy_reason: Optional[str]
    evidence_direction_note: Optional[str] = None
    if not readiness_met:
        label = "channel_head_machinery_not_ready_substrate_not_ready_requeue"
        outcome = "FAIL"
        evidence_direction = "non_contributory"
        evidence_direction_note = (
            "NO COMPARISON WAS MADE. Set explicitly so build_experiment_indexes.py "
                "does NOT rewrite this run's direction: the indexer computes "
                "direction_explicitly_set = bool(manifest['evidence_direction_note']) "
                "(line 1871) and then rewrites any non-explicit 'unknown' on a FAIL "
                "to 'weakens' (lines 3519-3520). V3-EXQ-993 emitted a bare 'unknown' "
                "and its 2026-09-03 autopsy had to hand-correct the record to "
                "'non_contributory'. non_degenerate/scoring_excluded does NOT rescue "
                "this: the gap register's conflict_ratio counts entries regardless of "
                "scoring_excluded. So the direction is stated as non_contributory AND "
                "this note is present, which are the two independent things that keep "
                "an instrument-not-ready run from being logged as evidence against "
                "ARC-021 or MECH-069. ""Trigger: a readiness precondition was unmet over the "
            "full grid."
        )
        non_degenerate = False
        degeneracy_reason = "readiness preconditions not met (see interpretation.preconditions)"
    elif neither_nd:
        label = "separated_baseline_signal_absent_both_conditions"
        outcome = "FAIL"
        evidence_direction = "non_contributory"
        evidence_direction_note = (
            "NO COMPARISON WAS MADE. Set explicitly so build_experiment_indexes.py "
                "does NOT rewrite this run's direction: the indexer computes "
                "direction_explicitly_set = bool(manifest['evidence_direction_note']) "
                "(line 1871) and then rewrites any non-explicit 'unknown' on a FAIL "
                "to 'weakens' (lines 3519-3520). V3-EXQ-993 emitted a bare 'unknown' "
                "and its 2026-09-03 autopsy had to hand-correct the record to "
                "'non_contributory'. non_degenerate/scoring_excluded does NOT rescue "
                "this: the gap register's conflict_ratio counts entries regardless of "
                "scoring_excluded. So the direction is stated as non_contributory AND "
                "this note is present, which are the two independent things that keep "
                "an instrument-not-ready run from being logged as evidence against "
                "ARC-021 or MECH-069. ""Trigger: neither hazard-density condition's CONTROL arm "
            "cleared the non-degeneracy floor with full seed-pair coverage, so the "
            "ablation had nothing to remove -- exactly V3-EXQ-993's own outcome."
        )
        non_degenerate = False
        degeneracy_reason = (
            f"Neither DENSE (mean_sep={dense_v['mean_gap_separated']}, "
            f"n_pairs={dense_v['n_seed_pairs']}/{len(SEEDS)}) nor SPARSE "
            f"(mean_sep={sparse_v['mean_gap_separated']}, n_pairs={sparse_v['n_seed_pairs']}/{len(SEEDS)}) "
            f"cleared SEPARATED_SIGNAL_FLOOR={SEPARATED_SIGNAL_FLOOR} with full seed-pair coverage."
        )
    elif both_nd:
        overall = dense_v["criterion_passed"] and sparse_v["criterion_passed"]
        outcome = "PASS" if overall else "FAIL"
        non_degenerate = True
        degeneracy_reason = None
        if overall:
            label = "merged_channel_degrades_calibration_both_conditions"
            evidence_direction = "supports"
        elif dense_v["criterion_passed"] or sparse_v["criterion_passed"]:
            label = "merged_channel_degrades_calibration_one_condition_only"
            evidence_direction = "mixed"
        else:
            label = "merged_channel_hypothesis_not_supported"
            evidence_direction = "weakens"
    else:
        scored, scored_name = (dense_v, "DENSE") if only_dense_nd else (sparse_v, "SPARSE")
        unscored_name = "SPARSE" if only_dense_nd else "DENSE"
        outcome = "PASS" if scored["criterion_passed"] else "FAIL"
        # Red-team second-pass finding 3: a single hazard-density condition
        # was never meant to carry full evidential weight -- ARC-021's own
        # falsifying-signature text asks for BOTH dense- and sparse-hazard
        # conditions. A claim-scoring pipeline reading only
        # evidence_direction/evidence_direction_per_claim has no visibility
        # into interpretation.label, so cap this branch at "mixed" rather
        # than a full-strength "supports"/"weakens" regardless of which way
        # the single scored condition's criterion fell.
        evidence_direction = "mixed"
        label = f"single_condition_only_partial_evidence_{scored_name.lower()}"
        non_degenerate = True
        degeneracy_reason = (
            f"{unscored_name} condition did not clear SEPARATED_SIGNAL_FLOOR with full seed-pair "
            f"coverage -- only {scored_name} contributes (partial evidence, capped at evidence_direction=mixed)."
        )

    evidence_direction_per_claim = {c: evidence_direction for c in CLAIM_IDS}
    # A note is written on EVERY branch that reports a real comparison too, so a
    # later reader never has to infer whether the direction was deliberate.
    if evidence_direction_note is None:
        evidence_direction_note = (
            f"Direction set deliberately from the pre-registered combination rule "
            f"(label={label}); a comparison WAS made in at least one condition."
        )

    metrics = {
        "mean_calibration_gap_dense_separated": dense_v["mean_gap_separated"],
        "mean_calibration_gap_dense_merged": dense_v["mean_gap_merged"],
        "mean_calibration_gap_sparse_separated": sparse_v["mean_gap_separated"],
        "mean_calibration_gap_sparse_merged": sparse_v["mean_gap_merged"],
        "dense_n_seed_pairs": dense_v["n_seed_pairs"],
        "sparse_n_seed_pairs": sparse_v["n_seed_pairs"],
        "worst_harm_action_sensitivity": worst_harm_sensitivity,
        "dense_harm_events": dense_harm_events,
        "sparse_harm_events": sparse_harm_events,
        "n_finite_violations_total": n_finite_violations_total,
        "margin": MARGIN,
        "separated_signal_floor": SEPARATED_SIGNAL_FLOOR,
    }

    return {
        "outcome": outcome,
        "evidence_direction": evidence_direction,
        "evidence_direction_per_claim": evidence_direction_per_claim,
        "evidence_direction_note": evidence_direction_note,
        "non_degenerate": non_degenerate,
        "degeneracy_reason": degeneracy_reason,
        "metrics": metrics,
        "per_seed_rows": rows,
        "arm_results": rows,
        "interpretation": {
            "label": label,
            "preconditions": preconditions,
            "combination_rule": (
                "PASS iff C1_dense AND C2_sparse when BOTH conditions clear non-degeneracy; "
                "if only one condition clears it, outcome is that condition's criterion alone "
                "(label prefixed single_condition_only_partial_evidence); if neither clears it, "
                f"unknown/non_degenerate=false. Each criterion additionally requires ALL "
                f"{len(SEEDS)} seeds to show MERGED<=SEPARATED (sign-consistency), not just a "
                f"pooled mean margin. MARGIN={MARGIN} and SEPARATED_SIGNAL_FLOOR="
                f"{SEPARATED_SIGNAL_FLOOR} are derived from the control arm's measured range "
                f"(see the module docstring), and both are additionally gated at runtime by "
                f"the dv_headroom preconditions H1/H2, which are evaluated over the SEPARATED "
                f"cells BEFORE any MERGED cell is trained."
            ),
            "criteria": [
                {"name": "C1_dense_merged_degrades_vs_separated", "load_bearing": True, "passed": dense_v["criterion_passed"]},
                {"name": "C2_sparse_merged_degrades_vs_separated", "load_bearing": True, "passed": sparse_v["criterion_passed"]},
            ],
            "criteria_non_degenerate": {
                "C1_dense_merged_degrades_vs_separated": bool(readiness_met and dense_v["non_degenerate"]),
                "C2_sparse_merged_degrades_vs_separated": bool(readiness_met and sparse_v["non_degenerate"]),
            },
            "condition_detail": {"DENSE": dense_v, "SPARSE": sparse_v},
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--smoke-force-gate", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()

    t0 = time.perf_counter()
    global SEEDS, P0_EPISODES, P1_EPISODES, STEPS_PER_EPISODE, TOTAL_EPISODES_PER_CELL, PROBE_RESETS
    global _SMOKE_FORCE_GATE
    if args.smoke_force_gate:
        if not args.dry_run:
            raise SystemExit(
                "--smoke-force-gate is a dry-run-only smoke positive control and "
                "must never be used for a scored run."
            )
        _SMOKE_FORCE_GATE = True
    if args.dry_run:
        SEEDS = [101]
        P0_EPISODES = 2
        P1_EPISODES = 2
        STEPS_PER_EPISODE = 10
        TOTAL_EPISODES_PER_CELL = P0_EPISODES + P1_EPISODES
        PROBE_RESETS = 2

    result = run_experiment()
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    full_config = {
        "seeds": SEEDS,
        "conditions": CONDITIONS,
        "arms": ARMS,
        "num_hazards": NUM_HAZARDS,
        "grid_size": GRID_SIZE,
        "obs_view_size": OBS_VIEW_SIZE,
        "world_dim": WORLD_DIM,
        "hidden_dim": HIDDEN_DIM,
        "p0_episodes": P0_EPISODES,
        "p1_episodes": P1_EPISODES,
        "steps_per_episode": STEPS_PER_EPISODE,
        "probe_resets": PROBE_RESETS,
        "margin": MARGIN,
        "separated_signal_floor": SEPARATED_SIGNAL_FLOOR,
        "harm_action_sensitivity_floor": HARM_ACTION_SENSITIVITY_FLOOR,
        "harm_readout": "action_conditioned_pre_sigmoid_logit",
        "supersedes": SUPERSEDES,
    }
    manifest = {
        "run_id": f"{EXPERIMENT_TYPE}_{ts}_v3",
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": "V3-EXQ-993a",
        "supersedes": SUPERSEDES,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids": CLAIM_IDS,
        "claim_ids_tested": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "evidence_class": "experimental",
        "evidence_direction": result["evidence_direction"],
        "evidence_direction_per_claim": result["evidence_direction_per_claim"],
        "evidence_direction_note": result.get("evidence_direction_note"),
        "non_degenerate": result["non_degenerate"],
        "degeneracy_reason": result["degeneracy_reason"],
        "outcome": result["outcome"],
        "timestamp_utc": ts,
        "metrics": result["metrics"],
        "per_seed_rows": result["per_seed_rows"],
        "arm_results": result["arm_results"],
        "interpretation": result["interpretation"],
        "summary": (
            f"ARC-021/MECH-069 merged-channel ablation (action-conditioned pre-sigmoid "
            f"harm readout, supersedes V3-EXQ-993): {result['outcome']} "
            f"({result['interpretation']['label']}). "
            f"DENSE calibration_gap: separated={result['metrics']['mean_calibration_gap_dense_separated']} "
            f"vs merged={result['metrics']['mean_calibration_gap_dense_merged']}. "
            f"SPARSE calibration_gap: separated={result['metrics']['mean_calibration_gap_sparse_separated']} "
            f"vs merged={result['metrics']['mean_calibration_gap_sparse_merged']}. "
            f"non_degenerate={result['non_degenerate']}."
        ),
    }
    out_path = write_flat_manifest(
        manifest,
        dry_run=args.dry_run,
        config=full_config,
        seeds=SEEDS,
        script_path=Path(__file__),
        started_at=t0,
    )
    print(f"outcome: {result['outcome']}", flush=True)
    print(f"label: {result['interpretation']['label']}", flush=True)
    print(
        f"C1_dense={result['interpretation']['criteria'][0]['passed']} "
        f"C2_sparse={result['interpretation']['criteria'][1]['passed']} "
        f"non_degenerate={result['non_degenerate']}",
        flush=True,
    )
    print(f"wrote: {out_path}", flush=True)
    return result, out_path, args.dry_run


if __name__ == "__main__":
    _result, _out_path, _dry_run = main()
    _outcome_raw = str(_result["outcome"]).upper()
    emit_outcome(
        outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=str(_out_path),
        dry_run=_dry_run,
    )
