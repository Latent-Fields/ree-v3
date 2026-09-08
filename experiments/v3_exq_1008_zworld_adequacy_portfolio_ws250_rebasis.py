"""V3-EXQ-1008 -- z_world ACTOR-ADEQUACY PORTFOLIO (post-1002): the two parallel legs the
confirmed V3-EXQ-1002 autopsy specified, on V3-EXQ-1002's own dataset, adapter and standardiser.

  LEG 1 (instrumentation axis; adjudicates H-E-channel-input-capacity)
        A 32-dim TASK-AGNOSTIC LINEAR compression of the FULL 250-dim `world_state` -- the
        encoder's ACTUAL input -- read by the same capacity-matched adapter. Declared null: this
        250 -> 32 projection also fails the 0.80 bar.
  LEG 2 (representation axis; the OWED H-C corroborator)
        An INFORMATION-PRESERVING, DECISION-RELEVANT linear re-basis of the frozen 978-OFF
        z_world -- an invertible 32x32 map whose first 25 axes are the train-split LINEAR DECODE
        of the resource field (the "train-split-fitted supervised linear re-basis" the autopsy
        named), plus the orthogonal complement -- read by the same adapter, with the CLOSED-FORM
        "decode-then-oracle" agreement recorded alongside as the witness of how much of the
        mapping the linearly decodable content supports at all. Unsupervised full-covariance
        (ZCA) and within-class (Fisher) whitening are carried as secondaries. Declared null:
        agreement is flat under the decision-relevant re-basis. A flat result attributable to
        the linear CONTENT (the closed-form reader is flat too) WEAKENS the geometry reading of
        H-C; a lift corroborates it.

Question (hypothesis_space_registry.v1.json qid `zworld_actor_adequacy_locus`):
  H-B eliminated, H-C confirmed (caveated basis), H-D confirmed, H-E alive (V3-EXQ-1002,
  confirmed autopsy failure_autopsy_V3-EXQ-1002_2026-09-05 sections 6-9). This run adjudicates
  H-E and corroborates-or-weakens H-C. Both legs' declared nulls are informative.

Claims: NONE (claim_ids = []). Bears on INV-088 / MECH-457 but does not tag them -- both were
recorded PERIPHERAL co-tags by the 978 autopsy and MECH-457's re-derive brake stands at 13;
re-attaching them here would increment two brake counters on a run that exercises neither
claim's mechanism. What this run adjudicates is a hypothesis-space question, not a claim.

EXPERIMENT_PURPOSE = "diagnostic"

SLEEP DRIVER: not applicable -- no sleep flag is set (x734 all-ON stack at this rung enables no
sleep loop). Recorded as sleep_driver_pattern="none".

red-team (opus): CONTESTED, no BLOCKING -- see the REPAIR RECORD at the end of this docstring
and the queue entry note for V3-EXQ-1008.

=== WHY THIS RUN, AND WHY IT IS NOT A 1002b ===

V3-EXQ-1002 established, with credit assignment removed and the reader at the consumer's
exact capacity, that the oracle is recoverable from the raw 25-dim field at 0.973-0.985 and
from 978's frozen z_world at only 0.648-0.674 -- and that an UNTRAINED z_world (a random
projection through the same encoder channel) scores 0.682-0.695, at or above the warmed arm
on every seed. So H-B is eliminated, H-C stands at its registered gate, but the gloss "THIS
latent's LEARNED geometry blocks the mapping" is NOT established: the encoder's actual input
is the full 250-dim world_state (the 25-dim field is 10% of it), and no run carries a 32-dim
reference OF THAT INPUT known to be able to clear the bar. The autopsy's routing (section 8):
a parallel two-leg portfolio on the same dataset, same adapter, same standardiser -- NOT a
power bump (the positive control converged at CE 0.025-0.041; a 1002b is REFUSED by the
autopsy's own section 7).

=== WHAT "BANKED 1002 DATASET" MEANS HERE, STATED HONESTLY ===

V3-EXQ-1002 persisted no observations and no encoder weights (978 saved none either); its
arm-fingerprint cells bank the READOUTS, not the data. What IS banked is the RECIPE, and it is
deterministic: `_collect_episodes` drives the oracle / a random policy from env seeds
seed*1000+ep with the policies seeded from `seed`, so re-collecting reproduces the identical
step-for-step dataset (verified at authoring time: this driver's rawfield_ceiling arm on seed
42 reproduces 1002's 0.9846 held-out agreement to four decimals). Leg 1 therefore needs NO
warmup at all. Leg 2 needs the frozen 978-OFF latent, which -- as in 1002 -- is REPRODUCED by
re-running the imported 978 warmup (60 P0a + 200 P0 + 90 P1 episodes per seed, every constant
imported from x734/x808/x724 exactly as 1002 did). That is the same warmup FAMILY as 978/1002,
not a new one; it is the one unavoidable cost of the corroborator and it is paid once per seed
and shared by every leg-2 arm. "No new rollouts" in the campaign plan is an over-compression;
"no new warmup family and no ree_core change" (the autopsy's own cost statement) is exact.
The frozen latent's dataset and warmup reproduce 1002's bit-exactly (verified on landing:
warmup scalars match to full double precision on 3/3 seeds, held-out split identical,
machine_class identical in both manifests -- linux-x86_64-py3.10-torch2.12.0+cpu; the
"machine class differ" clause in red-team F8 was wrong and is corrected here). Only the
adapter initialisation draw is re-drawn per this run, so the in-run zworld_off_diag baseline
is expected NEAR 0.648-0.674, not AT it, and every leg-2 threshold is applied to the in-run
baseline, never to 1002's numbers.

=== THE INPUT LEG 1 COMPRESSES, MEASURED AT AUTHORING TIME ===

world_state (use_proxy_fields=True, D3_hazard_free): 250 dims = local_view 5x5x7 one-hot [0:175]
+ contamination_view [175:200] + hazard_field_view [200:225] + resource_field_view [225:250].
On seed 42's training split: numeric rank 135, participation ratio 14.1, top-32 PCs carry
0.865 of the variance (0.72 at 16, 0.97 at 64; 78 PCs for 0.99).

THE ORACLE READS FIVE OF THOSE 250 COORDINATES (red-team F3). local_view_greedy's action is the
argmax over the five destination cells of the agent-centred field view -- world_state indices
{232, 236, 237, 238, 242} -- so "the compression is lossy" (variance) does NOT by itself make
leg 1's null realisable: a projection could drop 13.5% of the variance and keep that 5-dim
subspace exactly. Two preconditions therefore certify two different things, and both are
measured in-run: `ws250_compression_lossy` (the projection is not injective on the input --
variance explained by the top-32 PCs <= 0.99) and `ws250_projection_mixes_decision_subspace`
(the LARGEST retained norm ||W^T e_j|| over the five decision directions <= 0.95, i.e. no
decision coordinate survives the compression intact). Measured on seed 42: PCA-32 retains
0.23-0.29 of each decision direction, the random orthonormal projection 0.29-0.39 -- both
compressions genuinely mix the decision subspace, so the null IS realisable, and it is
realised in practice (the two projections land 0.10 apart). A projection of the 25-dim field
ALONE is rank-preserving (any injective R^25 -> R^32 composed with Linear(32,128) spans exactly
Linear(25,128)); the autopsy DROPPED that version and this driver does not run it.

=== ARMS (10 per seed) ===

  ANCHORS (replicate 1002 in-run, so every comparison is paired on the same dataset):
    rawfield_ceiling         adapter on the raw 25-dim field. POSITIVE CONTROL, gates the
                             instrument (>= 0.60). 1002: 0.985/0.980/0.973.
    zworld_untrained_diag    adapter on an UNTRAINED z_world (same construction, warmup
                             skipped), train-split diagonal z-score. 1002: 0.688/0.682/0.695.
    zworld_off_diag          adapter on the frozen 978-OFF z_world, diagonal z-score --
                             1002's verdict arm and LEG 2's BASELINE. 1002: 0.670/0.674/0.648.
  LEG 1 (world_state -> 32 dims, then the SAME diagonal z-score instrument as 1002):
    ws250_full               the uncompressed 250-dim input through the adapter. LEG 1's
                             READINESS ANCHOR (must reach the bar). NOT verdict-bearing.
    ws250_randproj           a seeded random ORTHONORMAL 250 -> 32 projection. The leg's
                             task-agnostic untrained-compression control. A SECOND independent
                             draw is fitted inside the same cell and reported
                             (`oracle_action_agreement_randproj_draw_b`), so projection-draw
                             variance is recorded separately from seed variance (red-team F6).
    ws250_pca                PCA-32 fitted on the training split -- the variance-optimal
                             task-agnostic linear compression, i.e. the best a
                             reconstruction-trained LINEAR encoder could do.
                             ** LEG 1 VERDICT ARM (H-E). **
  LEG 2 (frozen 978-OFF z_world under invertible linear re-bases fitted on the train split):
    zworld_off_fielddecode   ** LEG 2 VERDICT ARM. ** Invertible 32x32 map: axes 1-25 are the
                             least-squares linear decode of the 25-dim resource field from the
                             latent (fitted on the train split), axes 26-32 an orthonormal basis
                             of the orthogonal complement; then the diagonal z-score. The
                             decision variable (the field's five destination cells) becomes an
                             explicit coordinate. Alongside, the CLOSED-FORM readout
                             `linear_decode_oracle_agreement`: local_view_greedy's own rule
                             applied to the decoded field, no adapter -- how much of the
                             mapping the linearly decodable content supports.
    zworld_off_zca           full-covariance (ZCA) whitening. Unsupervised. Secondary.
    zworld_off_ldawhiten     within-class (Fisher) whitening. Secondary (measured at authoring
                             time to be FLAT on both leg-1 projections, so a flat here is
                             expected and uninformative on its own -- red-team F2).
    zworld_untrained_fielddecode  the verdict re-basis on the UNTRAINED latent. Enters the
                             grid: a lift that appears on BOTH latents is a property of the
                             encoder channel's output geometry, not of the warmup (F1).

WHY LINEAR, INVERTIBLE RE-BASES ONLY, AND WHAT THAT MEANS FOR H-C. Any invertible linear map
R of the input is absorbable by the adapter's first layer (W' = W R^-1), so the FUNCTION CLASS
the capacity-matched adapter can represent is IDENTICAL across leg-2 arms, and the information
content is identical BY ALGEBRA (an invertible affine map cannot change what is linearly
decodable; the held-out field-decode r2 before/after is recorded as a numerics witness, NOT as
a gate -- red-team F5 showed such a gate is arithmetically incapable of firing). A lift can
therefore come ONLY from the learning dynamics -- which is exactly what the registered H-C leg
asserts ("the geometry does not make the relationships accessible under the actual learning
dynamics"). 1002 already showed the dynamics are scale-sensitive on this latent (standardised
0.670 vs unstandardised 0.535 on the same arm). A NON-linear re-basis would add capacity and
could not be called information-preserving in the relevant sense; it is deliberately not run.

WHY THE VERDICT RE-BASIS IS THE DECODE BASIS AND NOT A WHITENING (red-team F2, applied).
Measured at authoring time on seed 42 with this adapter: within-class whitening moves NEITHER
leg-1 projection (randproj 0.786 -> 0.783; pca 0.884 -> 0.881), and a random invertible
reweighting of the raw field at condition number 20 / 100 moves the reader by only -0.025 /
-0.047. A blind whitening of the latent would therefore land FLAT for reasons that say nothing
about the latent, and the declared null would be unattributable. The decode basis is different
in kind: it is fitted to put the decision variable on explicit axes, and it comes with its own
closed-form comparator. If the closed-form decode-then-oracle agreement sits at the diag
baseline, the linearly decodable CONTENT only supports baseline agreement -- the block is
information (the encoder discards it, as V3-EXQ-948 found), not geometry, and no re-basis can
lift it: FLAT is then attributable. If the closed-form reader sits well ABOVE the baseline while
the adapter on the decoded basis does not reach it, that is a reader/instrument shortfall on a
basis where the content is explicit -- its own cell, never a hypothesis verdict.

=== THE CAPACITY MATCH (inherited from 1002, unchanged) ===

The adapter IS `x734.PPOPolicyNet` at `x734.PPO_TRUNK_HIDDEN` -- the exact class 978
instantiated as its reader -- imported via x1002._make_adapter, never redefined. The z_world
arms match the consumer exactly (21,381 action-path params at in_dim 32); leg-1 arms at
in_dim 32 match too; rawfield_ceiling (in_dim 25) and ws250_full (in_dim 250) differ in the
first layer only, in the conservative and the non-verdict-bearing direction respectively (the
250-dim anchor has MORE first-layer capacity and is a readiness anchor, never a verdict arm --
so its extra capacity cannot manufacture an H-E verdict). Measured per arm into
`capacity_match`. The adapter's init is re-seeded from the cell seed immediately before every
fit (red-team F8), so at a given seed every arm's adapter starts from the same draw and the
paired differentials carry no init-draw component.

=== PRE-REGISTERED BARS, AND THE RANGE THEY SIT INSIDE ===

All three 1002 bar constants are IMPORTED, not re-typed: AGREEMENT_BAR 0.80,
AGREEMENT_ELEVATION_MIN 0.20 over the strongest TRIVIAL predictor (max(majority-class,
repeat-previous-executed-action), measured 0.566-0.580), UNTRAINED_CONTROL_MARGIN 0.10.

LEG 1 verdict (H-E): `pca_clears` = ws250_pca reaches AGREEMENT_BAR AND AGREEMENT_ELEVATION_MIN
on >= SEED_MAJORITY of seeds. The autopsy's stated criterion is the 0.80 bar. The chip brief's
generic bar also lists ">= 0.10 over the paired untrained control"; for LEG 1 that margin
(ws250_pca minus ws250_randproj, paired per seed) is REPORTED AS ATTRIBUTION, not entered as a
conjunct of the H-E verdict, for a stated reason: the margin in 1002 protected the H-B label
from an untrained projection that also cleared the bar (which would defeat attribution to the
WARMUP). In leg 1 a random compression ALSO clearing the bar does not weaken the H-E
elimination -- it strengthens it (ANY task-agnostic 32-dim linear compression of the input
supports the mapping) -- so gating the verdict on beating the random control would refuse the
leg's strongest possible answer. Authoring-time measurement, seed 42, full scale, this adapter:
ws250_full 0.940, ws250_pca 0.884, ws250_randproj 0.786 (two independent draws 0.786/0.788),
so the random control sits AT the bar and the PCA-minus-random margin is 0.098 -- which is why
that margin is reported with its per-seed value rather than thresholded into the verdict.
The grid distinguishes "PCA clears, random also clears" from "PCA clears, random does not";
the two labels share the H-E verdict, and the label boundary sits within ~one seed-SD of the
random control's authoring-time value (red-team F6) -- the continuous margins are recorded per
seed so the reading survives whichever side the control lands on.

LEG 2 verdict (H-C corroborator): per seed, lift = zworld_off_fielddecode minus zworld_off_diag
(paired on the identical frozen latent, dataset and adapter init). REBASIS_LIFT_MIN =
UNTRAINED_CONTROL_MARGIN = 0.10 (the same margin 1002 used for its paired differential, so the
two runs' margins are commensurable): lift >= 0.10 -> LIFT; 0.05 <= lift < 0.10 -> MARGINAL;
-0.05 < lift < 0.05 -> FLAT; lift <= -0.05 -> DEGRADE. The seed-majority class decides; no
majority -> inconsistent. Three further per-seed quantities enter the grid: the untrained
pair's lift class (zworld_untrained_fielddecode minus zworld_untrained_diag), the CONTENT class
(closed-form linear_decode_oracle_agreement minus the diag baseline, same bands), and the
READER SHORTFALL (closed-form minus the adapter on the decoded basis, >= 0.10 = short).
`rebased_clears` (the verdict arm reaching bar + elevation) is carried into the label.
Authoring-time range: the baseline sits at 0.648-0.674 (1002) so the lift criterion has ~0.33
of headroom against the 0.10 it needs (dv_headroom_rebasis_lift, margin 2.0, computed from
THIS run's baseline) and the raw-field ceiling (0.97+) bounds it from above. No full-scale
authoring-time measurement of the latent arms exists (the Mac warmup could not finish under
load); leg 2 is fully differential and self-witnessing in-run.

SEEDS. The pre-registered majority is SEED_MAJORITY = 2 of the three lineage seeds. A run
started with fewer than SEED_MAJORITY seeds cannot form a majority, so it cannot emit a
hypothesis verdict on either leg (red-team F4: the count-based criterion would otherwise route
a single seed that cleared the bar to H-E CONFIRMED); such a run routes both legs to
`insufficient_seeds_for_majority` and reports its readouts only.

OUTCOME tracks whether the PORTFOLIO ADJUDICATED both legs, and the labels carry the science:
PASS iff both C_leg1_adjudicated and C_leg2_adjudicated (each leg's readiness gates green and
its verdict cell is not `undetermined`). Every branch of both legs is a result; a null on
either leg is informative by construction (see NULL TABLE). A FAIL therefore means "at least
one leg could not be adjudicated", never "the hypothesis lost".

=== NULL TABLE (declared up front) ===

LEG 1 -- `_adjudicate_leg1(gate_green, leg1_ready, pca_clears, rand_clears)`:
  not ready                      -> leg1_not_ready / undetermined. The uncompressed input did
                                    not reach the bar, a compression precondition failed, the
                                    instrument gate is red, or too few seeds. NOT an H-E verdict.
  pca_clears & rand_clears       -> ws250_any_linear_compression_supports_mapping / H-E
                                    ELIMINATED (strong). ANY task-agnostic 32-dim linear
                                    compression of the encoder's input supports the mapping, and
                                    both REE-channel latents (untrained 0.69, trained 0.67 in
                                    1002) score BELOW a random linear projection -- the deficit
                                    is what the encoder channel DOES, not the dimensionality.
  pca_clears & not rand_clears   -> ws250_optimal_linear_compression_supports_mapping / H-E
                                    ELIMINATED. The variance-optimal task-agnostic compression
                                    preserves what the task needs; the channel does not.
  not pca_clears & rand_clears   -> ws250_random_clears_pca_does_not / undetermined. PCA
                                    discards task-relevant LOW-variance directions that a
                                    random projection keeps; H-E is not confirmed (a working
                                    compression exists) but the optimal-compression reading
                                    fails. Rare; own cell so it is never folded into H-E.
  neither clears                 -> ws250_linear_compression_fails_bar / H-E CONFIRMED. Neither
                                    TASK-AGNOSTIC 32-dim linear compression (variance-optimal
                                    or random) of the actual input supports the mapping at this
                                    capacity, although the uncompressed input does. This is the
                                    REGISTERED hypothesis, not a universal: a task-INFORMED
                                    coordinate selection (the raw field itself, an arm of this
                                    run at 0.97+) trivially supports it (red-team F3b). The
                                    deficit is a task-agnostic channel-input bound; the
                                    confirmed H-C leg is weakened to its narrow gate sense.
LEG 2 -- `_adjudicate_leg2(gate_green, leg2_ready, lift_class, rebased_clears,
                            untrained_lift_class, content_class, reader_short)`:
  not ready                      -> leg2_not_ready / undetermined (encoder untrained,
                                    collapsed latent, no lift headroom, too few seeds).
  LIFT & untrained also LIFT     -> rebasis_lifts_trained_and_untrained_alike / H-C
                                    CORROBORATED (channel-level): the decision-relevant
                                    re-basis makes the mapping accessible on BOTH latents, so
                                    the geometry block is a property of the encoder channel's
                                    output, not of this latent's learned geometry (consistent
                                    with H-D; the registry's caveat stands).
  LIFT & rebased_clears          -> rebasis_restores_oracle_mapping / H-C CORROBORATED
                                    (strong): the information was there and a linear re-basis
                                    alone makes it accessible -- the interface fix is a
                                    re-basis, not a side channel.
  LIFT & not rebased_clears      -> rebasis_lifts_accessibility_below_bar / H-C CORROBORATED.
  MARGINAL                       -> rebasis_marginal_lift / undetermined.
  FLAT or DEGRADE, content NOT a LIFT
                                 -> linear_content_ceiling_information_not_geometry / H-C
                                    WEAKENED: the closed-form reader on the decoded field
                                    reaches only ~baseline, so the linearly decodable content
                                    itself supports no more than the adapter already found --
                                    the block is INFORMATION (the encoder discards the
                                    decision-relevant part of its input; V3-EXQ-948), not
                                    geometry, and no re-basis can lift it.
  FLAT or DEGRADE, content LIFT & reader short
                                 -> decoded_content_exceeds_reader_instrument_shortfall /
                                    undetermined: the closed-form reader finds the mapping in
                                    the decoded field and the trained adapter does not even on
                                    that basis -- an instrument finding about the reader, not
                                    a geometry verdict. Own cell, never H-C either way.
  FLAT or DEGRADE, content LIFT, reader not short
                                 -> rebasis_flat_content_lift_inconsistent / undetermined
                                    (band-edge; report the continuous quantities).
  inconsistent across seeds      -> rebasis_effect_inconsistent_across_seeds / undetermined.

The joint label is `<leg1_label>__<leg2_label>` and `hypothesis_verdict` reads
"H-E: <eliminated|confirmed|undetermined>; H-C: <corroborated|weakened|undetermined>".

=== DV-SYMMETRY INVARIANCE (per arm) ===

DV = mean over held-out states of 1[argmax(adapter logits) == oracle action]. Symmetry group:
permutation of held-out states; consistent relabelling of the action set. Every arm's
manipulation is a change of the adapter's INPUT REPRESENTATION -- a different feature vector
per state (raw field / 250-dim input / random projection / PCA projection / untrained latent /
frozen latent / decode-basis / ZCA / LDA re-basis). None is a broadcast constant on the logits,
a monotone rescaling of a ranked quantity, or a permutation of interchangeable units, so no arm
is invariant. Leg 2 deserves one more sentence: an invertible linear re-basis IS absorbable by
the adapter's first layer, but the adapter is TRAINED from a fresh (seeded) init on the
re-based input, so the manipulation reaches the DV through the optimisation trajectory -- the
path is open, and it is the path H-C is about. The closed-form decode-then-oracle readout has
NO adapter in its path at all: it is a witness of content, not of dynamics.

Corollary, per readiness precondition: `adapter_capacity_sufficient_on_raw_field` certifies the
instrument for every arm; `ws250_full_input_reaches_bar`, `ws250_compression_lossy` and
`ws250_projection_mixes_decision_subspace` certify leg 1 only; `zworld_encoder_trained_in_p0` /
`zworld_not_collapsed` certify the z_world arms. Readiness is worst-seed while the verdict is
seed-majority (red-team F7): an unlucky seed on an anchor abstains the leg, never mislabels it.

=== KNOWN OPEN SUBSTRATE DEFECTS OVERLAPPING THIS DRIVER (skill Step 2.5c) ===

SD-018 (`amend_implemented_pending_validation`, DEGRADING; stack.py, zworld_p0.py,
agent.py::compute_resource_proximity_loss) is exercised by the warmup -- recorded, not blocking,
and reproducing 978's warmup DEFECTS INCLUDED is the design requirement (the latent must be
978's). Open CORRUPTING entries naming ree_core/agent.py or utils/config.py
(mode-governance-engagement, SD-082) name mechanisms (salience-coordinator affinity clamping, a
lateral-PFC bias head) this run's DV never reads: the DV is a supervised decode of a frozen
latent / raw observation and never consults action selection. Held constant across arms.

=== REPAIR RECORD (red-team, opus, 2026-09-07: CONTESTED, no BLOCKING; one pass, not iterated) ===

  F1 (highest) the leg-2 grid ignored the untrained-latent pair, so a lift appearing on both
     latents would have read "H-C corroborated (strong)" -- the exact caveat leg 2 exists to
     resolve. APPLIED: `untrained_lift_class` enters `_adjudicate_leg2`; a lift on both routes
     to its own channel-level cell.
  F2 (highest) the declared null (FLAT) had no positive control for the manipulation class.
     VERIFIED at authoring time -- within-class whitening is flat on both leg-1 projections and
     a cond-20/100 raw-field reweighting moves the reader by only 0.02-0.05 -- so a blind
     whitening verdict arm would have landed FLAT uninformatively. APPLIED: the verdict re-basis
     is now the decode basis, with the closed-form decode-then-oracle CONTENT witness and the
     reader-shortfall cell in the grid; whitening arms stay as secondaries.
  F3 (high) `ws250_compression_lossy` certified variance loss, not null realisability (the oracle
     reads 5 coordinates). APPLIED: `ws250_projection_mixes_decision_subspace` added (max
     retained decision-direction norm <= 0.95; measured 0.29 / 0.39), retention recorded per
     projection, the lossy gate re-described as what it certifies (non-injectivity).
  F3b (medium) the H-E-CONFIRMED wording asserted a universal the raw-field arm refutes.
     APPLIED: "task-agnostic" throughout; the raw field named as the task-informed exception.
     The `C_leg1_adjudicated` degeneracy flag now also requires the random control green.
  F4 (medium) SEED_MAJORITY 2 against a free --seeds could route a single cleared seed to H-E
     CONFIRMED. APPLIED: `seeds_sufficient` gate; fewer than SEED_MAJORITY seeds -> both legs
     `insufficient_seeds_for_majority`, no verdict.
  F5 (medium) `rebasis_preserves_linear_decode` is invariant by algebra and
     `rebasis_well_conditioned` capped at ~1e3 by the relative ridge -- neither could fire.
     APPLIED: both DEMOTED from gates to recorded witnesses (`rebasis_witnesses`); the reported
     condition number is now that of the transform T (sqrt of the scatter's), as described.
  F6 (medium) label boundary at ~1 seed-SD of the random control; one projection draw per seed.
     APPLIED: a second independent draw is fitted and reported inside the randproj cell; the
     boundary is disclosed (above), the verdict is unaffected by which label fires.
  F7 (low) worst-seed readiness vs majority verdict: DISCLOSED, kept (conservative direction).
  F8 (low) adapter init unseeded, so the leg-2 baseline is a reproduction not a replay.
     APPLIED: `reset_all_rng(seed)` immediately before every adapter fit (same init across arms
     at a seed); the "anchors ARE 1002's arms" wording softened; all thresholds in-run.
  F9 (none) DV ceiling < 1.0 vs dv_bounds (0,1): <= 0.015 overstatement, DISCLOSED, kept.
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

from experiment_protocol import emit_outcome  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402
from experiments._metrics import (  # noqa: E402
    P0NotReady,
    dv_headroom_check,
    p0_readiness_gate,
)
from experiments._lib.arm_fingerprint import arm_cell, reset_all_rng  # noqa: E402
from experiments._lib.capability_eval import (  # noqa: E402
    COMPETENCE_RESOURCE_FLOOR,
    LocalViewGreedyPolicy,
    Policy,
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
# IMPORTED, NEVER REDEFINED: the dataset recipe, the adapter, the standardiser, the scoring
# and the 978-warmup reproduction are 1002's own functions, so this run's anchors are
# reproductions of 1002's arms and the two legs are read with 1002's instrument.
import experiments.v3_exq_1002_zworld_actor_adequacy_oracle_adapter as x1002  # noqa: E402
import experiments.v3_exq_724_competence_localization_diagnostic as x724  # noqa: E402
import experiments.v3_exq_734_env_difficulty_competence_recovery_sweep as x734  # noqa: E402
import experiments.v3_exq_737_ree_latent_policy_head_competence_probe as x737  # noqa: E402
import experiments.v3_exq_808_return_decomposition_objective_misspecification as x808  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_1008_zworld_adequacy_portfolio_ws250_rebasis"
QUEUE_ID = "V3-EXQ-1008"
EXPERIMENT_PURPOSE = "diagnostic"
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"
CLAIM_IDS: List[str] = []          # deliberately empty -- see the docstring's Claims line
HYPOTHESIS_QID = x1002.HYPOTHESIS_QID
DEVICE = x1002.DEVICE

SEEDS: List[int] = list(x1002.SEEDS)              # [42, 43, 44] -- 978/1002 lineage seeds

# ---- IMPORTED CONSTANTS (never re-typed) --------------------------------------------------
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
SEED_MAJORITY = x1002.SEED_MAJORITY                # 2 of 3

AGREEMENT_BAR = x1002.AGREEMENT_BAR                        # 0.80
AGREEMENT_ELEVATION_MIN = x1002.AGREEMENT_ELEVATION_MIN    # 0.20
UNTRAINED_CONTROL_MARGIN = x1002.UNTRAINED_CONTROL_MARGIN  # 0.10
RAW_FIELD_CONTROL_FLOOR = x1002.RAW_FIELD_CONTROL_FLOOR    # 0.60
PARTICIPATION_RATIO_FLOOR = x1002.PARTICIPATION_RATIO_FLOOR  # 2.0

# ---- NEW pre-registered constants ---------------------------------------------------------
# The compression target: z_world's own width, so a leg-1 arm is exactly "the encoder's
# input at the encoder's output dimensionality". Asserted equal to the latent's dim at runtime.
PROJECTION_DIM = 32
# Leg 1 readiness: the UNCOMPRESSED input must reach the verdict bar with this adapter, or a
# compressed arm's failure is an instrument reading, not a compression reading. Same constant
# as the verdict bar on purpose (measured 0.940 on seed 42 at authoring time).
WS250_FULL_INPUT_FLOOR = AGREEMENT_BAR
# Leg 1 readiness (i): the 32-dim compression must NOT be injective on the input (variance
# explained by the top-32 PCs of the training split strictly below this). Measured 0.865.
WS250_VARIANCE_EXPLAINED_CEILING = 0.99
# Leg 1 readiness (ii): the compression must MIX the oracle's decision subspace -- the largest
# retained norm ||W^T e_j|| over the five destination-cell coordinates below this. Measured
# 0.29 (PCA) / 0.39 (random) on seed 42. Without this, "lossy" would not make the null
# realisable (red-team F3).
DECISION_SUBSPACE_RETENTION_CEILING = 0.95
# Leg 2 lift classes (see docstring). LIFT_MIN == UNTRAINED_CONTROL_MARGIN so the two runs'
# paired differentials are commensurable. The same bands classify the CONTENT margin
# (closed-form decode-then-oracle minus baseline) and READER_SHORTFALL_MIN marks the reader
# falling short of the closed-form comparator on the decoded basis.
REBASIS_LIFT_MIN = UNTRAINED_CONTROL_MARGIN                # 0.10
REBASIS_FLAT_EPS = 0.05
READER_SHORTFALL_MIN = REBASIS_LIFT_MIN                    # 0.10
LDA_RIDGE_FRAC = 1.0e-3            # ridge on the within-class scatter, as a fraction of its
                                   # largest eigenvalue (a well-conditioned inverse sqrt)
WHITEN_EPS = 1.0e-6
DECODE_RIDGE = 1.0e-6              # tiny ridge on the field-decode least squares

# The oracle's decision coordinates inside world_state: view cell [2+dx, 2+dy] for each move
# delta in LocalViewGreedyPolicy._DELTAS, flattened row-major inside resource_field_view, which
# occupies world_state[225:250] (CausalGridWorldV2 layout; SplitEncoder.RESOURCE_FIELD_SLICE).
RESOURCE_FIELD_OFFSET = 225
_ORACLE_DELTAS = dict(LocalViewGreedyPolicy._DELTAS)   # {0:(-1,0),1:(1,0),2:(0,-1),3:(0,1),4:(0,0)}
DECISION_VIEW_INDICES = {a: (2 + dx) * 5 + (2 + dy) for a, (dx, dy) in _ORACLE_DELTAS.items()}
DECISION_WORLD_STATE_INDICES = sorted(RESOURCE_FIELD_OFFSET + v for v in DECISION_VIEW_INDICES.values())
ORACLE_FLAT_EPS = 1.0e-3           # LocalViewGreedyPolicy's default flat_eps

# ---- arms --------------------------------------------------------------------------------
ARM_RAW = x1002.ARM_RAW                 # "rawfield_ceiling"
ARM_UNT_DIAG = "zworld_untrained_diag"
ARM_OFF_DIAG = "zworld_off_diag"
ARM_WS_FULL = "ws250_full"
ARM_WS_RAND = "ws250_randproj"
ARM_WS_PCA = "ws250_pca"
ARM_OFF_FD = "zworld_off_fielddecode"
ARM_OFF_ZCA = "zworld_off_zca"
ARM_OFF_LDA = "zworld_off_ldawhiten"
ARM_UNT_FD = "zworld_untrained_fielddecode"

# Order per seed. The frozen latents are produced in the *_diag cells and shared with their
# siblings, so the diag cell must precede its re-based siblings.
ARM_IDS = [ARM_RAW, ARM_WS_FULL, ARM_WS_RAND, ARM_WS_PCA,
           ARM_UNT_DIAG, ARM_UNT_FD, ARM_OFF_DIAG, ARM_OFF_FD, ARM_OFF_ZCA, ARM_OFF_LDA]
LEG1_ARMS = [ARM_WS_FULL, ARM_WS_RAND, ARM_WS_PCA]
LEG2_ARMS = [ARM_OFF_DIAG, ARM_OFF_FD, ARM_OFF_ZCA, ARM_OFF_LDA]
LEG1_VERDICT_ARM = ARM_WS_PCA
LEG1_CONTROL_ARM = ARM_WS_RAND
LEG1_ANCHOR_ARM = ARM_WS_FULL
LEG2_VERDICT_ARM = ARM_OFF_FD
LEG2_BASELINE_ARM = ARM_OFF_DIAG
Z_ARMS = [ARM_UNT_DIAG, ARM_UNT_FD, ARM_OFF_DIAG, ARM_OFF_FD, ARM_OFF_ZCA, ARM_OFF_LDA]
TRAINED_Z_ARMS = [ARM_OFF_DIAG, ARM_OFF_FD, ARM_OFF_ZCA, ARM_OFF_LDA]
REBASED_ARMS = [ARM_UNT_FD, ARM_OFF_FD, ARM_OFF_ZCA, ARM_OFF_LDA]
FIELDDECODE_ARMS = [ARM_UNT_FD, ARM_OFF_FD]
PROJECTED_ARMS = [ARM_WS_RAND, ARM_WS_PCA]
ANCHOR_ARMS = [ARM_RAW, ARM_UNT_DIAG, ARM_OFF_DIAG]   # get 1002's unstandardised secondary

DRY_RUN_SEEDS = [42]
DRY_RUN_ZWORLD_P0 = x1002.DRY_RUN_ZWORLD_P0
DRY_RUN_P0 = x1002.DRY_RUN_P0
DRY_RUN_P1 = x1002.DRY_RUN_P1
DRY_RUN_EVAL = x1002.DRY_RUN_EVAL
DRY_RUN_STEPS = x1002.DRY_RUN_STEPS
DRY_RUN_BC_EPISODES = x1002.DRY_RUN_BC_EPISODES
DRY_RUN_BC_RANDOM_EPISODES = x1002.DRY_RUN_BC_RANDOM_EPISODES
DRY_RUN_ADAPTER_PASSES = x1002.DRY_RUN_ADAPTER_PASSES

# The computation of every cell lives in x1002 (+ the modules it imports) and in this driver.
# Folding those files into the substrate hash keeps the fingerprint honest: an edit to 1002's
# adapter or dataset recipe correctly refuses a stale match. ONE mode for every cell (driver
# included) so the run records a single substrate identity across all 30 cells.
_EXTRA_SUBSTRATE_PATHS = [Path(m.__file__) for m in (x1002, x724, x734, x737, x808)]

_ZG = ZGoalStreamAccumulator()


# --------------------------------------------------------------------------------------
# PRECONDITIONS: 1002's six, unchanged, plus three new ones scoped to the arms they certify
# --------------------------------------------------------------------------------------
def _arm_ctx(aid: str) -> Dict[str, Any]:
    return {"id": aid,
            "has_encoder": aid in Z_ARMS,
            "trained_encoder": aid in TRAINED_Z_ARMS,
            "leg1": aid in LEG1_ARMS,
            "projected": aid in PROJECTED_ARMS,
            "rebased": aid in REBASED_ARMS}


def _arm_contexts() -> List[Dict[str, Any]]:
    return [_arm_ctx(a) for a in ARM_IDS]


NEW_PRECONDITION_SPECS = [
    PreconditionSpec(
        name="ws250_full_input_reaches_bar",
        description=(
            "LEG 1 READINESS ANCHOR. The UNCOMPRESSED 250-dim world_state, through the same "
            "adapter/optimiser/passes/standardiser, must reach the verdict bar on the worst "
            "seed. If the instrument cannot read the input uncompressed, a 250 -> 32 arm's "
            "failure is an instrument reading, not a compression reading, and H-E must not "
            "be confirmed from it. Measured 0.940 on seed 42 at authoring time."),
        control="ws250_full worst-seed held-out oracle_action_agreement",
        threshold=float(WS250_FULL_INPUT_FLOOR), direction="lower", kind="readiness",
        applies_to=lambda ctx: bool(ctx["leg1"]),
        applies_note="certifies the 250-dim input channel; z_world and raw-field arms do not read it",
        structural_max=lambda ctx: 1.0,
    ),
    PreconditionSpec(
        name="ws250_compression_lossy",
        description=(
            "LEG 1 NON-INJECTIVITY. Variance explained by the top-32 principal components of "
            "the training-split world_state must sit BELOW this ceiling (worst = largest "
            "seed), i.e. the 250 -> 32 projection is not injective on the input the run "
            "actually sees. This certifies only that the compression discards SOMETHING; "
            "whether it discards the oracle's decision coordinates is certified separately by "
            "ws250_projection_mixes_decision_subspace (red-team F3). Measured 0.865 on seed 42."),
        control="largest-seed variance explained by the top-32 PCs of the train-split world_state",
        threshold=float(WS250_VARIANCE_EXPLAINED_CEILING), direction="upper", kind="readiness",
        applies_to=lambda ctx: bool(ctx["projected"]),
        applies_note="only the two projected leg-1 arms compress; ws250_full and the rest do not",
        structural_min=lambda ctx: 0.0,
    ),
    PreconditionSpec(
        name="ws250_projection_mixes_decision_subspace",
        description=(
            "LEG 1 NULL REALISABILITY. local_view_greedy reads exactly five world_state "
            "coordinates (the field's destination cells, indices %s). The projection under "
            "test must NOT preserve any of them intact: the largest retained norm ||W^T e_j|| "
            "over those five directions (worst = largest across this arm's seeds) must sit "
            "below this ceiling. A projection that kept the decision subspace exactly would "
            "clear the bar by construction and leg 1's null would be unrealisable. Measured "
            "on seed 42: PCA-32 0.23-0.29, random orthonormal 0.29-0.39."
            % (DECISION_WORLD_STATE_INDICES,)),
        control="largest retained norm of a decision-cell unit vector under this arm's projection",
        threshold=float(DECISION_SUBSPACE_RETENTION_CEILING), direction="upper", kind="readiness",
        applies_to=lambda ctx: bool(ctx["projected"]),
        applies_note="only the projected leg-1 arms have a projection matrix",
        structural_min=lambda ctx: 0.0,
    ),
]
PRECONDITION_SPECS = list(x1002.PRECONDITION_SPECS) + NEW_PRECONDITION_SPECS


# --------------------------------------------------------------------------------------
# FEATURE TRANSFORMS -- every one fitted on the TRAINING split only, applied to all splits
# --------------------------------------------------------------------------------------
class _DiagZ:
    """1002's train-split diagonal z-score, verbatim (the anchors' instrument)."""
    kind = "train_split_zscore"

    def __init__(self, x_tr: torch.Tensor) -> None:
        self._st = x1002._fit_standardiser(x_tr)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x1002._apply_standardiser(x, self._st)

    def report(self) -> Dict[str, Any]:
        return dict(x1002._standardiser_report(self._st), kind=self.kind)


class _GuardedDiagZ:
    """Diagonal z-score with a CONSTANT-DIMENSION GUARD, for the leg-1 arms.

    1002's standardiser clamps a fitted std at STANDARDISER_EPS, which maps a dimension that is
    constant on the training split to (x - mean) / 1e-6 -- harmless where the dimension stays
    constant, but a one-hot entity channel that never appears in the train split and appears
    once in the test split would be scaled by 1e6 and saturate the adapter on that row. The
    250-dim input has many such channels (wall / resource one-hots at particular view cells).
    A train-constant dimension carries no information on the split the adapter learns from, so
    the honest treatment is to ZERO it (scale 0), which is what this does. Identical to _DiagZ
    on any input with no train-constant dimensions (the projected arms, by construction)."""
    kind = "train_split_zscore_constant_dim_guard"

    def __init__(self, x_tr: torch.Tensor, eps: float = x1002.STANDARDISER_EPS) -> None:
        self.n = int(x_tr.shape[0])
        self.mean = x_tr.mean(dim=0, keepdim=True) if self.n else None
        sd = x_tr.std(dim=0, unbiased=False, keepdim=True) if self.n else None
        self._eps = float(eps)
        if sd is not None:
            live = (sd >= self._eps)
            self.scale = torch.where(live, 1.0 / torch.clamp(sd, min=self._eps),
                                     torch.zeros_like(sd))
            self.n_constant = int((~live).sum().item())
            flat = sd.reshape(-1)
            self._sd_stats = (float(flat.min()), float(flat.median()), float(flat.max()))
        else:
            self.scale = None
            self.n_constant = None
            self._sd_stats = (None, None, None)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.mean is None or int(x.shape[0]) == 0:
            return x
        return (x - self.mean) * self.scale

    def report(self) -> Dict[str, Any]:
        return {"kind": self.kind, "fitted": self.mean is not None, "n_train_rows": self.n,
                "n_dims": (int(self.mean.shape[1]) if self.mean is not None else None),
                "n_train_constant_dims_zeroed": self.n_constant,
                "raw_per_dim_std_min": self._sd_stats[0],
                "raw_per_dim_std_median": self._sd_stats[1],
                "raw_per_dim_std_max": self._sd_stats[2], "eps": self._eps}


def _decision_subspace_retention(W: torch.Tensor) -> Dict[str, Any]:
    """||W^T e_j|| for the oracle's five decision coordinates and for the whole field block.
    1.0 = that coordinate survives the projection intact; 0.0 = it is discarded entirely."""
    if int(W.shape[0]) <= max(DECISION_WORLD_STATE_INDICES):
        return {"applicable": False}
    ret = {int(j): float(W[j, :].norm()) for j in DECISION_WORLD_STATE_INDICES}
    field = [float(W[j, :].norm()) for j in range(RESOURCE_FIELD_OFFSET,
                                                    RESOURCE_FIELD_OFFSET + RESOURCE_FIELD_DIM)]
    return {"applicable": True, "decision_cell_retained_norm": ret,
            "decision_cell_retained_norm_max": max(ret.values()),
            "decision_cell_retained_norm_min": min(ret.values()),
            "field_block_retained_norm_mean": float(np.mean(field)),
            "field_block_retained_norm_min": float(np.min(field))}


class _LinearProjection:
    """x -> (x - mean_train) @ W, W: [in_dim, PROJECTION_DIM]. Followed by _GuardedDiagZ."""

    def __init__(self, x_tr: torch.Tensor, W: torch.Tensor, kind: str,
                 extra: Optional[Dict[str, Any]] = None) -> None:
        self.kind = kind
        self.mean = x_tr.mean(dim=0, keepdim=True) if int(x_tr.shape[0]) else None
        self.W = W
        self._extra = dict(extra or {})
        self._extra["decision_subspace_retention"] = _decision_subspace_retention(W)
        p_tr = self._project(x_tr)
        self._z = _GuardedDiagZ(p_tr)

    def _project(self, x: torch.Tensor) -> torch.Tensor:
        if int(x.shape[0]) == 0:
            return torch.zeros(0, int(self.W.shape[1]))
        m = self.mean if self.mean is not None else 0.0
        return (x - m) @ self.W

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self._z(self._project(x))

    @property
    def retention(self) -> Dict[str, Any]:
        return self._extra["decision_subspace_retention"]

    def report(self) -> Dict[str, Any]:
        out = {"kind": self.kind, "in_dim": int(self.W.shape[0]),
               "out_dim": int(self.W.shape[1]), "post_projection": self._z.report()}
        out.update(self._extra)
        return out


def _world_state_pca_stats(x_tr: torch.Tensor, k: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Top-k principal directions of the training split plus the spectrum stats the manifest
    records (rank, participation ratio, variance explained at k and at a few references)."""
    n = int(x_tr.shape[0])
    if n < 2:
        return torch.eye(int(x_tr.shape[1]))[:, :k], {"fitted": False}
    xc = x_tr - x_tr.mean(dim=0, keepdim=True)
    _u, s, vt = torch.linalg.svd(xc, full_matrices=False)
    var = (s ** 2) / float(n - 1)
    tot = float(var.sum())
    cum = torch.cumsum(var, 0) / max(tot, 1e-12)
    rank = int((s > s[0] * 1e-6).sum().item()) if s.numel() else 0
    pr = float(var.sum() ** 2 / max(float((var ** 2).sum()), 1e-12))

    def _at(j: int) -> Optional[float]:
        return float(cum[j - 1]) if cum.numel() >= j else (float(cum[-1]) if cum.numel() else None)

    stats = {"fitted": True, "n_train_rows": n, "in_dim": int(x_tr.shape[1]),
             "numeric_rank": rank, "participation_ratio": pr,
             "variance_explained_at_k": _at(k), "k": int(k),
             "variance_explained_at_16": _at(16), "variance_explained_at_64": _at(64),
             "n_components_for_99pct": int((cum < 0.99).sum().item()) + 1,
             "top_singular_values": [float(v) for v in s[:8]]}
    W = vt[:k].T.contiguous()
    if W.shape[1] < k:   # fewer rows than k: pad with zero columns so the width is stable
        W = torch.cat([W, torch.zeros(W.shape[0], k - W.shape[1])], dim=1)
    return W, stats


def _random_orthonormal(in_dim: int, k: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(int(seed))
    q, _r = torch.linalg.qr(torch.randn(in_dim, k, generator=g))
    return q.contiguous()


def _inv_sqrt_psd(mat: torch.Tensor, ridge: float) -> Tuple[torch.Tensor, Dict[str, Any]]:
    lam, v = torch.linalg.eigh(mat)
    lam = lam.clamp(min=0.0)
    t = v @ torch.diag(1.0 / torch.sqrt(lam + ridge)) @ v.T
    cond_scatter = float((lam.max() + ridge) / (lam.min() + ridge)) if lam.numel() else 1.0
    return t, {"eig_min": float(lam.min()) if lam.numel() else None,
               "eig_max": float(lam.max()) if lam.numel() else None,
               "ridge": float(ridge),
               "condition_number_scatter": cond_scatter,
               # cond(T) = sqrt(cond(scatter)) for T = scatter^(-1/2) -- red-team F5
               "condition_number": float(np.sqrt(cond_scatter))}


class _ZCAWhiten:
    """Full-covariance (ZCA) whitening fitted on the train split. Invertible linear map."""
    kind = "zca_whitening"

    def __init__(self, x_tr: torch.Tensor) -> None:
        n = int(x_tr.shape[0])
        self.mean = x_tr.mean(dim=0, keepdim=True) if n else None
        if n >= 2:
            c = x_tr - self.mean
            cov = (c.T @ c) / float(n - 1)
            self.T, self._st = _inv_sqrt_psd(cov, WHITEN_EPS)
        else:
            self.T, self._st = torch.eye(int(x_tr.shape[1])), {"condition_number": 1.0}

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.mean is None or int(x.shape[0]) == 0:
            return x
        return (x - self.mean) @ self.T

    @property
    def condition_number(self) -> float:
        return float(self._st.get("condition_number", 1.0))

    def report(self) -> Dict[str, Any]:
        return dict(self._st, kind=self.kind, fitted=self.mean is not None)


class _LDAWhiten:
    """Within-class (Fisher) whitening: T = Sw^(-1/2), Sw the pooled within-class scatter of
    the oracle's action classes on the train split (+ a small relative ridge). Invertible
    linear map. SECONDARY arm only (see the docstring: measured flat on both leg-1 projections)."""
    kind = "within_class_whitening"

    def __init__(self, x_tr: torch.Tensor, y_tr: torch.Tensor, action_dim: int) -> None:
        n = int(x_tr.shape[0])
        self.mean = x_tr.mean(dim=0, keepdim=True) if n else None
        d = int(x_tr.shape[1])
        if n >= 2:
            sw = torch.zeros(d, d)
            n_used = 0
            for k in range(int(action_dim)):
                xk = x_tr[y_tr == k]
                if int(xk.shape[0]) < 2:
                    continue
                ck = xk - xk.mean(dim=0, keepdim=True)
                sw += ck.T @ ck
                n_used += int(xk.shape[0])
            sw = sw / float(max(1, n_used - int(action_dim)))
            lam_max = float(torch.linalg.eigvalsh(sw).max()) if d else 0.0
            ridge = LDA_RIDGE_FRAC * lam_max + WHITEN_EPS
            self.T, self._st = _inv_sqrt_psd(sw, ridge)
            self._st["n_rows_in_scatter"] = n_used
        else:
            self.T, self._st = torch.eye(d), {"condition_number": 1.0}

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.mean is None or int(x.shape[0]) == 0:
            return x
        return (x - self.mean) @ self.T

    @property
    def condition_number(self) -> float:
        return float(self._st.get("condition_number", 1.0))

    def report(self) -> Dict[str, Any]:
        return dict(self._st, kind=self.kind, fitted=self.mean is not None,
                    ridge_frac=float(LDA_RIDGE_FRAC))


class _FieldDecodeRebasis:
    """LEG 2's VERDICT re-basis: an invertible linear map of the latent whose first
    RESOURCE_FIELD_DIM axes are the train-split least-squares linear decode of the resource
    field, and whose remaining axes are an orthonormal basis of the orthogonal complement of
    the decode directions. Followed by the diagonal z-score (the same instrument as the
    baseline). `decode(z)` returns the closed-form field estimate (with intercept) for the
    decode-then-oracle witness."""
    kind = "field_decode_rebasis"

    def __init__(self, z_tr: torch.Tensor, field_tr: torch.Tensor) -> None:
        n, d = int(z_tr.shape[0]), int(z_tr.shape[1])
        self.d = d
        self.mean = z_tr.mean(dim=0, keepdim=True) if n else None
        if n >= 2:
            zc = z_tr - self.mean
            fm = field_tr.mean(dim=0, keepdim=True)
            # ridge least squares: W_dec = (Zc^T Zc + ridge I)^-1 Zc^T (F - fm)
            gram = zc.T @ zc + float(DECODE_RIDGE) * torch.eye(d)
            self.W_dec = torch.linalg.solve(gram, zc.T @ (field_tr - fm))   # [d, 25]
            self.b_dec = fm                                                 # [1, 25]
            # orthonormal basis of col(W_dec) and of its complement
            u, s, _vt = torch.linalg.svd(self.W_dec, full_matrices=True)
            rank = int((s > (s[0] * 1e-6 if s.numel() else 0.0)).sum().item()) if s.numel() else 0
            self.rank = rank
            self.Q_perp = u[:, rank:]                                        # [d, d-rank]
            self.T = torch.cat([self.W_dec, self.Q_perp], dim=1)            # [d, 25 + d - rank]
            sv = torch.linalg.svdvals(self.T)
            self._cond = float(sv.max() / max(float(sv.min()), 1e-12)) if sv.numel() else 1.0
            self._decode_r2_train = _r2(zc @ self.W_dec + fm, field_tr)
        else:
            self.W_dec, self.b_dec = torch.zeros(d, RESOURCE_FIELD_DIM), torch.zeros(1, RESOURCE_FIELD_DIM)
            self.rank, self.Q_perp, self.T = 0, torch.eye(d), torch.eye(d)
            self._cond, self._decode_r2_train = 1.0, None
        self._z = _DiagZ(self._rebase(z_tr))

    def _rebase(self, z: torch.Tensor) -> torch.Tensor:
        if self.mean is None or int(z.shape[0]) == 0:
            return z
        return (z - self.mean) @ self.T

    def __call__(self, z: torch.Tensor) -> torch.Tensor:
        return self._z(self._rebase(z))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Closed-form field estimate [n, 25] (no adapter)."""
        if self.mean is None or int(z.shape[0]) == 0:
            return torch.zeros(int(z.shape[0]), RESOURCE_FIELD_DIM)
        return (z - self.mean) @ self.W_dec + self.b_dec

    @property
    def condition_number(self) -> float:
        return float(self._cond)

    def report(self) -> Dict[str, Any]:
        return {"kind": self.kind, "fitted": self.mean is not None, "latent_dim": self.d,
                "decode_rank": int(self.rank), "rebased_dim": int(self.T.shape[1]),
                "condition_number": float(self._cond), "decode_ridge": float(DECODE_RIDGE),
                "decode_r2_train": self._decode_r2_train,
                "post_rebasis": self._z.report()}


def _r2(pred: torch.Tensor, target: torch.Tensor) -> Optional[float]:
    if int(target.shape[0]) < 2:
        return None
    ss_res = float(((target - pred) ** 2).sum())
    ss_tot = float(((target - target.mean(dim=0, keepdim=True)) ** 2).sum())
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0.0 else None


def _lin_decode_r2(x_tr: torch.Tensor, t_tr: torch.Tensor,
                   x_te: torch.Tensor, t_te: torch.Tensor) -> Optional[float]:
    """Held-out r2 of a least-squares linear decode of `t` from `x` (the 978 decode probe's
    statistic). Under an invertible affine map of `x` it is invariant BY ALGEBRA; it is recorded
    for the re-based arms as a numerics witness, never as a gate (red-team F5)."""
    if int(x_tr.shape[0]) < 2 or int(x_te.shape[0]) < 2:
        return None
    ones_tr = torch.ones(x_tr.shape[0], 1)
    ones_te = torch.ones(x_te.shape[0], 1)
    w = torch.linalg.lstsq(torch.cat([x_tr, ones_tr], 1), t_tr).solution
    pred = torch.cat([x_te, ones_te], 1) @ w
    return _r2(pred, t_te)


def _oracle_rule_actions(field: torch.Tensor) -> torch.Tensor:
    """local_view_greedy's decision rule applied to a (decoded) 25-dim field, deterministic
    form: argmax over the destination cells of the move deltas. Where the window is flat
    (max - min < ORACLE_FLAT_EPS) the real oracle draws a random non-stay move that no reader
    can predict; this rule still returns the argmax there, so the closed-form agreement is a
    LOWER bound on what a perfect reader of the decoded content could achieve, by the flat-window
    fraction (recorded)."""
    if int(field.shape[0]) == 0:
        return torch.zeros(0, dtype=torch.long)
    acts = sorted(DECISION_VIEW_INDICES.keys())
    cols = torch.stack([field[:, DECISION_VIEW_INDICES[a]] for a in acts], dim=1)   # [n, 5]
    best = torch.argmax(cols, dim=1)
    return torch.tensor([acts[int(i)] for i in best], dtype=torch.long)


def _decode_oracle_agreement(field_pred: torch.Tensor, y: torch.Tensor) -> Dict[str, Any]:
    if int(field_pred.shape[0]) == 0:
        return {"linear_decode_oracle_agreement": None, "decoded_flat_window_frac": None}
    pred_a = _oracle_rule_actions(field_pred)
    acts = sorted(DECISION_VIEW_INDICES.keys())
    cols = torch.stack([field_pred[:, DECISION_VIEW_INDICES[a]] for a in acts], dim=1)
    flat = ((cols.max(dim=1).values - cols.min(dim=1).values) < ORACLE_FLAT_EPS)
    return {"linear_decode_oracle_agreement": float((pred_a == y).float().mean().item()),
            "decoded_flat_window_frac": float(flat.float().mean().item())}


class _TransformedNet:
    """The adapter with its fitted transform welded on the front (rolled-out secondary)."""

    def __init__(self, net, f: Callable[[torch.Tensor], torch.Tensor]) -> None:
        self.net = net
        self._f = f

    def __call__(self, x: torch.Tensor):
        return self.net(self._f(x))


class WorldStateAdapterPolicy(Policy):
    """Greedy rollout of an adapter reading (a transform of) the 250-dim world_state."""

    name = "ws250_adapter"

    def __init__(self, tnet: _TransformedNet) -> None:
        self.tnet = tnet

    def act(self, env: Any, obs_dict: Dict[str, Any]) -> int:
        w = obs_dict["world_state"]
        w = (w if isinstance(w, torch.Tensor) else torch.as_tensor(w)).reshape(1, -1).float()
        with torch.no_grad():
            logits, _v = self.tnet(w)
        if not torch.isfinite(logits).all():
            return int(np.random.randint(0, int(env.action_dim)))
        return int(torch.argmax(logits.reshape(-1), dim=-1).item())


def _world_state_features(episodes: List[Dict[str, Any]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """The FULL exteroceptive channel the encoder reads, row-aligned with 1002's extractors
    (episodes in order, steps in order)."""
    xs, ys = [], []
    for ep in episodes:
        for obs, lab in zip(ep["obs"], ep["labels"]):
            w = obs.get("world_state")
            if w is None:
                raise KeyError("obs_dict has no 'world_state' -- the encoder's input channel is absent")
            w = w if isinstance(w, torch.Tensor) else torch.as_tensor(w)
            xs.append(w.reshape(-1).float())
            ys.append(int(lab))
    if not xs:
        return torch.zeros(0, 1), torch.zeros(0, dtype=torch.long)
    return torch.stack(xs), torch.tensor(ys, dtype=torch.long)


# --------------------------------------------------------------------------------------
# CONFIG SLICE: 1002's, extended per arm
# --------------------------------------------------------------------------------------
def _config_slice(base: Dict[str, Any], arm_id: str) -> Dict[str, Any]:
    d = dict(base)
    d["arm_id"] = arm_id
    d["arm_input"] = ("resource_field_view" if arm_id == ARM_RAW
                      else "world_state" if arm_id in LEG1_ARMS else "z_world")
    d["arm_projection"] = ("none" if arm_id not in PROJECTED_ARMS
                           else "random_orthonormal" if arm_id == ARM_WS_RAND else "pca")
    d["arm_projection_dim"] = (int(PROJECTION_DIM) if arm_id in PROJECTED_ARMS else None)
    d["arm_rebasis"] = ("zca" if arm_id == ARM_OFF_ZCA
                        else "lda_whiten" if arm_id == ARM_OFF_LDA
                        else "field_decode" if arm_id in FIELDDECODE_ARMS
                        else "diag_zscore")
    d["arm_constant_dim_guard"] = bool(arm_id in LEG1_ARMS)
    d["adapter_init_reseeded_per_fit"] = True
    if arm_id in Z_ARMS:
        d["arm_p0a_field_weight"] = 0.0
        d["arm_warmup_skipped"] = bool(arm_id in (ARM_UNT_DIAG, ARM_UNT_FD))
    if arm_id in REBASED_ARMS:
        d["lda_ridge_frac"] = float(LDA_RIDGE_FRAC)
        d["whiten_eps"] = float(WHITEN_EPS)
        d["decode_ridge"] = float(DECODE_RIDGE)
    return d


# --------------------------------------------------------------------------------------
# CELLS
# --------------------------------------------------------------------------------------
def _rollout_row(policy: Policy, seed: int, env_kwargs: Dict[str, Any],
                 eval_eps: int, steps: int) -> Dict[str, Any]:
    ev = evaluate_seed(policy, x734._make_env(seed, env_kwargs), eval_eps, steps)
    return {"cloned_foraging_competence": float(ev["foraging_competence"]),
            "cloned_competence_supra_floor": bool(ev["competence_supra_floor"]),
            "cloned_survival_horizon": float(ev["survival_horizon"]),
            "cloned_death_rate": float(ev["death_rate"]),
            "cloned_per_episode_resources": list(ev["per_episode_resources"])}


def _fit_and_score(arm_id: str, seed: int, data: Dict[str, Any], action_dim: int,
                   passes: int, x_tr: torch.Tensor, y_tr: torch.Tensor,
                   x_te: torch.Tensor, y_te: torch.Tensor,
                   xr_te: torch.Tensor, yr_te: torch.Tensor,
                   transform: Any, unstd_secondary: bool) -> Tuple[Any, Dict[str, Any]]:
    """Transform -> (re-seeded) adapter fit -> every agreement readout. Shared by all arms."""
    xs_tr, xs_te, xsr_te = transform(x_tr), transform(x_te), transform(xr_te)
    # Same adapter init draw for every arm at a given seed (red-team F8): the paired
    # differentials then carry no init-draw component.
    reset_all_rng(seed)
    net, train_stats = x1002._train_adapter(xs_tr, y_tr, action_dim, passes, seed, arm_id)
    row = {
        "cell_id": "%s|seed%d" % (arm_id, seed),
        "arm_id": arm_id,
        "seed": int(seed),
        "feature_dim": int(xs_tr.shape[1]) if int(xs_tr.shape[0]) else None,
        "input_dim": int(x_tr.shape[1]) if int(x_tr.shape[0]) else None,
        "capacity_match": x1002._capacity_report(net, int(xs_tr.shape[1]), action_dim),
        "adapter_training": train_stats,
        "feature_transform": transform.report(),
    }
    row.update(x1002._score_cell(net, xs_tr, y_tr, xs_te, y_te, xsr_te, yr_te, action_dim,
                                 prev_te=x1002._prev_action_vector(data["test"])))
    if unstd_secondary:
        reset_all_rng(seed)
        row.update(x1002._unstandardised_secondary(x_tr, y_tr, x_te, y_te, action_dim,
                                                   passes, seed, arm_id))
    return net, row


def _print_verdict(row: Dict[str, Any]) -> None:
    agree = row.get("oracle_action_agreement") or 0.0
    elev = row.get("agreement_elevation") or 0.0
    print("verdict: %s" % ("PASS" if (agree >= AGREEMENT_BAR and elev >= AGREEMENT_ELEVATION_MIN)
                           else "FAIL"), flush=True)


def _warm_off_agent(seed: int, env_kwargs: Dict[str, Any], zworld_p0: int, p0: int, p1: int,
                    steps: int, dry_run: bool) -> Tuple[Any, Dict[str, Any], Dict[str, Any]]:
    """Reproduce 978's OFF-arm warmup exactly as 1002 did (imports, not re-definitions)."""
    warm_env = x734._make_env(seed, env_kwargs)
    agent = x1002._make_agent(warm_env)
    before = latent_stack_snapshot(agent)
    stats = x734._train_all_on_agent(
        agent, warm_env, seed=seed, p0_episodes=p0, p1_episodes=p1,
        steps_per_episode=steps, rung_id=RUNG_ID, total_denominator=(p0 + p1),
        zworld_p0_episodes=zworld_p0,
        zworld_p0_env=(x734._make_env(seed, env_kwargs) if zworld_p0 > 0 else None),
        zworld_p0_dry_run=dry_run,
        zworld_p0_resource_field_weight=0.0,   # 978's OFF arm
    )
    guard = latent_stack_weight_delta(agent, before)
    return agent, stats, guard


def _z_feats(agent, data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    z_tr, _ = x1002._zworld_features(agent, data["train"])
    z_te, _ = x1002._zworld_features(agent, data["test"])
    zr_te, _ = x1002._zworld_features(agent, data["random"])
    return {"tr": z_tr, "te": z_te, "r": zr_te}


def run_cell(arm_id: str, seed: int, data: Dict[str, Any], feats: Dict[str, Any],
             frozen: Dict[str, Any], action_dim: int, sched: Dict[str, int],
             env_kwargs: Dict[str, Any], cfg_base: Dict[str, Any],
             dry_run: bool) -> Dict[str, Any]:
    """One (arm x seed) cell. `feats` carries the seed's raw-field / world_state matrices
    (shared, read-only); `frozen` carries the seed's frozen agents + z features once the
    *_diag cells have produced them."""
    print("Seed %d Condition %s:%s" % (seed, RUNG_ID, arm_id), flush=True)
    y_tr, y_te, yr_te = feats["y_tr"], feats["y_te"], feats["yr_te"]
    field_tr, field_te = feats["field"]["tr"], feats["field"]["te"]
    shared_reason = None
    if arm_id in (ARM_OFF_FD, ARM_OFF_ZCA, ARM_OFF_LDA):
        shared_reason = "frozen_off_agent_and_latent_shared_from_%s_cell" % ARM_OFF_DIAG
    elif arm_id == ARM_UNT_FD:
        shared_reason = "untrained_agent_and_latent_shared_from_%s_cell" % ARM_UNT_DIAG
    with arm_cell(seed, config_slice=_config_slice(cfg_base, arm_id),
                  script_path=Path(__file__), config_slice_declared=True,
                  extra_substrate_paths=_EXTRA_SUBSTRATE_PATHS,
                  extra_ineligible_reasons=([shared_reason] if shared_reason else None)) as cell:
        extra: Dict[str, Any] = {}
        agent = None
        # ---- inputs + transform per arm --------------------------------------------
        if arm_id == ARM_RAW:
            x = feats["field"]
            transform = _DiagZ(x["tr"])
        elif arm_id in LEG1_ARMS:
            x = feats["ws"]
            if arm_id == ARM_WS_FULL:
                transform = _GuardedDiagZ(x["tr"])
            elif arm_id == ARM_WS_RAND:
                W = _random_orthonormal(int(x["tr"].shape[1]), PROJECTION_DIM, seed)
                transform = _LinearProjection(x["tr"], W, "random_orthonormal_projection",
                                              {"projection_seed": int(seed)})
                extra["decision_subspace_retention_max"] = transform.retention.get(
                    "decision_cell_retained_norm_max")
            else:
                W, pstats = _world_state_pca_stats(x["tr"], PROJECTION_DIM)
                transform = _LinearProjection(x["tr"], W, "pca_projection", {"pca": pstats})
                extra["world_state_pca_stats"] = pstats
                extra["decision_subspace_retention_max"] = transform.retention.get(
                    "decision_cell_retained_norm_max")
        else:
            key = "off" if arm_id in TRAINED_Z_ARMS else "unt"
            if arm_id in (ARM_OFF_DIAG, ARM_UNT_DIAG):
                warm_env = x734._make_env(seed, env_kwargs)
                if arm_id == ARM_UNT_DIAG:
                    print("  [%s] no warmup (negative control)" % arm_id, flush=True)
                    agent = x1002._make_agent(warm_env)
                    before = latent_stack_snapshot(agent)
                    stats = {}
                    guard = latent_stack_weight_delta(agent, before)
                else:
                    agent, stats, guard = _warm_off_agent(
                        seed, env_kwargs, sched["zworld_p0"], sched["p0"], sched["p1"],
                        sched["steps"], dry_run)
                frozen[key] = {"agent": agent, "guard": guard,
                               "p0a": (stats or {}).get("zworld_p0", {}) or {},
                               "z": _z_feats(agent, data)}
            fz = frozen[key]
            agent = fz["agent"]
            x = fz["z"]
            extra["warmup_skipped"] = bool(key == "unt")
            extra["zworld_weight_delta"] = fz["guard"]
            extra["zworld_participation_ratio"] = x1002._participation_ratio(x["tr"])
            p0a = fz["p0a"]
            extra["p0a"] = {"ran": bool(p0a.get("p0a_ran")),
                            "resource_field_weight": p0a.get("p0a_resource_field_weight"),
                            "used_resource_field_head": bool(p0a.get("p0a_used_resource_field_head")),
                            "resource_field_holdout": p0a.get("p0a_resource_field_holdout")}
            if arm_id in (ARM_OFF_DIAG, ARM_UNT_DIAG):
                transform = _DiagZ(x["tr"])
            elif arm_id == ARM_OFF_ZCA:
                transform = _ZCAWhiten(x["tr"])
            elif arm_id == ARM_OFF_LDA:
                transform = _LDAWhiten(x["tr"], y_tr, action_dim)
            else:
                transform = _FieldDecodeRebasis(x["tr"], field_tr)
                # The CLOSED-FORM content witness: the oracle's own rule on the decoded field.
                extra.update(_decode_oracle_agreement(transform.decode(x["te"]), y_te))
                extra["linear_decode_oracle_agreement_train"] = _decode_oracle_agreement(
                    transform.decode(x["tr"]), y_tr)["linear_decode_oracle_agreement"]
                extra["field_decode_r2_heldout"] = _r2(transform.decode(x["te"]), field_te)
            if arm_id in REBASED_ARMS:
                # Recorded WITNESSES, not gates (red-team F5): invariant by algebra.
                r2_raw = _lin_decode_r2(x["tr"], field_tr, x["te"], field_te)
                r2_reb = _lin_decode_r2(transform(x["tr"]), field_tr, transform(x["te"]), field_te)
                extra["rebasis_condition_number"] = float(transform.condition_number)
                extra["field_decode_r2_raw_latent"] = r2_raw
                extra["field_decode_r2_rebased_latent"] = r2_reb
                extra["field_decode_r2_abs_delta"] = (abs(r2_reb - r2_raw)
                                                      if (r2_raw is not None and r2_reb is not None)
                                                      else None)
        if arm_id in Z_ARMS and int(x["tr"].shape[0]):
            assert int(x["tr"].shape[1]) == PROJECTION_DIM, (
                "z_world dim %d != PROJECTION_DIM %d -- leg 1 would not be 'the input at the "
                "encoder's output width'" % (int(x["tr"].shape[1]), PROJECTION_DIM))
        # ---- adapter + readouts -------------------------------------------------------
        net, row = _fit_and_score(arm_id, seed, data, action_dim, sched["passes"],
                                  x["tr"], y_tr, x["te"], y_te, x["r"], yr_te,
                                  transform, unstd_secondary=(arm_id in ANCHOR_ARMS))
        row.update(extra)
        if arm_id == ARM_WS_RAND:
            # A SECOND independent projection draw, same cell, same data, same adapter
            # (red-team F6): records projection-draw variance separately from seed variance.
            W_b = _random_orthonormal(int(x["tr"].shape[1]), PROJECTION_DIM, seed + 7919)
            t_b = _LinearProjection(x["tr"], W_b, "random_orthonormal_projection_draw_b",
                                    {"projection_seed": int(seed + 7919)})
            _net_b, row_b = _fit_and_score(arm_id + ":draw_b", seed, data, action_dim,
                                           sched["passes"], x["tr"], y_tr, x["te"], y_te,
                                           x["r"], yr_te, t_b, unstd_secondary=False)
            row["oracle_action_agreement_randproj_draw_b"] = row_b.get("oracle_action_agreement")
            row["agreement_elevation_randproj_draw_b"] = row_b.get("agreement_elevation")
            row["decision_subspace_retention_max_draw_b"] = t_b.retention.get(
                "decision_cell_retained_norm_max")
            row["projection_draw_b_seed"] = int(seed + 7919)
        # ---- rolled-out competence secondary (never verdict-bearing) ------------------
        tnet = _TransformedNet(net, transform)
        if arm_id == ARM_RAW:
            pol: Policy = x1002.RawFieldAdapterPolicy(tnet)
        elif arm_id in LEG1_ARMS:
            pol = WorldStateAdapterPolicy(tnet)
        else:
            pol = x737.LatentPPOEvalPolicy(tnet, agent)
        row.update(_rollout_row(pol, seed, env_kwargs, sched["eval_eps"], sched["steps"]))
        if agent is not None:
            _ZG.observe(agent)
        cell.stamp(row)
    _print_verdict(row)
    return row


# --------------------------------------------------------------------------------------
# THE VERDICT GRIDS -- pure functions, contract-tested by --self-test
# --------------------------------------------------------------------------------------
def _adjudicate_leg1(gate_green: bool, leg1_ready: bool, pca_clears: bool,
                     rand_clears: bool) -> Tuple[str, str]:
    """(label, verdict) for H-E. See the docstring's NULL TABLE."""
    if not gate_green or not leg1_ready:
        return ("leg1_not_ready", "undetermined")
    if pca_clears and rand_clears:
        return ("ws250_any_linear_compression_supports_mapping", "H-E-eliminated")
    if pca_clears:
        return ("ws250_optimal_linear_compression_supports_mapping", "H-E-eliminated")
    if rand_clears:
        return ("ws250_random_clears_pca_does_not", "undetermined")
    return ("ws250_linear_compression_fails_bar", "H-E-confirmed")


LIFT_CLASSES = ("lift", "marginal", "flat", "degrade")


def _lift_class(delta: Optional[float]) -> Optional[str]:
    if delta is None:
        return None
    if delta >= REBASIS_LIFT_MIN:
        return "lift"
    if delta >= REBASIS_FLAT_EPS:
        return "marginal"
    if delta > -REBASIS_FLAT_EPS:
        return "flat"
    return "degrade"


def _majority_class(classes: List[Optional[str]], majority: int) -> str:
    """The lift class held by >= `majority` seeds, else 'inconsistent'."""
    for c in LIFT_CLASSES:
        if sum(1 for k in classes if k == c) >= majority:
            return c
    return "inconsistent"


def _adjudicate_leg2(gate_green: bool, leg2_ready: bool, lift_class: str,
                     rebased_clears: bool, untrained_lift_class: str,
                     content_class: str, reader_short: bool) -> Tuple[str, str]:
    """(label, verdict) for the H-C corroborator. See the docstring's NULL TABLE.

    `lift_class`            seed-majority class of (verdict arm - diag baseline)
    `untrained_lift_class`  the same re-basis on the UNTRAINED latent (red-team F1)
    `content_class`         seed-majority class of (closed-form decode-then-oracle - baseline)
    `reader_short`          the adapter on the decoded basis sits >= READER_SHORTFALL_MIN
                            below the closed-form reader on the seed majority (red-team F2)
    """
    if not gate_green or not leg2_ready:
        return ("leg2_not_ready", "undetermined")
    if lift_class == "lift":
        if untrained_lift_class == "lift":
            return ("rebasis_lifts_trained_and_untrained_alike", "H-C-corroborated-channel-level")
        if rebased_clears:
            return ("rebasis_restores_oracle_mapping", "H-C-corroborated-strong")
        return ("rebasis_lifts_accessibility_below_bar", "H-C-corroborated")
    if lift_class == "marginal":
        return ("rebasis_marginal_lift", "undetermined")
    if lift_class in ("flat", "degrade"):
        if content_class == "lift":
            if reader_short:
                return ("decoded_content_exceeds_reader_instrument_shortfall", "undetermined")
            return ("rebasis_flat_content_lift_inconsistent", "undetermined")
        return ("linear_content_ceiling_information_not_geometry", "H-C-weakened")
    return ("rebasis_effect_inconsistent_across_seeds", "undetermined")


def _combine(leg1: Tuple[str, str], leg2: Tuple[str, str]) -> Tuple[str, str, str]:
    """(outcome, joint label, hypothesis_verdict). PASS iff BOTH legs adjudicated."""
    l1, v1 = leg1
    l2, v2 = leg2
    he = ("eliminated" if v1 == "H-E-eliminated" else "confirmed" if v1 == "H-E-confirmed"
          else "undetermined")
    hc = ("corroborated" if v2.startswith("H-C-corroborated")
          else "weakened" if v2 == "H-C-weakened" else "undetermined")
    outcome = "PASS" if (he != "undetermined" and hc != "undetermined") else "FAIL"
    return (outcome, "%s__%s" % (l1, l2), "H-E: %s; H-C: %s" % (he, hc))


_L2 = dict(gate_green=True, leg2_ready=True, untrained_lift_class="flat",
           content_class="flat", reader_short=False)
_SELF_TEST_ROWS = [
    {"name": "leg1 gate red -> not ready",
     "fn": "leg1", "in": dict(gate_green=False, leg1_ready=True, pca_clears=True, rand_clears=True),
     "want": ("leg1_not_ready", "undetermined")},
    {"name": "leg1 anchor red -> not ready even if pca clears",
     "fn": "leg1", "in": dict(gate_green=True, leg1_ready=False, pca_clears=True, rand_clears=False),
     "want": ("leg1_not_ready", "undetermined")},
    {"name": "leg1 pca+rand clear -> H-E eliminated (strong)",
     "fn": "leg1", "in": dict(gate_green=True, leg1_ready=True, pca_clears=True, rand_clears=True),
     "want": ("ws250_any_linear_compression_supports_mapping", "H-E-eliminated")},
    {"name": "leg1 authoring-time seed-42 shape: pca 0.884 clears, rand 0.786 does not",
     "fn": "leg1", "in": dict(gate_green=True, leg1_ready=True, pca_clears=True, rand_clears=False),
     "want": ("ws250_optimal_linear_compression_supports_mapping", "H-E-eliminated")},
    {"name": "leg1 rand clears, pca not -> undetermined, never H-E",
     "fn": "leg1", "in": dict(gate_green=True, leg1_ready=True, pca_clears=False, rand_clears=True),
     "want": ("ws250_random_clears_pca_does_not", "undetermined")},
    {"name": "leg1 neither clears -> H-E confirmed",
     "fn": "leg1", "in": dict(gate_green=True, leg1_ready=True, pca_clears=False, rand_clears=False),
     "want": ("ws250_linear_compression_fails_bar", "H-E-confirmed")},
    {"name": "leg2 not ready",
     "fn": "leg2", "in": dict(_L2, leg2_ready=False, lift_class="lift", rebased_clears=True),
     "want": ("leg2_not_ready", "undetermined")},
    {"name": "leg2 lift + clears, untrained flat -> corroborated strong",
     "fn": "leg2", "in": dict(_L2, lift_class="lift", rebased_clears=True),
     "want": ("rebasis_restores_oracle_mapping", "H-C-corroborated-strong")},
    {"name": "leg2 lift below bar, untrained flat -> corroborated",
     "fn": "leg2", "in": dict(_L2, lift_class="lift", rebased_clears=False),
     "want": ("rebasis_lifts_accessibility_below_bar", "H-C-corroborated")},
    {"name": "leg2 lift on BOTH latents -> channel-level, never strong (red-team F1)",
     "fn": "leg2", "in": dict(_L2, lift_class="lift", rebased_clears=True, untrained_lift_class="lift"),
     "want": ("rebasis_lifts_trained_and_untrained_alike", "H-C-corroborated-channel-level")},
    {"name": "leg2 flat, content flat -> information ceiling, weakened",
     "fn": "leg2", "in": dict(_L2, lift_class="flat", rebased_clears=False),
     "want": ("linear_content_ceiling_information_not_geometry", "H-C-weakened")},
    {"name": "leg2 degrade, content marginal -> information ceiling, weakened",
     "fn": "leg2", "in": dict(_L2, lift_class="degrade", rebased_clears=False, content_class="marginal"),
     "want": ("linear_content_ceiling_information_not_geometry", "H-C-weakened")},
    {"name": "leg2 flat, content lift, reader short -> instrument shortfall (red-team F2)",
     "fn": "leg2", "in": dict(_L2, lift_class="flat", rebased_clears=False, content_class="lift", reader_short=True),
     "want": ("decoded_content_exceeds_reader_instrument_shortfall", "undetermined")},
    {"name": "leg2 flat, content lift, reader not short -> inconsistent, undetermined",
     "fn": "leg2", "in": dict(_L2, lift_class="flat", rebased_clears=False, content_class="lift"),
     "want": ("rebasis_flat_content_lift_inconsistent", "undetermined")},
    {"name": "leg2 marginal -> undetermined",
     "fn": "leg2", "in": dict(_L2, lift_class="marginal", rebased_clears=False),
     "want": ("rebasis_marginal_lift", "undetermined")},
    {"name": "leg2 inconsistent -> undetermined",
     "fn": "leg2", "in": dict(_L2, lift_class="inconsistent", rebased_clears=False),
     "want": ("rebasis_effect_inconsistent_across_seeds", "undetermined")},
    {"name": "lift class boundaries",
     "fn": "lift", "in": [0.10, 0.099, 0.05, 0.049, -0.049, -0.05, None],
     "want": ["lift", "marginal", "marginal", "flat", "flat", "degrade", None]},
    {"name": "majority class 2/3",
     "fn": "maj", "in": (["lift", "flat", "lift"], 2), "want": "lift"},
    {"name": "majority class none",
     "fn": "maj", "in": (["lift", "flat", "degrade"], 2), "want": "inconsistent"},
    {"name": "oracle rule on a decoded field: argmax over the five destination cells",
     "fn": "rule", "in": None, "want": None},
]


def _run_self_test() -> int:
    n_fail = 0
    for row in _SELF_TEST_ROWS:
        if row["fn"] == "leg1":
            got: Any = _adjudicate_leg1(**row["in"])
            want: Any = row["want"]
        elif row["fn"] == "leg2":
            got = _adjudicate_leg2(**row["in"])
            want = row["want"]
        elif row["fn"] == "lift":
            got = [_lift_class(d) for d in row["in"]]
            want = row["want"]
        elif row["fn"] == "maj":
            got = _majority_class(*row["in"])
            want = row["want"]
        else:
            # one synthetic field per action: put the peak on that action's destination cell
            f = torch.zeros(5, RESOURCE_FIELD_DIM)
            acts = sorted(DECISION_VIEW_INDICES.keys())
            for i, a in enumerate(acts):
                f[i, DECISION_VIEW_INDICES[a]] = 1.0
            got = [int(v) for v in _oracle_rule_actions(f)]
            want = acts
        ok = (got == want)
        n_fail += 0 if ok else 1
        print("[self-test] %-4s %s" % ("OK" if ok else "FAIL", row["name"]), flush=True)
        if not ok:
            print("            got  %r" % (got,), flush=True)
            print("            want %r" % (want,), flush=True)
    # Whole-cube invariants.
    for gg in (True, False):
        for rd in (True, False):
            for pc in (True, False):
                for rc in (True, False):
                    lab, v = _adjudicate_leg1(gg, rd, pc, rc)
                    if v == "H-E-eliminated" and not (gg and rd and pc):
                        print("[self-test] FAIL H-E eliminated without pca_clears under green gates", flush=True)
                        n_fail += 1
                    if v == "H-E-confirmed" and (pc or rc or not (gg and rd)):
                        print("[self-test] FAIL H-E confirmed with a clearing compression or red gate", flush=True)
                        n_fail += 1
                    if not (gg and rd) and v != "undetermined":
                        print("[self-test] FAIL leg1 verdict on a red gate", flush=True)
                        n_fail += 1
            for lc in LIFT_CLASSES + ("inconsistent",):
                for ulc in LIFT_CLASSES + ("inconsistent",):
                    for cc in LIFT_CLASSES + ("inconsistent",):
                        for rs in (True, False):
                            for rc in (True, False):
                                lab, v = _adjudicate_leg2(gg, rd, lc, rc, ulc, cc, rs)
                                if v.startswith("H-C-corroborated") and not (gg and rd and lc == "lift"):
                                    print("[self-test] FAIL H-C corroborated without a majority lift", flush=True)
                                    n_fail += 1
                                if v in ("H-C-corroborated-strong", "H-C-corroborated") and ulc == "lift":
                                    print("[self-test] FAIL latent-specific corroboration with an untrained lift", flush=True)
                                    n_fail += 1
                                if v == "H-C-weakened" and not (gg and rd and lc in ("flat", "degrade")
                                                                and cc != "lift"):
                                    print("[self-test] FAIL H-C weakened outside flat/degrade with content not-lift", flush=True)
                                    n_fail += 1
                                if not (gg and rd) and v != "undetermined":
                                    print("[self-test] FAIL leg2 verdict on a red gate", flush=True)
                                    n_fail += 1
                                # PASS only when both legs adjudicated
                                for l1 in [_adjudicate_leg1(True, True, True, False),
                                           _adjudicate_leg1(True, True, False, False),
                                           _adjudicate_leg1(True, True, False, True),
                                           _adjudicate_leg1(False, True, True, True)]:
                                    o, _j, _hv = _combine(l1, (lab, v))
                                    both = (l1[1] in ("H-E-eliminated", "H-E-confirmed")
                                            and (v.startswith("H-C-corroborated") or v == "H-C-weakened"))
                                    if (o == "PASS") != both:
                                        print("[self-test] FAIL outcome/adjudication mismatch %r %r -> %s"
                                              % (l1, (lab, v), o), flush=True)
                                        n_fail += 1
    print("[self-test] %d failure(s)" % n_fail, flush=True)
    return 1 if n_fail else 0


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
    assert_no_structurally_unsatisfiable_gate(PRECONDITION_SPECS, _arm_contexts())
    # Red-team F4: a run with fewer seeds than the pre-registered majority cannot form one, so
    # it cannot emit a hypothesis verdict on either leg. The majority itself is never lowered.
    seeds_sufficient = bool(len(seeds) >= SEED_MAJORITY)
    majority = int(SEED_MAJORITY)

    env_kwargs = x734._env_kwargs_for_rung(RUNG)
    cfg_base = x1002._off_path_config_slice(
        dry_run, sched["zworld_p0"], sched["p0"], sched["p1"], sched["steps"],
        sched["bc_eps"], sched["bc_rand"], sched["passes"], sched["eval_eps"])
    probe_env = x734._make_env(seeds[0], env_kwargs)
    action_dim = int(probe_env.action_dim)

    # ---- the 1002 dataset, re-collected from its deterministic recipe, once per seed --------
    per_seed_data: Dict[int, Dict[str, Any]] = {}
    per_seed_feats: Dict[int, Dict[str, Any]] = {}
    anchor_rows: List[Dict[str, Any]] = []
    for s in seeds:
        torch.manual_seed(s)
        np.random.seed(s)
        oracle_eps = x1002._collect_episodes(s, env_kwargs, "oracle", sched["bc_eps"], sched["steps"])
        rand_eps = x1002._collect_episodes(s, env_kwargs, "random", sched["bc_rand"], sched["steps"])
        tr, te = x1002._split_episodes(oracle_eps)
        data = {"train": tr, "test": te, "random": rand_eps}
        per_seed_data[s] = data
        f_tr, y_tr = x1002._rawfield_features(tr)
        f_te, y_te = x1002._rawfield_features(te)
        fr_te, yr_te = x1002._rawfield_features(rand_eps)
        w_tr, _ = _world_state_features(tr)
        w_te, _ = _world_state_features(te)
        wr_te, _ = _world_state_features(rand_eps)
        per_seed_feats[s] = {"y_tr": y_tr, "y_te": y_te, "yr_te": yr_te,
                             "field": {"tr": f_tr, "te": f_te, "r": fr_te},
                             "ws": {"tr": w_tr, "te": w_te, "r": wr_te}}
        ev = evaluate_seed(LocalViewGreedyPolicy(s), x734._make_env(s, env_kwargs),
                           sched["eval_eps"], sched["steps"])
        # The oracle's own rule on the TRUE field reproduces its labels except on flat windows
        # (random move) -- recorded as the ceiling of the closed-form content witness.
        rule_true = _decode_oracle_agreement(f_te, y_te)
        anchor_rows.append({
            "cell_id": "local_view_greedy|seed%d" % s, "anchor_id": "local_view_greedy",
            "seed": int(s), "foraging_competence": float(ev["foraging_competence"]),
            "competence_supra_floor": bool(ev["competence_supra_floor"]),
            "n_train_episodes": len(tr), "n_test_episodes": len(te),
            "n_random_episodes": len(rand_eps),
            "n_train_steps": int(f_tr.shape[0]), "n_heldout_steps": int(f_te.shape[0]),
            "world_state_dim": int(w_tr.shape[1]) if int(w_tr.shape[0]) else None,
            "oracle_rule_on_true_field_agreement": rule_true["linear_decode_oracle_agreement"],
            "true_field_flat_window_frac": rule_true["decoded_flat_window_frac"],
        })

    # ---- positive control first, on every seed: it gates everything else ----------------
    frozen: Dict[int, Dict[str, Any]] = {s: {} for s in seeds}
    raw_rows = [run_cell(ARM_RAW, s, per_seed_data[s], per_seed_feats[s], frozen[s],
                         action_dim, sched, env_kwargs, cfg_base, dry_run) for s in seeds]

    # ---- run-level DV headroom (bar + elevation), from the control's own values --------
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
            control=("strongest TRIVIAL predictor's held-out agreement per seed -- "
                     "max(state-blind majority class, repeat-previous-executed-action)"),
            description=("The elevation criterion must be reachable: 1.0 minus the worst "
                         "trivial baseline, margin 2.0. Inherited from 1002 where it passed by "
                         "0.02-0.03 (measured headroom 0.418-0.434 vs 0.40); NOT loosened. The "
                         "true DV ceiling is <= 0.015 below 1.0 (flat-window random moves, "
                         "red-team F9); disclosed, not corrected.")))
    if raw_agree_vals:
        dv_checks.append(dv_headroom_check(
            "dv_headroom_agreement_absolute",
            dv_name="oracle_action_agreement",
            criterion_threshold=float(AGREEMENT_BAR),
            control_values=raw_agree_vals, statistic="max_abs", margin=1.0,
            control="rawfield_ceiling held-out agreement per seed",
            description=("The absolute bar must sit inside what the DV demonstrably reaches "
                         "at this capacity on this dataset (1002: 0.973-0.985 vs 0.80).")))
    try:
        dv_preconditions = p0_readiness_gate(dv_checks) if dv_checks else []
        dv_gate_green, dv_gate_reason = True, ""
    except P0NotReady as e:
        dv_preconditions = list(e.preconditions)
        dv_gate_green = False
        dv_gate_reason = "dv_headroom unmet: " + ", ".join(
            str(p.get("name")) for p in dv_preconditions if not p.get("met"))

    raw_worst, raw_worst_cell = x1002._worst_cell(raw_rows, "oracle_action_agreement", "min")
    maj_worst, _mc = x1002._worst_cell(raw_rows, "train_majority_class_share", "max")
    nte_worst, _nc = x1002._worst_cell(raw_rows, "n_heldout_steps", "min")
    lvg_worst, lvg_worst_cell = x1002._worst_cell(anchor_rows, "foraging_competence", "min")
    instrument_ready = bool(raw_worst >= RAW_FIELD_CONTROL_FLOOR) and dv_gate_green

    # ---- the remaining nine arms, per seed (diag cells before their re-based siblings) --
    # `or dry_run`: the smoke MUST execute every arm, including the warmup and every
    # re-basis; at dry scale the control cannot reach its floor and the gate would otherwise
    # short-circuit past the code this run exists to exercise (V3-EXQ-591g).
    other_rows: List[Dict[str, Any]] = []
    rest = [a for a in ARM_IDS if a != ARM_RAW]
    for s in seeds:
        if instrument_ready or dry_run:
            for aid in rest:
                other_rows.append(run_cell(aid, s, per_seed_data[s], per_seed_feats[s],
                                           frozen[s], action_dim, sched, env_kwargs,
                                           cfg_base, dry_run))
        else:
            for aid in rest:
                print("Seed %d Condition %s:%s" % (s, RUNG_ID, aid), flush=True)
                print("  [skip] instrument not certified; arm not run", flush=True)
                print("verdict: FAIL", flush=True)
        frozen[s].clear()   # release the agents
    all_rows = raw_rows + other_rows

    def _rows(aid: str) -> List[Dict[str, Any]]:
        return [r for r in all_rows if r["arm_id"] == aid]

    def _by_seed(aid: str, key: str) -> Dict[int, Any]:
        return {r["seed"]: r.get(key) for r in _rows(aid)}

    def _arm_summary(aid: str) -> Dict[str, Any]:
        rows = _rows(aid)
        agr = [r["oracle_action_agreement"] for r in rows if r.get("oracle_action_agreement") is not None]
        elev = [r["agreement_elevation"] for r in rows if r.get("agreement_elevation") is not None]
        n_clear = int(sum(1 for r in rows
                          if (r.get("oracle_action_agreement") or 0.0) >= AGREEMENT_BAR
                          and (r.get("agreement_elevation") or 0.0) >= AGREEMENT_ELEVATION_MIN))
        return {
            "arm_id": aid, "ran": bool(rows), "n_seeds": len(rows),
            "n_seeds_clearing_bar": n_clear,
            "majority_clears_bar": bool(rows and seeds_sufficient and n_clear >= majority),
            "mean_oracle_action_agreement": (x1002._mean(agr) if agr else None),
            "per_seed_oracle_action_agreement": [r.get("oracle_action_agreement") for r in rows],
            "per_seed_oracle_action_agreement_train": [r.get("oracle_action_agreement_train") for r in rows],
            "per_seed_oracle_action_agreement_random_states": [
                r.get("oracle_action_agreement_random_states") for r in rows],
            "per_seed_oracle_action_agreement_turn_states": [
                r.get("oracle_action_agreement_turn_states") for r in rows],
            "per_seed_oracle_action_agreement_unstandardised": [
                r.get("oracle_action_agreement_unstandardised") for r in rows],
            "mean_agreement_elevation": (x1002._mean(elev) if elev else None),
            "per_seed_agreement_elevation": [r.get("agreement_elevation") for r in rows],
            "per_seed_trivial_baseline": [r.get("trivial_baseline") for r in rows],
            "mean_cloned_foraging_competence": x1002._mean(
                [r["cloned_foraging_competence"] for r in rows
                 if r.get("cloned_foraging_competence") is not None]),
            "per_seed_cloned_foraging_competence": [r.get("cloned_foraging_competence") for r in rows],
            "per_seed_cloned_death_rate": [r.get("cloned_death_rate") for r in rows],
            "per_seed_participation_ratio": [r.get("zworld_participation_ratio") for r in rows],
            "per_seed_final_ce_loss": [(r.get("adapter_training") or {}).get("final_ce_loss") for r in rows],
            "per_seed_rebasis_condition_number": [r.get("rebasis_condition_number") for r in rows],
            "per_seed_field_decode_r2_abs_delta": [r.get("field_decode_r2_abs_delta") for r in rows],
            "per_seed_linear_decode_oracle_agreement": [r.get("linear_decode_oracle_agreement") for r in rows],
            "per_seed_decision_subspace_retention_max": [r.get("decision_subspace_retention_max") for r in rows],
            "per_seed_oracle_action_agreement_randproj_draw_b": [
                r.get("oracle_action_agreement_randproj_draw_b") for r in rows],
            "action_path_params": (rows[0]["capacity_match"]["action_path_params"] if rows else None),
            "feature_dim": (rows[0].get("feature_dim") if rows else None),
        }

    per_arm = {aid: _arm_summary(aid) for aid in ARM_IDS}

    # ---- per-arm precondition gates (never AND'd across arms) ----------------------------
    full_worst, _fc = x1002._worst_cell(_rows(ARM_WS_FULL), "oracle_action_agreement", "min") \
        if _rows(ARM_WS_FULL) else (0.0, None)
    ve_rows = [{"cell_id": r["cell_id"],
                "ve": float(((r.get("world_state_pca_stats") or {}).get("variance_explained_at_k")
                             or 1.0))} for r in _rows(ARM_WS_PCA)]
    ve_worst, _vc = x1002._worst_cell(ve_rows, "ve", "max") if ve_rows else (1.0, None)
    arm_gates = []
    for aid in ARM_IDS:
        rows = _rows(aid)
        ctx = _arm_ctx(aid)
        measured = {
            "adapter_capacity_sufficient_on_raw_field": raw_worst,
            "oracle_labels_non_degenerate": maj_worst,
            "heldout_split_sufficient": nte_worst,
            "d3_local_view_greedy_clears_floor": lvg_worst,
        }
        if ctx["leg1"]:
            measured["ws250_full_input_reaches_bar"] = full_worst
        if ctx["projected"]:
            measured["ws250_compression_lossy"] = ve_worst
            ret_max, _rc = x1002._worst_cell(rows, "decision_subspace_retention_max", "max") \
                if rows else (1.0, None)
            measured["ws250_projection_mixes_decision_subspace"] = ret_max
        if ctx["has_encoder"]:
            prmin, _pc = x1002._worst_cell(rows, "zworld_participation_ratio", "min") if rows else (0.0, None)
            measured["zworld_not_collapsed"] = prmin
        if ctx["trained_encoder"]:
            dmin, _dc = x1002._worst_cell(
                [{"cell_id": r["cell_id"],
                  "d": float((r.get("zworld_weight_delta") or {}).get("world_encoder_max_abs_delta", 0.0) or 0.0)}
                 for r in rows], "d", "min") if rows else (0.0, None)
            measured["zworld_encoder_trained_in_p0"] = dmin
        arm_gates.append(evaluate_arm_gate(aid, ctx, PRECONDITION_SPECS, measured))
    gate = aggregate_arm_gates(arm_gates)
    green = set(gate.get("green_arms") or [])
    gate_green = bool(gate["non_degenerate"]) and dv_gate_green and seeds_sufficient

    # ---- LEG 1: H-E ------------------------------------------------------------------------
    pca_clears = bool(per_arm[ARM_WS_PCA]["majority_clears_bar"])
    rand_clears = bool(per_arm[ARM_WS_RAND]["majority_clears_bar"])
    full_clears = bool(per_arm[ARM_WS_FULL]["majority_clears_bar"])
    leg1_ready = bool(LEG1_VERDICT_ARM in green and LEG1_ANCHOR_ARM in green
                      and LEG1_CONTROL_ARM in green)
    pca_by_seed = _by_seed(ARM_WS_PCA, "oracle_action_agreement")
    rand_by_seed = _by_seed(ARM_WS_RAND, "oracle_action_agreement")
    unt_by_seed = _by_seed(ARM_UNT_DIAG, "oracle_action_agreement")
    off_by_seed = _by_seed(ARM_OFF_DIAG, "oracle_action_agreement")
    leg1_per_seed = []
    for r in _rows(ARM_WS_PCA):
        s = r["seed"]
        a_pca, a_rand = pca_by_seed.get(s), rand_by_seed.get(s)
        triv = r.get("trivial_baseline")
        eff = [float(AGREEMENT_BAR)] + ([float(triv) + float(AGREEMENT_ELEVATION_MIN)] if triv is not None else [])
        leg1_per_seed.append({
            "seed": int(s), "ws250_pca_agreement": a_pca, "ws250_randproj_agreement": a_rand,
            "ws250_randproj_agreement_draw_b": _by_seed(ARM_WS_RAND, "oracle_action_agreement_randproj_draw_b").get(s),
            "ws250_full_agreement": _by_seed(ARM_WS_FULL, "oracle_action_agreement").get(s),
            "trivial_baseline": triv, "effective_pass_threshold": max(eff),
            "pca_minus_randproj": ((a_pca - a_rand) if (a_pca is not None and a_rand is not None) else None),
            "pca_beats_randproj_by_margin": bool(a_pca is not None and a_rand is not None
                                                 and (a_pca - a_rand) >= UNTRAINED_CONTROL_MARGIN),
            # the H-E-relevant channel comparison: the REE channel's own outputs vs a random
            # LINEAR compression of the identical input, paired per seed
            "randproj_minus_zworld_untrained": ((a_rand - unt_by_seed[s])
                                                if (a_rand is not None and unt_by_seed.get(s) is not None) else None),
            "randproj_minus_zworld_off": ((a_rand - off_by_seed[s])
                                          if (a_rand is not None and off_by_seed.get(s) is not None) else None),
            "pca_decision_subspace_retention_max": r.get("decision_subspace_retention_max"),
        })
    n_pca_beats_rand = int(sum(1 for m in leg1_per_seed if m["pca_beats_randproj_by_margin"]))
    pca_beats_rand = bool(leg1_per_seed and seeds_sufficient and n_pca_beats_rand >= majority)
    if not seeds_sufficient:
        leg1_label, leg1_verdict = ("insufficient_seeds_for_majority", "undetermined")
    else:
        leg1_label, leg1_verdict = _adjudicate_leg1(gate_green, leg1_ready, pca_clears, rand_clears)

    # ---- LEG 2: the H-C corroborator ------------------------------------------------------
    fd_by_seed = _by_seed(ARM_OFF_FD, "oracle_action_agreement")
    lda_by_seed = _by_seed(ARM_OFF_LDA, "oracle_action_agreement")
    zca_by_seed = _by_seed(ARM_OFF_ZCA, "oracle_action_agreement")
    untfd_by_seed = _by_seed(ARM_UNT_FD, "oracle_action_agreement")
    content_by_seed = _by_seed(ARM_OFF_FD, "linear_decode_oracle_agreement")
    content_unt_by_seed = _by_seed(ARM_UNT_FD, "linear_decode_oracle_agreement")
    leg2_per_seed = []
    for r in _rows(ARM_OFF_DIAG):
        s = r["seed"]
        a_diag, a_fd = off_by_seed.get(s), fd_by_seed.get(s)
        a_lda, a_zca = lda_by_seed.get(s), zca_by_seed.get(s)
        a_unt, a_untfd = unt_by_seed.get(s), untfd_by_seed.get(s)
        c_off, c_unt = content_by_seed.get(s), content_unt_by_seed.get(s)

        def _d(a: Optional[float], b: Optional[float]) -> Optional[float]:
            return (a - b) if (a is not None and b is not None) else None

        d_fd, d_lda, d_zca, d_unt = _d(a_fd, a_diag), _d(a_lda, a_diag), _d(a_zca, a_diag), _d(a_untfd, a_unt)
        d_content, short = _d(c_off, a_diag), _d(c_off, a_fd)
        leg2_per_seed.append({
            "seed": int(s), "zworld_off_diag_agreement": a_diag,
            "zworld_off_fielddecode_agreement": a_fd,
            "lift_fielddecode": d_fd, "lift_class_fielddecode": _lift_class(d_fd),
            "zworld_off_ldawhiten_agreement": a_lda, "lift_ldawhiten": d_lda,
            "lift_class_ldawhiten": _lift_class(d_lda),
            "zworld_off_zca_agreement": a_zca, "lift_zca": d_zca, "lift_class_zca": _lift_class(d_zca),
            "zworld_untrained_diag_agreement": a_unt,
            "zworld_untrained_fielddecode_agreement": a_untfd,
            "lift_fielddecode_untrained": d_unt, "lift_class_fielddecode_untrained": _lift_class(d_unt),
            "linear_decode_oracle_agreement_off": c_off,
            "linear_decode_oracle_agreement_untrained": c_unt,
            "content_margin_over_diag": d_content, "content_class": _lift_class(d_content),
            "reader_shortfall_on_decoded_basis": short,
            "reader_short": bool(short is not None and short >= READER_SHORTFALL_MIN),
            "trivial_baseline": r.get("trivial_baseline"),
        })
    lift_class = _majority_class([m["lift_class_fielddecode"] for m in leg2_per_seed], majority) \
        if leg2_per_seed else "inconsistent"
    lift_class_lda = _majority_class([m["lift_class_ldawhiten"] for m in leg2_per_seed], majority) \
        if leg2_per_seed else "inconsistent"
    lift_class_zca = _majority_class([m["lift_class_zca"] for m in leg2_per_seed], majority) \
        if leg2_per_seed else "inconsistent"
    lift_class_untrained = _majority_class([m["lift_class_fielddecode_untrained"] for m in leg2_per_seed], majority) \
        if leg2_per_seed else "inconsistent"
    content_class = _majority_class([m["content_class"] for m in leg2_per_seed], majority) \
        if leg2_per_seed else "inconsistent"
    n_reader_short = int(sum(1 for m in leg2_per_seed if m["reader_short"]))
    reader_short = bool(leg2_per_seed and seeds_sufficient and n_reader_short >= majority)
    rebased_clears = bool(per_arm[ARM_OFF_FD]["majority_clears_bar"])
    # Lift headroom: the criterion must have room above the BASELINE arm (bounded DV).
    diag_vals = [float(v) for v in off_by_seed.values() if v is not None]
    lift_dv_preconditions: List[Dict[str, Any]] = []
    lift_gate_green = True
    if diag_vals:
        try:
            lift_dv_preconditions = p0_readiness_gate([dv_headroom_check(
                "dv_headroom_rebasis_lift", dv_name="rebasis_lift_over_diag_baseline",
                criterion_threshold=float(REBASIS_LIFT_MIN), control_values=diag_vals,
                statistic="ceiling_headroom", dv_bounds=(0.0, 1.0), margin=2.0,
                control="zworld_off_diag held-out agreement per seed (the leg-2 baseline, in-run)",
                description=("The lift criterion must be reachable: 1.0 minus the worst "
                             "(largest) IN-RUN baseline agreement, margin 2.0. 1002 measured the "
                             "baseline at 0.648-0.674, i.e. ~0.33 of headroom against 0.10; the "
                             "true DV ceiling is <= 0.015 below 1.0 (red-team F9), disclosed."))])
        except P0NotReady as e:
            lift_dv_preconditions = list(e.preconditions)
            lift_gate_green = False
    leg2_ready = bool(LEG2_VERDICT_ARM in green and LEG2_BASELINE_ARM in green
                      and ARM_UNT_FD in green and ARM_UNT_DIAG in green and lift_gate_green)
    if not seeds_sufficient:
        leg2_label, leg2_verdict = ("insufficient_seeds_for_majority", "undetermined")
    else:
        leg2_label, leg2_verdict = _adjudicate_leg2(
            gate_green, leg2_ready, lift_class, rebased_clears,
            lift_class_untrained, content_class, reader_short)

    outcome, label, hypothesis_verdict = _combine((leg1_label, leg1_verdict),
                                                  (leg2_label, leg2_verdict))
    leg1_adjudicated = leg1_verdict in ("H-E-eliminated", "H-E-confirmed")
    leg2_adjudicated = leg2_verdict.startswith("H-C-corroborated") or leg2_verdict == "H-C-weakened"
    raw_clears = bool(raw_worst >= RAW_FIELD_CONTROL_FLOOR)

    criteria = [
        {"name": "C_leg1_adjudicated", "load_bearing": True, "passed": bool(leg1_adjudicated),
         "description": ("LEG 1 reached an H-E verdict: the instrument gate, the 250-dim "
                         "readiness anchor (ws250_full >= %.2f), the two compression "
                         "preconditions and the seed count are green, and ws250_pca either "
                         "clears the bar (H-E eliminated) or neither task-agnostic compression "
                         "does (H-E confirmed)." % WS250_FULL_INPUT_FLOOR)},
        {"name": "C_leg2_adjudicated", "load_bearing": True, "passed": bool(leg2_adjudicated),
         "description": ("LEG 2 reached an H-C reading: encoder trained, latents not collapsed, "
                         "lift headroom present, seed count sufficient, and the seed-majority "
                         "lift class of the decode re-basis is LIFT (corroborated; channel-level "
                         "if the untrained latent lifts too) or FLAT/DEGRADE with the closed-form "
                         "content witness not lifting (weakened: information, not geometry).")},
        {"name": "C_ws250_pca_clears_bar", "load_bearing": False, "passed": pca_clears,
         "description": ("%s reaches >= %.2f held-out agreement AND >= %.2f elevation over "
                         "that seed's strongest trivial predictor on >= %d of %d seeds. TRUE -> "
                         "H-E eliminated; FALSE (with the anchor green and the random control "
                         "also failing) -> H-E confirmed."
                         % (ARM_WS_PCA, AGREEMENT_BAR, AGREEMENT_ELEVATION_MIN, majority, len(seeds)))},
        {"name": "C_ws250_randproj_clears_bar", "load_bearing": False, "passed": rand_clears,
         "description": ("The random orthonormal compression clears bar + elevation. Reported; "
                         "TRUE alongside pca_clears is the STRONG H-E elimination.")},
        {"name": "C_ws250_pca_beats_randproj", "load_bearing": False, "passed": pca_beats_rand,
         "description": ("%s exceeds paired %s by >= %.2f on >= %d seeds (%d of %d cleared). "
                         "ATTRIBUTION readout for the compression CHOICE; deliberately NOT a "
                         "conjunct of the H-E verdict (see the docstring)."
                         % (ARM_WS_PCA, ARM_WS_RAND, UNTRAINED_CONTROL_MARGIN, majority,
                            n_pca_beats_rand, len(leg1_per_seed)))},
        {"name": "C_rebasis_lifts_by_margin", "load_bearing": False, "passed": bool(lift_class == "lift"),
         "description": ("Seed-majority lift class of %s minus paired %s is LIFT (>= %.2f). "
                         "Classes: lift / marginal / flat / degrade / inconsistent -- this run's: "
                         "%s (untrained pair: %s; ZCA: %s; LDA: %s; content witness: %s)."
                         % (ARM_OFF_FD, ARM_OFF_DIAG, REBASIS_LIFT_MIN, lift_class,
                            lift_class_untrained, lift_class_zca, lift_class_lda, content_class))},
        {"name": "C_rebasis_clears_bar", "load_bearing": False, "passed": rebased_clears,
         "description": "%s clears bar + elevation on >= %d seeds." % (ARM_OFF_FD, majority)},
        {"name": "C_linear_content_supports_mapping", "load_bearing": False,
         "passed": bool(content_class == "lift"),
         "description": ("CLOSED-FORM content witness: local_view_greedy's rule applied to the "
                         "linearly decoded field (no adapter) exceeds the diag baseline by >= "
                         "%.2f on the seed majority. TRUE says the linearly decodable content "
                         "supports more of the mapping than the adapter found on the raw basis; "
                         "FALSE says the content itself is the ceiling." % REBASIS_LIFT_MIN)},
        {"name": "C_positive_control_learns_from_raw_field", "load_bearing": False, "passed": raw_clears,
         "description": ("The same adapter reaches >= %.2f from the raw 25-dim field on every "
                         "seed (a gate on interpretability, as in 1002)." % RAW_FIELD_CONTROL_FLOOR)},
        {"name": "C_ws250_full_input_reaches_bar", "load_bearing": False, "passed": full_clears,
         "description": ("The UNCOMPRESSED 250-dim input clears the bar on >= %d seeds; leg 1's "
                         "readiness anchor (a gate, never a verdict)." % majority)},
    ]
    combination_rule = (
        "TWO INDEPENDENT LEGS, each a pure function of seed-majority classes under its own "
        "readiness gates (`_adjudicate_leg1`, `_adjudicate_leg2`, contract-tested by --self-test). "
        "LEG 1 (H-E): verdict on %s alone -- clears bar+elevation -> H-E eliminated; neither "
        "task-agnostic compression clears -> H-E confirmed; random-clears-but-PCA-does-not -> "
        "undetermined. The PCA-minus-random margin is reported, not a conjunct. LEG 2 (H-C "
        "corroborator): verdict on the paired lift of %s over %s -- seed-majority LIFT (>= %.2f) "
        "-> corroborated (strong if the re-based arm also clears the bar; CHANNEL-LEVEL, never "
        "latent-specific, if the untrained latent lifts by the same class); FLAT or DEGRADE -> "
        "weakened ONLY when the closed-form decode-then-oracle content witness is itself not a "
        "lift (information ceiling), otherwise undetermined (reader shortfall / inconsistent); "
        "MARGINAL or inconsistent -> undetermined. Fewer than %d seeds -> no verdict on either "
        "leg. A red gate on either leg licenses NO verdict for that leg and never for the other. "
        "OUTCOME = PASS iff BOTH legs adjudicated (C_leg1_adjudicated AND C_leg2_adjudicated); "
        "the science is in the labels and hypothesis_verdict, not in PASS/FAIL."
        % (ARM_WS_PCA, ARM_OFF_FD, ARM_OFF_DIAG, REBASIS_LIFT_MIN, SEED_MAJORITY))
    overall_pass = bool(outcome == "PASS")

    non_degenerate_flags = arm_criteria_non_degenerate(
        {ARM_WS_PCA: ["C_leg1_adjudicated", "C_ws250_pca_clears_bar", "C_ws250_pca_beats_randproj"],
         ARM_WS_RAND: ["C_ws250_randproj_clears_bar"],
         ARM_WS_FULL: ["C_ws250_full_input_reaches_bar"],
         ARM_OFF_FD: ["C_leg2_adjudicated", "C_rebasis_lifts_by_margin", "C_rebasis_clears_bar",
                      "C_linear_content_supports_mapping"],
         ARM_RAW: ["C_positive_control_learns_from_raw_field"]},
        gate,
        extra={
            # a paired differential is degenerate when its comparator is red or absent
            "C_ws250_pca_beats_randproj": bool(ARM_WS_RAND in green
                                               and any(m["ws250_randproj_agreement"] is not None
                                                       for m in leg1_per_seed)),
            "C_leg1_adjudicated": bool(ARM_WS_FULL in green and ARM_WS_RAND in green and seeds_sufficient),
            "C_rebasis_lifts_by_margin": bool(ARM_OFF_DIAG in green and lift_gate_green
                                              and any(m["lift_fielddecode"] is not None for m in leg2_per_seed)),
            "C_linear_content_supports_mapping": bool(ARM_OFF_DIAG in green
                                                      and any(m["content_margin_over_diag"] is not None
                                                              for m in leg2_per_seed)),
            "C_leg2_adjudicated": bool(ARM_OFF_DIAG in green and ARM_UNT_FD in green
                                       and lift_gate_green and seeds_sufficient),
        })

    metrics = {
        "agreement_bar": float(AGREEMENT_BAR),
        "agreement_elevation_min": float(AGREEMENT_ELEVATION_MIN),
        "untrained_control_margin": float(UNTRAINED_CONTROL_MARGIN),
        "rebasis_lift_min": float(REBASIS_LIFT_MIN),
        "rebasis_flat_eps": float(REBASIS_FLAT_EPS),
        "reader_shortfall_min": float(READER_SHORTFALL_MIN),
        "raw_field_control_floor": float(RAW_FIELD_CONTROL_FLOOR),
        "ws250_full_input_floor": float(WS250_FULL_INPUT_FLOOR),
        "ws250_variance_explained_ceiling": float(WS250_VARIANCE_EXPLAINED_CEILING),
        "decision_subspace_retention_ceiling": float(DECISION_SUBSPACE_RETENTION_CEILING),
        "projection_dim": int(PROJECTION_DIM),
        "seed_majority": int(SEED_MAJORITY), "n_seeds": len(seeds),
        "seeds_sufficient": bool(seeds_sufficient),
        "leg1_verdict_arm": LEG1_VERDICT_ARM, "leg2_verdict_arm": LEG2_VERDICT_ARM,
        "rawfield_worst_seed_agreement": raw_worst, "rawfield_worst_cell": raw_worst_cell,
        "ws250_full_worst_seed_agreement": full_worst,
        "ws250_pca_variance_explained_worst_seed": ve_worst,
        "leg2_lift_class_fielddecode": lift_class,
        "leg2_lift_class_fielddecode_untrained": lift_class_untrained,
        "leg2_lift_class_ldawhiten": lift_class_lda,
        "leg2_lift_class_zca": lift_class_zca,
        "leg2_content_class": content_class,
        "leg2_n_seeds_reader_short": n_reader_short,
        "n_seeds_pca_beats_randproj": n_pca_beats_rand,
        "worst_seed_trivial_baseline": (max(state_blind_vals) if state_blind_vals else None),
        "worst_seed_oracle_majority_share": maj_worst,
        "local_view_greedy_worst_seed_competence": lvg_worst,
    }
    for aid in ARM_IDS:
        metrics["%s_mean_agreement" % aid] = per_arm[aid]["mean_oracle_action_agreement"]
        metrics["%s_n_seeds_clearing_bar" % aid] = per_arm[aid]["n_seeds_clearing_bar"]
        metrics["%s_mean_cloned_competence" % aid] = per_arm[aid]["mean_cloned_foraging_competence"]

    interpretation = {
        "label": label,
        "question_id": HYPOTHESIS_QID,
        "hypothesis_verdict": hypothesis_verdict,
        "leg_verdicts": {"H-E-channel-input-capacity": leg1_verdict,
                         "H-C-geometry-mismatch": leg2_verdict},
        "seeds_sufficient": bool(seeds_sufficient),
        "leg1": {"label": leg1_label, "verdict": leg1_verdict, "ready": leg1_ready,
                 "verdict_arm": LEG1_VERDICT_ARM, "control_arm": LEG1_CONTROL_ARM,
                 "anchor_arm": LEG1_ANCHOR_ARM, "pca_clears": pca_clears,
                 "randproj_clears": rand_clears, "full_clears": full_clears,
                 "pca_beats_randproj": pca_beats_rand, "per_seed": leg1_per_seed},
        "leg2": {"label": leg2_label, "verdict": leg2_verdict, "ready": leg2_ready,
                 "verdict_arm": LEG2_VERDICT_ARM, "baseline_arm": LEG2_BASELINE_ARM,
                 "lift_class": lift_class, "lift_class_untrained": lift_class_untrained,
                 "lift_class_zca": lift_class_zca, "lift_class_ldawhiten": lift_class_lda,
                 "content_class": content_class, "reader_short": reader_short,
                 "rebased_clears": rebased_clears,
                 "lift_specific_to_trained_latent": bool(lift_class == "lift"
                                                         and lift_class_untrained != "lift"),
                 "per_seed": leg2_per_seed},
        "rebasis_witnesses": {
            "note": ("Recorded, NOT gates (red-team F5): the held-out field-decode r2 is "
                     "invariant under any invertible affine re-basis by algebra, so its delta "
                     "witnesses lstsq numerics only; the condition number is that of the "
                     "transform T (sqrt of the scatter's for the whitenings)."),
            "per_arm": {aid: {"per_seed_condition_number": per_arm[aid]["per_seed_rebasis_condition_number"],
                              "per_seed_field_decode_r2_abs_delta": per_arm[aid]["per_seed_field_decode_r2_abs_delta"]}
                        for aid in REBASED_ARMS},
        },
        "question": (
            "Two parallel legs on V3-EXQ-1002's dataset, adapter and standardiser. LEG 1: does "
            "a TASK-AGNOSTIC 32-dim LINEAR compression of the encoder's actual 250-dim input "
            "support the oracle mapping at the consumer's capacity (H-E-channel-input-capacity)? "
            "LEG 2: does an information-preserving, decision-relevant linear re-basis of the "
            "frozen 978-OFF z_world change its accessibility to the same reader, and how much "
            "of the mapping does the linearly decodable content support in closed form (the "
            "owed corroborator of the confirmed H-C-geometry-mismatch leg)?"),
        "hypotheses": {
            "H-E-channel-input-capacity": (
                "A task-agnostic 32-dim compression of the full 250-dim world_state cannot "
                "support the mapping under this reader; the deficit is a channel-INPUT bound "
                "and no rotation of z_world can fix it. ELIMINATED if ws250_pca clears the bar; "
                "CONFIRMED if neither compression does while the uncompressed input can. A "
                "task-INFORMED coordinate selection (the raw field) trivially supports it and "
                "is not what the hypothesis is about."),
            "H-C-geometry-mismatch": (
                "The information is present and linearly decodable but the geometry does not "
                "make the mapping accessible under the actual learning dynamics. CORROBORATED "
                "by a >= %.2f paired lift under the decode re-basis (channel-level if the "
                "untrained latent lifts alike); WEAKENED by a flat/degrading result whose "
                "closed-form content witness is itself flat -- the block is then information, "
                "not geometry." % REBASIS_LIFT_MIN),
        },
        "combination_rule": combination_rule,
        "preconditions": (list(gate["adjudication_preconditions"]) + list(dv_preconditions)
                          + list(lift_dv_preconditions)),
        "per_arm_gate": gate,
        "dv_headroom_gate_green": bool(dv_gate_green and lift_gate_green),
        "dv_headroom_reason": dv_gate_reason,
        "criteria_non_degenerate": non_degenerate_flags,
        "criteria": criteria,
        "null_reading": {
            "insufficient_seeds_for_majority": ("fewer seeds than the pre-registered majority; "
                                                "readouts only, no verdict (red-team F4)."),
            "leg1_not_ready": ("the instrument gate, the 250-dim readiness anchor or a "
                               "compression precondition is red; no H-E verdict."),
            "ws250_any_linear_compression_supports_mapping": (
                "H-E ELIMINATED (strong): even a random 32-dim linear compression of the "
                "input supports the mapping; the REE channel's own outputs (untrained and "
                "trained, see leg1.per_seed randproj_minus_zworld_*) sit below it. The "
                "deficit is what the encoder does to its input, not the dimensionality."),
            "ws250_optimal_linear_compression_supports_mapping": (
                "H-E ELIMINATED: the variance-optimal task-agnostic linear compression supports "
                "the mapping; the REE channel does not. H-C's learned-geometry gloss survives."),
            "ws250_random_clears_pca_does_not": (
                "undetermined: a working compression exists (so H-E is not confirmed) but "
                "the variance-optimal one discards task-relevant low-variance directions."),
            "ws250_linear_compression_fails_bar": (
                "H-E CONFIRMED (as registered): neither task-agnostic 32-dim linear compression "
                "-- variance-optimal or random -- of the actual input supports the mapping "
                "although the uncompressed input does. Not a universal: the raw field itself, a "
                "task-informed 250 -> 25 selection, supports it at 0.97+ in this same run. The "
                "confirmed H-C leg is weakened to its narrow gate sense; route is interface "
                "redesign."),
            "leg2_not_ready": ("encoder untrained / latent collapsed / no lift headroom / a "
                               "leg-2 arm red; no H-C reading."),
            "rebasis_lifts_trained_and_untrained_alike": (
                "H-C CORROBORATED (channel-level): the decode re-basis makes the mapping "
                "accessible on BOTH the trained and the untrained latent, so the geometry block "
                "belongs to the encoder channel's output, not to this latent's learned geometry "
                "(H-D; the registry caveat stands). Not 'strong' whatever the bar says."),
            "rebasis_restores_oracle_mapping": (
                "H-C CORROBORATED (strong): a linear re-basis alone makes the frozen latent "
                "actionable at the bar, and the untrained latent does not lift alike -- the "
                "interface fix is a re-basis, not a side channel."),
            "rebasis_lifts_accessibility_below_bar": (
                "H-C CORROBORATED: accessibility improves by the margin under an "
                "information-preserving re-basis specific to the trained latent."),
            "rebasis_marginal_lift": "undetermined: 0.05 <= lift < 0.10 on the seed majority.",
            "linear_content_ceiling_information_not_geometry": (
                "H-C WEAKENED: the re-basis is flat or hurts AND the closed-form reader on the "
                "linearly decoded field reaches only ~baseline, so the linearly decodable "
                "content itself supports no more than the adapter found -- the block is "
                "INFORMATION (the encoder discards the decision-relevant part of its input; "
                "V3-EXQ-948), not geometry, and no re-basis can lift it."),
            "decoded_content_exceeds_reader_instrument_shortfall": (
                "undetermined (instrument): the closed-form reader finds the mapping in the "
                "decoded field and the trained adapter does not even on that basis. A finding "
                "about the reader, never an H-C verdict either way."),
            "rebasis_flat_content_lift_inconsistent": (
                "undetermined: band-edge combination; read the continuous per-seed quantities."),
            "rebasis_effect_inconsistent_across_seeds": "undetermined: no lift class holds a seed majority.",
        },
        "attribution_note": (
            "Leg 1's PCA-minus-random margin (UNTRAINED_CONTROL_MARGIN 0.10) is reported per "
            "seed and as C_ws250_pca_beats_randproj, not thresholded into the H-E verdict: a "
            "random compression also clearing the bar STRENGTHENS the elimination rather than "
            "defeating attribution (the role the margin played in 1002). Authoring-time seed-42 "
            "measurement: pca 0.884, randproj 0.786/0.788, margin 0.098. The strong-vs-plain "
            "H-E label therefore turns on whether the random control lands above 0.80, a "
            "boundary within ~one seed-SD of its authoring-time value (red-team F6); the "
            "verdict is the same on both sides."),
    }

    _rows_md = []
    for aid in ARM_IDS:
        pa = per_arm[aid]
        _a = pa["mean_oracle_action_agreement"]
        _rows_md.append("| %s | %s | %s | %s | %d/%d | %.3f |" % (
            aid, pa["feature_dim"], pa["action_path_params"],
            ("%.4f" % _a) if _a is not None else "not run",
            pa["n_seeds_clearing_bar"], pa["n_seeds"], pa["mean_cloned_foraging_competence"]))

    def _f(v: Optional[float], fmt: str = "%.4f") -> str:
        return (fmt % v) if v is not None else "-"

    _lift_md = "\n".join("| %d | %s | %s | %s | %s | %s | %s | %s |" % (
        m["seed"], _f(m["zworld_off_diag_agreement"]), _f(m["zworld_off_fielddecode_agreement"]),
        _f(m["lift_fielddecode"], "%+.4f"), m["lift_class_fielddecode"],
        _f(m["lift_fielddecode_untrained"], "%+.4f"), _f(m["linear_decode_oracle_agreement_off"]),
        m["content_class"]) for m in leg2_per_seed)
    _leg1_md = "\n".join("| %d | %s | %s | %s | %s | %s |" % (
        m["seed"], _f(m["ws250_full_agreement"]), _f(m["ws250_pca_agreement"]),
        _f(m["ws250_randproj_agreement"]), _f(m["ws250_randproj_agreement_draw_b"]),
        _f(m["pca_minus_randproj"], "%+.4f")) for m in leg1_per_seed)
    summary_markdown = """# %s -- z_world actor-adequacy portfolio (H-E; H-C corroborator)

Outcome: **%s** -- label `%s` -- **%s**

| arm | feature dim | action-path params | mean held-out agreement | seeds clearing bar | mean cloned res/ep |
|---|---|---|---|---|---|
%s

LEG 1 (H-E): `%s`. Verdict arm ws250_pca; anchor ws250_full (must reach %.2f); control ws250_randproj (second draw reported).

| seed | ws250_full | ws250_pca | ws250_randproj | randproj draw b | pca - randproj |
|---|---|---|---|---|---|
%s

LEG 2 (H-C corroborator): `%s`. Seed-majority lift class of zworld_off_fielddecode over zworld_off_diag: **%s** (untrained pair: %s; content witness: %s; zca: %s; ldawhiten: %s).

| seed | off_diag | off_fielddecode | lift | class | lift on untrained | decode-then-oracle | content class |
|---|---|---|---|---|---|---|---|
%s

Bar: agreement >= %.2f AND elevation >= %.2f over the strongest trivial predictor (worst seed %s), on >= %d of %d seeds. Lift classes: lift >= %.2f, marginal >= %.2f, flat within +-%.2f, degrade <= -%.2f. Demonstrator anchor local_view_greedy worst seed %.2f res/ep (cell %s).

%s
""" % (QUEUE_ID, outcome, label, hypothesis_verdict, "\n".join(_rows_md),
       leg1_label, WS250_FULL_INPUT_FLOOR, _leg1_md,
       leg2_label, lift_class, lift_class_untrained, content_class, lift_class_zca, lift_class_lda,
       _lift_md,
       AGREEMENT_BAR, AGREEMENT_ELEVATION_MIN,
       ("%.4f" % max(state_blind_vals)) if state_blind_vals else "n/a", majority, len(seeds),
       REBASIS_LIFT_MIN, REBASIS_FLAT_EPS, REBASIS_FLAT_EPS, REBASIS_FLAT_EPS,
       lvg_worst, lvg_worst_cell, combination_rule)

    return {
        "status": outcome, "outcome": outcome, "overall_pass": overall_pass,
        "metrics": metrics, "interpretation": interpretation,
        "summary_markdown": summary_markdown,
        "claim_ids": CLAIM_IDS, "evidence_direction": "unknown",
        "experiment_purpose": EXPERIMENT_PURPOSE, "experiment_type": EXPERIMENT_TYPE,
        "sleep_driver_pattern": "none",
        "arm_results": all_rows, "per_arm": per_arm, "anchor_results": anchor_rows,
        "rung_id": RUNG_ID, "level_id": LEVEL_ID,
        "world_state_input_stats": {
            "per_seed": [r.get("world_state_pca_stats") for r in _rows(ARM_WS_PCA)],
            "layout": ("use_proxy_fields=True: local_view 5x5x7 one-hot [0:175] + "
                       "contamination_view [175:200] + hazard_field_view [200:225] + "
                       "resource_field_view [225:250]"),
            "oracle_decision_world_state_indices": DECISION_WORLD_STATE_INDICES,
            "per_seed_decision_subspace_retention": {
                aid: [(r.get("feature_transform") or {}).get("decision_subspace_retention")
                      for r in _rows(aid)] for aid in PROJECTED_ARMS},
        },
        "capacity_match": {
            "requirement": ("governance-20260903 red-team amendment 5, inherited from 1002: the "
                            "adapter IS x734.PPOPolicyNet at x734.PPO_TRUNK_HIDDEN."),
            "trunk_hidden": int(x734.PPO_TRUNK_HIDDEN),
            "per_arm_action_path_params": {aid: per_arm[aid]["action_path_params"] for aid in ARM_IDS},
            "note": ("32-dim arms match the consumer exactly; rawfield_ceiling (25) is smaller "
                     "(conservative); ws250_full (250) is larger and is a READINESS ANCHOR only, "
                     "never a verdict arm. Adapter init re-seeded from the cell seed before "
                     "every fit."),
        },
        "frozen_latent_source": {
            "run_id": "v3_exq_978_sd018_directional_field_fishtank_20260903T111718Z_v3",
            "queue_id": "V3-EXQ-978", "via": "V3-EXQ-1002",
            "reproduced_not_loaded": ("978 and 1002 saved no checkpoint; the OFF latent is "
                                      "reproduced by re-running 978's warmup with every constant "
                                      "imported from x734/x808/x724, exactly as 1002 did -- "
                                      "bit-exact through the warmup (machine_class identical in "
                                      "both manifests: linux-x86_64-py3.10-torch2.12.0+cpu); "
                                      "only the adapter init draw differs; every leg-2 threshold "
                                      "is applied to the IN-RUN baseline."),
            "dataset_reproduced_not_loaded": ("1002 persisted no observations; its deterministic "
                                              "recipe (_collect_episodes) is re-run per seed."),
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
                        help="Push the pre-registered synthetic rows through both verdict grids "
                             "and the cube invariants, then exit. Seconds, no env.")
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
        "rung": RUNG, "level_id": LEVEL_ID,
        "env_kwargs": x734._env_kwargs_for_rung(RUNG),
        "zworld_p0_episodes": (DRY_RUN_ZWORLD_P0 if args.dry_run else ZWORLD_P0_EPISODES),
        "p0_warmup_episodes": (DRY_RUN_P0 if args.dry_run else P0_WARMUP_EPISODES),
        "p1_reinforce_episodes": (DRY_RUN_P1 if args.dry_run else P1_REINFORCE_EPISODES),
        "eval_episodes": (DRY_RUN_EVAL if args.dry_run else EVAL_EPISODES),
        "steps_per_episode": (DRY_RUN_STEPS if args.dry_run else STEPS_PER_EPISODE),
        "bc_episodes": (DRY_RUN_BC_EPISODES if args.dry_run else BC_EPISODES),
        "bc_random_episodes": (DRY_RUN_BC_RANDOM_EPISODES if args.dry_run else BC_RANDOM_EPISODES),
        "bc_train_frac": x1002.BC_TRAIN_FRAC,
        "adapter_passes": (DRY_RUN_ADAPTER_PASSES if args.dry_run else ADAPTER_PASSES),
        "adapter_batch": x1002.ADAPTER_BATCH, "adapter_lr": x1002.ADAPTER_LR,
        "adapter_class": "experiments.v3_exq_734...PPOPolicyNet",
        "adapter_trunk_hidden": int(x734.PPO_TRUNK_HIDDEN),
        "adapter_init_reseeded_per_fit": True,
        "projection_dim": PROJECTION_DIM,
        "agreement_bar": AGREEMENT_BAR, "agreement_elevation_min": AGREEMENT_ELEVATION_MIN,
        "untrained_control_margin": UNTRAINED_CONTROL_MARGIN,
        "rebasis_lift_min": REBASIS_LIFT_MIN, "rebasis_flat_eps": REBASIS_FLAT_EPS,
        "reader_shortfall_min": READER_SHORTFALL_MIN,
        "lda_ridge_frac": LDA_RIDGE_FRAC, "whiten_eps": WHITEN_EPS, "decode_ridge": DECODE_RIDGE,
        "ws250_full_input_floor": WS250_FULL_INPUT_FLOOR,
        "ws250_variance_explained_ceiling": WS250_VARIANCE_EXPLAINED_CEILING,
        "decision_subspace_retention_ceiling": DECISION_SUBSPACE_RETENTION_CEILING,
        "decision_world_state_indices": DECISION_WORLD_STATE_INDICES,
        "raw_field_control_floor": RAW_FIELD_CONTROL_FLOOR,
        "oracle_majority_ceiling": x1002.ORACLE_MAJORITY_CEILING,
        "heldout_min_steps": x1002.HELDOUT_MIN_STEPS,
        "participation_ratio_floor": PARTICIPATION_RATIO_FLOOR,
        "seed_majority": SEED_MAJORITY, "arms": ARM_IDS,
        "dry_run": bool(args.dry_run),
    }
    out_path = write_flat_manifest(
        result, None, dry_run=args.dry_run,
        config=full_config, seeds=seeds, script_path=Path(__file__),
        started_at=t0, z_goal_stream_stats=_ZG.stats(),
    )
    print("outcome: %s (%s)" % (result["outcome"], result["interpretation"]["label"]), flush=True)

    _outcome_raw = str(result["outcome"]).upper()
    emit_outcome(
        outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=out_path,
        dry_run=args.dry_run,
    )
