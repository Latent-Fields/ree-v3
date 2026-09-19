"""V3-EXQ-1043a -- MECH-537: is the oracle's decision content ENCODED BUT NOT EXPOSED?

INSTRUMENT REPAIR of V3-EXQ-1043 (FAIL, `routing_signature_incomplete_undetermined`,
run `..._20260916T111630Z_v3`). SAME scientific question, same claim, same arms, same
estimator. An ALPHABETIC SUFFIX, not a new EXQ number, per the confirmed autopsy
`failure_autopsy_V3-EXQ-1043_2026-09-17.json` `routing_detail.recommended_form`.

Tests MECH-537 (communication-subspace routing failure) per its registered
`what_would_answer` and proposal EXP-1403 (minted 2026-09-15, user-approved, hand-off
`/queue-experiment`). Read-only over a FROZEN encoder: no `ree_core` change, no new
substrate, no behavioural DV.

  SENDER    X = the full 250-dim `world_state` -- the tensor the encoder ACTUALLY reads.
  RECEIVER  Y = sense()-time `z_world` (32-dim) -- the tensor the consumer ACTUALLY reads.
  ESTIMATOR   `experiments/_lib/interface_probe.communication_subspace` (reduced-rank
              regression, rank chosen by grouped cross-validation on held-out EPISODES).

QUESTION. On a source that passes adequacy, is the oracle's action decodable from the full
sender X yet POORLY decodable from the estimated communication subspace `P_comm X`, with the
orthogonal complement carrying the decodability and the consumer insensitive to that
complement? That is failure class F2 (routing) as opposed to F1 (absent from sender) or F3
(consumer insensitivity).

EXPERIMENT_PURPOSE = "diagnostic"

SLEEP DRIVER: not applicable -- no sleep flag is set (the x734 all-ON stack at this rung
enables no sleep loop). Recorded as sleep_driver_pattern="none".

red-team (see the RED-TEAM RECORD at the end of this docstring and the V3-EXQ-1043 queue
entry note).

=== WHAT 1043a CHANGES, AND WHAT RATIFIED EACH CHANGE ===

Four changes, all from `routing_detail.required_changes` of the CONFIRMED autopsy. Nothing
else about the design moves: the claim, the sender, the receiver, the estimator, the four
arms, the decoder protocol and the premise route are all V3-EXQ-1043 verbatim.

(R1) C2 IS JUDGED AGAINST A WITHIN-RUN PERMUTATION NULL, NOT A HAND-SET FLOOR.
     Autopsy: "GIVE C2 A WITHIN-RUN REFERENCE DISTRIBUTION -- refit the RRR on shuffled
     sender-receiver pairing to build a permutation null, so the orientation contrast is
     judged against the instrument's own noise rather than a hand-set 0.05 floor."
     C2's DEFINITION is unchanged (`D_randrank - D_comm`, the only rank-matched contrast).
     What changed is its reference: `ORIENTATION_MARGIN = 0.05` is no longer scored, and the
     per-seed statistic is a one-sided permutation p-value against `PERMUTATION_ALPHA`.
     THE PERMUTATION IS WITHIN EPISODE GROUP, AND THAT IS LOAD-BEARING. `_kfold_indices`
     (interface_probe.py) builds a GROUPED k-fold: every row of an episode lands entirely in
     one fold. Shuffling the receiver rows ACROSS episodes would pair a test episode's sender
     rows with a training episode's receiver rows, so a test row's target is also literally a
     training target -- the fit is optimistic, the null inflates, and the test is biased
     toward calling the observed contrast unremarkable. Shuffling WITHIN each episode keeps
     every row's group label and its partner's group label identical, so the fold structure
     is bit-identical to the observed fit's while the timestep-level pairing -- where the
     routing signal lives -- is destroyed.
     THE `ws250_randrank` COMPARATOR IS HELD FIXED ACROSS REPLICATES, deliberately. It is the
     same draw that enters the observed C2, so the comparison is PAIRED and the common term
     cancels: the test is equivalent to asking whether `D_comm` sits below the distribution of
     `D_comm_perm`. Re-drawing it per replicate would add the draw's noise to the null without
     matching it in the observed statistic, costing a second decoder fit per replicate for
     strictly less power.

(R2) C2 IS READ AT THE PARSIMONIOUS RANK, NOT AT rank = dy.
     Autopsy: "RE-ESTIMATE AND REPORT C2 AS A FUNCTION OF RANK, and read it at the
     PARSIMONIOUS rank (8-10, where heldout r2 is already within 0.001 of its maximum). The
     low-rank premise the biology supplies is not testable at rank = dy, where the RRR fit is
     unconstrained OLS; it is testable there."
     `_parsimonious_rank` = the SMALLEST rank whose grouped-CV held-out R^2 is within
     `PARSIMONIOUS_R2_TOL` of the ladder maximum. A within-run, cross-validated rule, not a
     hand-set rank. Replaying it over V3-EXQ-1043's landed `rrr_heldout_r2_by_rank` gives
     8 / 10 / 10 on seeds 42/43/44 -- exactly the autopsy's figures.
     The comm / perp / randrank arms are instantiated at BOTH ranks; `ws250_full` is
     rank-independent and is fitted once. C2 at the CV-selected rank is still computed and
     RECORDED, so the 1043 comparison stays direct, but the SCORED C2 is the parsimonious one.
     WITHDRAWN by the autopsy and NOT done here: "EXTEND THE RRR RANK LADDER -- mathematically
     impossible (rank <= dy = 32; the ladder is already 1..32)."

(R3) C4b's ABSOLUTE CEILING IS ANCHORED TO A MEASURED, IN-RUN ATTAINABLE FLOOR.
     Autopsy: "ANCHOR C4b's ABSOLUTE 0.5 CEILING from a measured reference, the way the
     readiness anchors already are against V3-EXQ-1008."
     There was no such reference to import. Searched exhaustively before this driver was
     written: `v3_exq_1002`, `v3_exq_1008` and `v3_exq_1010` contain ZERO occurrences of
     `sensitivity_ratio` / `dz_world` / `INSENSITIV`; the only other evidence manifests
     carrying `sensitivity_ratio` (V3-EXQ-1000, V3-EXQ-1006) measure unrelated statistics. No
     run had ever measured this quantity except V3-EXQ-1043 itself, and NOTHING measured
     anywhere sat at or below 0.50 on the AIMED probe (lowest observed 0.8318). So the 0.50
     ceiling had no demonstrated reachability at all.
     The repair, user-ratified 2026-09-19 (decision chip
     `chip-20260918-mech537-c4b-ceiling-anchor`, OPTION C): MEASURE the reference in-run.
     `_jacobian_aligned_basis` builds the rank-r subspace of the STANDARDISED sender that the
     encoder is most sensitive to -- the top-r eigenvectors of the state-averaged
     `J_std^T J_std`, `J_std` the finite-difference Jacobian `d z_world / d (standardised
     sender)` at the canonical post-reset state -- and the IDENTICAL `_decision_sensitivity`
     machinery is then run on it. The resulting ratio `F` is the best-case routing reference
     for THIS encoder at THIS rank.
     HONEST LABEL: `F` is the ratio achieved by the encoder's own most-sensitive rank-r input
     subspace. That is the natural best-case routing subspace and it is what makes the ceiling
     reachable-by-measurement, but it is an ATTAINABLE reference, not a certified infimum --
     the ratio also depends on the un-normalised component weights a, b, which this
     construction does not separately optimise. Recorded under that name.
     The run also records `I`, the ISOTROPIC / no-routing reference
     (`mean_raw_norm_complement_component / mean_raw_norm_comm_component` on the FITTED
     decomposition) -- retention-MATCHED by construction, and verified to be the autopsy's own
     statistic: recomputing its C4a retention-geometry prediction from the 1043 manifest's
     component norms reproduces the autopsy's stated 0.348 / 0.267 / 0.365 exactly as
     0.3478 / 0.2667 / 0.3647.
     The floor -> ceiling RULE is `_c4b_ceiling`, a single pre-registered expression fixed
     BEFORE execution and NOT tunable after seeing data, computed PER SEED from that seed's
     own F and I. Per-seed is not a new choice: C5 already scores against a MEASURED per-seed
     chance level for exactly the same reason (an absolute bar means something different at
     every CV-selected rank).
     Do NOT promote C4a to the scored conjunct -- autopsy-forbidden, retention-confounded by
     the driver's own stated design decision. It stays recorded, with its confound stated.

(R4) n IS RAISED TO 6 AND `SEED_MAJORITY` IS RE-SPECIFIED PROPORTIONALLY TO 4.
     Autopsy: "IF n IS RAISED, RE-SPECIFY SEED_MAJORITY PROPORTIONALLY in the pre-registration
     (it is a fixed constant 2, inherited from x1002). Otherwise the seed-majority clause
     becomes trivially satisfiable and the successor's PASS is partly an artefact of the seed
     count."
     The inherited rule is 2 of 3 = 2/3, so 6 seeds -> 4. `SEED_MAJORITY` is therefore a LOCAL
     constant here and NOT `x1002.SEED_MAJORITY`; leaving it at the inherited 2 while raising
     n would convert a majority clause into "any 2 seeds".
     Carried verbatim, because it is the honest expectation and not a hedge: "at adequate n
     the likely outcome is a confidence interval EXCLUDING 0.05 -- i.e. H1 falsified rather
     than rescued. Raise n to settle the question, not to pass it."

NOT CHANGED, and each was explicitly considered:
  - `EXPERIMENT_PURPOSE` stays "diagnostic". The autopsy's debt class is "complex
    (probe-gated) / puzzle (known rules) ... what is missing is a FACT ... That is a spike".
    Promoting the contract to "evidence" is a scope change it did not authorise.
  - No lit-pull. `secondary_routing: none`;
    `targeted_review_mutual_legibility_communication_subspaces` already carries
    Semedo2019 / Binish2026 / Gonzalez2026 for exactly this question.
  - No criterion asserts an exact committed ACTION. `torch.multinomial` returns a different
    category across machine classes from a bit-identical probability tensor at the same seed
    (CLAUDE.md, "Running the test suite"), so every criterion here stays upstream of the
    discrete quantizer -- decode AGREEMENT and encoder SENSITIVITY, as in 1043.

STATED LIMITATION, carried forward rather than re-litigated: the encoder under test carries
the SD-106 limitation and the lineage ran with `p0a_field_weight_on = 0.0`, so the shipped P0
objective supplies no gradient toward decision-relevant content at the very encoder whose
communication subspace is being estimated. SD-106 is an OPEN `substrate_queue.json` entry but
its severity is `degrading`, not `corrupting`, so the skill's Step 2.5c gate does not block.
It is why `source_adequacy_ws250_full` is a PRECONDITION (1043 measured 0.9335 against a 0.80
floor) rather than an assumption.

=== WHY THIS SOURCE, AND WHY NOT z_world ===

V3-EXQ-1010 (`..._20260909T195348Z_v3.json`, PASS, `interpretation.label: H-F-confirmed`)
established that no decoder in a five-rung capacity ladder recovers the oracle above the
0.80 bar from the frozen 978-OFF `z_world`, on a protocol whose calibration anchor is sound
and at a capacity that demonstrably memorises the training split. So **z_world is DISQUALIFIED
as a SENDER** -- EXP-1403 says so explicitly, and using it would confound "no routing" with
"nothing to route".

The adequate sources measured on this exact protocol are (1008 authoring-time, seed 42):

    ws250_full        0.940      the uncompressed 250-dim world_state
    rawfield25        0.9735     world_state[225:250], the agent-centred resource field
    ws250_pca32       0.884      PCA-32 of world_state

This driver uses **the full 250-dim world_state**, for a reason the 25-dim field cannot
satisfy: MECH-537 is about directions the receiver COULD have read and did not. The encoder's
input IS `world_state`; `resource_field_view` is a 25-dim SLICE of it
(`world_state[225:250]`, verified at runtime by this driver's `--self-test`). An orthogonal
complement computed inside the 25-dim slice would consist of directions that are still inside
the encoder's input, but it would exclude 90% of that input from the complement -- and a
routing claim measured on 10% of the sender is not a routing claim about the sender. Taking
X = world_state makes P_comm and its complement a genuine orthogonal decomposition of
everything the encoder receives.

That is also what makes the causal leg well-posed: `world_state` is an ARGUMENT to
`agent.sense(...)`, so a perturbation along a chosen direction can actually be pushed
through the frozen encoder. A perturbation of `resource_field_view` could not be -- that key
is never handed to `sense()`.

=== THE FOUR ARMS -- IDENTICAL DECODER, IDENTICAL AMBIENT DIMENSION ===

All four arms feed a `x734.PPOPolicyNet` (literally the class V3-EXQ-978 instantiated as its
reader, at `x734.PPO_TRUNK_HIDDEN`) with a **250-dim** tensor, trained by the identical
cross-entropy protocol on the identical rows. The arms differ ONLY in which linear subspace
of the standardised sender the features are confined to:

    ws250_full        X                       the adequacy anchor / positive control
    ws250_comm        P_comm X                the estimated communication subspace, rank r
    ws250_perp        (I - P_comm) X          its orthogonal complement, rank 250 - r
    ws250_randrank    U U^T X                 a RANDOM orthonormal subspace of THE SAME rank r

The RRR rank ladder is 1..32 = 1..dy, the whole mathematically meaningful range (at rank dy
the rank constraint is inactive and the fit IS unconstrained OLS). The shared instrument
defaults to 1..min(16, dx, dy) and the mid-scale smoke showed cross-validation SATURATING at
that 16 -- the ladder, not the data, choosing the rank, which would have made "the rank
selected by cross-validation" untrue of this run. `rrr_rank_at_ladder_ceiling` is recorded per
seed either way.

Because every arm's ambient dimension is 250, `in_dim` is identical and the capacity match
holds by construction -- there is no in_dim confound (1008 had one: rawfield 25 against
ws250 250, and had to reason around it).

**ONLY C2 IS RANK-MATCHED, and that is the whole architecture of the criteria set.** C1
(`D_full - D_comm`) and the complement arm both compare subspaces of DIFFERENT rank, so a
difference between them is confounded by dimensionality; they are recorded as the PHENOTYPE.
C2 (`D_randrank - D_comm`, identical rank) is what licenses an ORIENTATION reading, and it is
the criterion marked `load_bearing`. C3 is stated as RETENTION against the full sender
(`D_full - D_perp <= tol`, an upper bound) rather than as `D_perp - D_comm`, precisely so that
it too is free of the rank confound and says what the claim says: deleting the communication
subspace costs essentially nothing.

**`ws250_randrank` is the load-bearing control and the reason this design can answer
anything.** A rank-r projection of a 250-dim input loses information WHATEVER its
orientation, so `D_full > D_comm` on its own is a statement about DIMENSIONALITY, not about
routing. Only `D_randrank > D_comm` -- the oriented subspace decoding WORSE than a random
subspace of the same rank -- says the communication subspace is actively oriented AWAY from
the decision directions. That is MECH-537's actual content, and it is criterion C2,
`load_bearing: true`.

=== EVERYTHING IS DONE IN THE STANDARDISED SENDER BASIS (a design-critical choice) ===

The train-split z-score (x1002's `_fit_standardiser`, fitted on the TRAIN EPISODES only) is
applied to X **once, before the RRR**, and every projection then acts in that standardised
basis. It is NOT re-fitted per arm.

This is not a stylistic choice. Per-arm standardisation would divide each projected arm by
its own per-dimension std, and a projected tensor has many near-zero-variance dimensions --
so `ws250_comm` and `ws250_perp` would have numerical noise amplified by up to 1/eps while
`ws250_full` would not. `D_comm` would then be depressed for a PREPROCESSING reason and the
run would fake a MECH-537 confirmation. Standardising once, up front, makes all four arms
exact linear maps of the same tensor.

=== THE CAUSAL LEG: AIMED AT THE FIVE COORDINATES THE ORACLE ACTUALLY READS ===

"The consumer demonstrably insensitive to the complement" is scored on a probe that is AIMED,
not sampled, and the distinction is the difference between measuring the claim and measuring
something adjacent to it.

The oracle is `LocalViewGreedyPolicy`: its action is the argmax over the five destination cells
of the agent-centred field view, which are `world_state` indices {232, 236, 237, 238, 242}
(x1008's `DECISION_WORLD_STATE_INDICES`, derived from the policy's own move deltas). A
direction drawn generically from the complement puts only about 5/(live - r) ~ 1.5% of its
energy on those five coordinates, so a generic complement/comm sensitivity ratio measures
whether the encoder's Jacobian is ISOTROPIC off the fitted subspace -- not whether the DECISION
CONTENT reaches the receiver. Those are different questions and only the second is MECH-537's.

So, for each decision coordinate e_j:

  1. express e_j in the STANDARDISED basis (divide by the fitted per-dimension std) -- the SAME
     basis in which C1/C2/C3 decompose the sender, so C3's complement and C4's complement are
     the same subspace rather than two subspaces differing by a non-orthogonal diagonal map;
  2. split it into its communication-subspace component and its complement component;
  3. map each component back to RAW sender space, normalise to a unit raw direction, and apply
     it at the same absolute magnitude `eps_abs`, at two magnitudes;
  4. report `mean ||dz_world|| (complement component) / mean ||dz_world|| (comm component)`.

No random draw is involved at all: the probe directions are named by the oracle's own
definition. The NULL CONTROL repeats the identical construction on a RANDOM subspace of the
same rank AND the same live-dimension support, so it differs in exactly one thing -- whether
the splitting subspace was fitted or drawn at random -- and its ratio is RECORDED. It is deliberately
NOT scored as a conjunct: the components are taken UN-NORMALISED (their weights in e_j are part
of the routing statement -- a unit-normalised ratio drops the factor b/a and can read
"insensitive" while the complement path in fact dominates), and once the weights are in, a
random subspace is no longer matched on RETENTION, so a null margin would be partly a retention
difference. C4 is therefore the ABSOLUTE ratio alone: control-free, and reachable by hypothesis
rather than by assumption -- under a genuine routing failure the encoder Jacobian is supported
on the communication subspace and the ratio goes to ~0, while under no routing failure it is
~b/a > 1 (the dry-run smoke measured 6.46 at that end).

The GENERIC whole-subspace probe (directions drawn as differences of two real held-out
observations, so they stay on the data manifold) is still run and recorded, but as a
DIAGNOSTIC only -- it is not a criterion.

Measured at the CANONICAL POST-RESET RECURRENT STATE, deliberately. `sense()` advances the
agent's recurrent state and sense-time z_world is
`(world_encoder(w) + world_topdown(beta_to_split(z_beta))) * prec`, so a trajectory-state probe
would mix the encoder's dependence on `w` with recurrent drift, and the baseline and perturbed
calls could not start from the same state at all. Resetting before every measurement holds the
top-down term fixed across every condition, which isolates exactly the quantity the claim is
about: WHICH OBSERVATION DIRECTIONS REACH THE CONSUMER-FACING REPRESENTATION.

STATED LIMITATION, not papered over: this probes the FEEDFORWARD path only. If a complement
direction influenced z_world solely through the recurrent term, this leg would miss it. That is
why the trajectory-based variance-routing readouts (`heldout_r2_from_comm` /
`heldout_r2_from_complement`, closed-form, computed where the recurrent term is live) are
recorded alongside -- as CORROBORATING diagnostics, never as criteria, because a rank-r RRR's
complement is expected a priori to predict Y less well and gating on that would be close to
tautological. Those numbers use PER-COLUMN R^2 normalisation and are NOT comparable to
`rrr_heldout_r2`, which comes from `interface_probe._r2` and normalises against a single global
mean; both are recorded and the difference is stated so nobody compares them.

=== THE PREMISE CHECK (this is a FALSIFIER the claim itself registers) ===

MECH-537's registered falsifier includes: "OR the subspace estimate is unstable across frame or
receiver-state strata, in which case the single-subspace premise fails and the question belongs
to MECH-547 / MECH-555." That, and only that, is the premise route here:

  - `principal_angles` between the per-seed bases (CROSS-SEED), which always counts -- those
    are replications of an identical protocol, so a disagreement is about the estimate;
  - `principal_angles` between the basis fitted on ORACLE-driven visitation and the one fitted
    on RANDOM-driven visitation (CROSS-STRATUM), which counts ONLY where the two strata's own
    data spans overlap. A ridge RRR solution lies in the row space of the visited data, so
    strata that visit different regions yield different bases for an IDENTICAL encoder; reading
    that as receiver-state dependence would be a mis-attribution. The span overlap is measured
    with the same principal-angle statistic on the strata's own top-r principal directions and
    recorded per seed.

Either routes to MECH-547 / MECH-555 with `evidence_direction: non_contributory` -- never
`mixed`, which would connote a measured-but-equivocal effect on a run where nothing about
MECH-537 was measured.

WITHDRAWN, and recorded because the withdrawal is part of the design: an earlier draft also let
a high complement-sensitivity ratio trigger this premise route. That was a second,
driver-invented operationalisation of a falsifier the claim states in terms of STRATA, and it
could have fired while the bases were demonstrably stable and the RRR explained the receiver
input at R^2 ~ 0.999 -- a `non_contributory` verdict nobody could attribute. A high ratio now
simply fails C4 and lands `undetermined`, with every continuous margin recorded.

=== TWO REACHABLE FALSIFICATIONS, NOT ONE ===

Given V3-EXQ-1010's H-F result, the claim's own registered falsifier -- the target as decodable
inside the subspace as in the full sender -- is unlikely to be reached on this source, so a grid
resting on it alone could confirm but not falsify. The reachable falsification is `C1 AND NOT
C2`: the phenotype IS present (the target drops inside the communication subspace) but a RANDOM
subspace of the SAME RANK drops it as far, so the drop is DIMENSIONALITY rather than a subspace
oriented away from the decision directions -- which falsifies MECH-537's distinctive content
and routes to F1 / MECH-532 instead. It is scored `weakens` only when the C2 contrast actually
discriminated; a structurally-zero contrast is `undetermined`, never a falsification.

=== SUBSTRATE-PATH OVERLAP GATE (skill Step 2.5c) -- the call, recorded ===

Four open `substrate_queue.json` entries carry `severity: corrupting`:
`MECH-320` (ree_core/policy/tonic_vigor.py, status substrate_landed_pending_behavioural_validation),
`contextmemory-write-path-addressing-degeneracy` (ree_core/predictors/e1_deep.py::ContextMemory.write,
implemented_pending_validation), `sd_blocked_agency_mismatch_floor_calibration`
(ree_core/affect/blocked_agency.py, implemented_pending_validation), and
`sd105_frozen_shared_entropy_floor_multiplier` (ree_core/regulators/selection_entropy_floor.py,
status proposed_REGISTRATION_ONLY_not_a_build_authorisation).

NOT BLOCKED, and the reasoning is recorded here so a later autopsy can disagree with it
rather than guess:

  (a) All four are POLICY / AFFECT / E1-MEMORY modules. They are exercised in the WARMUP,
      which produces the frozen encoder; NONE of them is read by the MEASUREMENT, which is a
      supervised decode from stored observations to ORACLE labels plus frozen forward passes
      through `sense()`. The agent's own policy never appears in any readout.
  (b) This run's object of study is the 978-OFF frozen encoder AS IT ACTUALLY IS -- the same
      object V3-EXQ-1002 / 1008 / 1010 studied, and the same object whose H-F verdict is the
      load-bearing input to EXP-1403's own design. A defect that changes WHICH encoder the
      warmup produces does not change the question, which is asked of whatever encoder is
      produced and is gated in-run by this run's own adequacy precondition.
  (c) Three of the four are already `implemented*` awaiting validation -- the defect is
      fixed-pending-confirmation, not live. The fourth (`sd105`) is by its own title a
      design constraint on "a difference-of-arms design whose DV is the entropy it
      regulates"; this run's DV is decode agreement and encoder sensitivity, and it arms no
      entropy set-point, so it is inapplicable by its own terms.

The `degrading` entries that DO touch this run's measurement path are recorded as known
limitations, per the gate's degrading rule: **SD-018** and **SD-106**
(`ree_core/latent/stack.py`, `ree_core/latent/zworld_p0.py`,
`agent.py::compute_resource_proximity_loss` -- the observation->z_world bottleneck, i.e.
precisely the mechanism under test), **SD-MECH303-THRESHOLD-SOURCING** and
**mech357-freeze-incompatible-pressure-mechanism** (`ree_core/environment/causal_grid_world.py`).

=== DV-SYMMETRY INVARIANCE, DECLARED PER ARM (skill Step 3) ===

Every arm's DV is held-out top-1 agreement between a trained decoder's argmax over 5 action
logits and the oracle's action. The symmetry group of that DV is: a uniform additive constant
across the 5 logits, any monotone rescaling of them, and a permutation of interchangeable
candidates.

  ws250_full / ws250_comm / ws250_perp / ws250_randrank -- the manipulation is a change of
  the LINEAR SUBSPACE the decoder's INPUT is confined to. It is not invariant under any of
  those three: a projection changes the information content of the input state-by-state, so
  the induced change in the logits is state-dependent, not a broadcast constant, not a
  monotone map of the logits, and not a relabelling of candidates. Two of the four arms
  additionally differ in the RANK of their input subspace and all four differ in retained
  norm on the five oracle decision coordinates -- both measured and recorded.

The causal leg's DV is `||dz_world||`, a norm, whose symmetry group is rotations of z_world
and permutations of the perturbation draws; the manipulation is the SUBSPACE the perturbation
direction is drawn from, which is not a rotation of z_world nor a reordering of draws.

=== PRE-REGISTERED, BEFORE EXECUTION ===

Every threshold below is a module constant, inherited where an inherited value exists
(AGREEMENT_BAR / AGREEMENT_ELEVATION_MIN / SEED_MAJORITY / HELDOUT_MIN_STEPS come from
x1002 unchanged, calibrated on this exact task, oracle and trivial-predictor family). None is
derived from this run's own statistics.

=== RED-TEAM RECORD ===

red-team (fable): see the V3-EXQ-1043 queue entry `note` for the verdict and dispositions.
"""

from __future__ import annotations

import argparse
import datetime
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from experiment_protocol import emit_outcome  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402
from experiments._metrics import (  # noqa: E402
    P0NotReady,
    check_degeneracy,
    p0_readiness_gate,
)
from experiments._lib.arm_fingerprint import arm_cell  # noqa: E402
from experiments._lib.interface_probe import (  # noqa: E402
    communication_subspace,
    principal_angles,
)
from experiments._lib.readiness_anchor import assert_anchor_reachable  # noqa: E402
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator  # noqa: E402
from experiments._lib.zworld_encoder_guard import (  # noqa: E402
    latent_stack_snapshot,
    latent_stack_weight_delta,
)
import experiments.v3_exq_1002_zworld_actor_adequacy_oracle_adapter as x1002  # noqa: E402
import experiments.v3_exq_1008_zworld_adequacy_portfolio_ws250_rebasis as x1008  # noqa: E402
import experiments.v3_exq_734_env_difficulty_competence_recovery_sweep as x734  # noqa: E402
import experiments.v3_exq_737_ree_latent_policy_head_competence_probe as x737  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_1043a_mech537_communication_subspace_permutation_null"
QUEUE_ID = "V3-EXQ-1043a"
SUPERSEDES = "V3-EXQ-1043"
EXPERIMENT_PURPOSE = "diagnostic"
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"

# MECH-537 is the single claim this run's implementation actually tests. The other claims in
# its depends_on edge set (MECH-517 / MECH-532 / MECH-518 / ARC-139 / INV-105 / INV-088) are
# DISTINGUISHED-FROM relations, not mechanisms exercised here, and tagging them would
# increment their evidence counts on a run that never touches a decoder collapse or a missing
# decompression stage. Single claim, so no `evidence_direction_per_claim` is required.
CLAIM_IDS: List[str] = ["MECH-537"]
BEARS_ON: List[str] = ["MECH-547", "MECH-555", "SD-080", "SD-106"]

DEVICE = torch.device("cpu")

# ---- inherited apparatus (NEVER re-defined -- see the docstring) --------------------------
RUNG = x1002.RUNG
RUNG_ID = x1002.RUNG_ID
LEVEL_ID = x1002.LEVEL_ID
WORLD_STATE_DIM = 250
RESOURCE_FIELD_OFFSET = x1008.RESOURCE_FIELD_OFFSET          # 225
DECISION_INDICES = list(x1008.DECISION_WORLD_STATE_INDICES)  # [232, 236, 237, 238, 242]

ZWORLD_P0_EPISODES = x1002.ZWORLD_P0_EPISODES        # 60
P0_EPISODES = x1002.P0_WARMUP_EPISODES               # 200
P1_EPISODES = x1002.P1_REINFORCE_EPISODES            # 90
STEPS_PER_EPISODE = x1002.STEPS_PER_EPISODE          # 200

BC_EPISODES = x1002.BC_EPISODES                    # 40 oracle-driven
BC_RANDOM_EPISODES = x1002.BC_RANDOM_EPISODES      # 20 random-driven
BC_TRAIN_FRAC = x1002.BC_TRAIN_FRAC                # 0.7, split BY EPISODE
ADAPTER_PASSES = x1002.ADAPTER_PASSES              # 60
STANDARDISER_EPS = x1002.STANDARDISER_EPS

ZWORLD_DELTA_FLOOR = x1002.ZWORLD_DELTA_FLOOR              # 1e-6
PARTICIPATION_RATIO_FLOOR = x1002.PARTICIPATION_RATIO_FLOOR  # 2.0

# ---- PRE-REGISTERED THRESHOLDS -----------------------------------------------------------
# Inherited, calibrated on this exact task/oracle/trivial-predictor family in V3-EXQ-1002 and
# carried unchanged through 1008 and 1010. Not re-derived here.
AGREEMENT_BAR = x1002.AGREEMENT_BAR                    # 0.80
AGREEMENT_ELEVATION_MIN = x1002.AGREEMENT_ELEVATION_MIN  # 0.20
# LOCAL, NOT x1002.SEED_MAJORITY -- see (R4) in the docstring. The inherited rule is 2 of
# 3 = 2/3; at n = 6 the proportional re-specification is 4. Leaving the inherited 2 in
# place while raising n would turn "a majority of seeds" into "any 2 seeds" and make a
# PASS partly an artefact of the seed count -- which the autopsy names as the failure mode.
SEED_MAJORITY = 4                                      # 4 of 6 (2/3, as inherited)
SEED_MAJORITY_INHERITED_RULE = "2 of 3 (x1002) -> 4 of 6, proportional"
HELDOUT_MIN_STEPS = x1002.HELDOUT_MIN_STEPS            # 500

# New to this run, and each one is justified where it is used in `_adjudicate`.
RRR_R2_FLOOR = 0.50            # the RRR must explain the receiver input, else "the estimated
                               # communication subspace" is not a subspace of anything
RANDRANK_CONTROL_MARGIN = 0.05  # the matched-rank random control must clear the trivial
                                # predictor by this much, else C2's comparator cannot move
ROUTING_DROP_MIN = 0.05        # absolute floor on a paired positive contrast (assay spec 1.6)
DELTA_SD_MULTIPLE = 2.0        # ... and mean(delta) >= 2 * SD(delta) across seeds (spec 1.6)
# NO LONGER SCORED (R1). Retained as a RECORDED reference line only, so 1043a's C2 numbers
# stay directly comparable to V3-EXQ-1043's and the old predicate is recomputable from the
# manifest. The scored C2 reference is the within-run permutation null below.
ORIENTATION_MARGIN_RECORDED_ONLY = 0.05   # C2 reference line from V3-EXQ-1043, unscored here
COMPLEMENT_RETENTION_TOL = 0.05  # C3: D_full - D_perp, an UPPER bound. Removing the
                               # communication subspace must cost essentially NOTHING -- that
                               # is the claim's own "no information having been destroyed".
                               # Deliberately NOT the earlier `D_perp - D_comm`: the
                               # complement has rank (live - r) against the subspace's r, so
                               # that difference is confounded by DIMENSIONALITY in exactly
                               # the way C1 is, and only C2 is rank-matched. Retention against
                               # the FULL sender is the rank-confound-free statement, and it
                               # is what the claim actually asserts.
EQUIVALENCE_BAND = 0.05        # TOST band for "no routing failure" (spec 1.6; same magnitude
                               # as the positivity floor, so the two cannot both be satisfied)
# C4 is scored against its OWN NULL CONTROL, not only against an absolute bar. The mid-scale
# smoke measured the null (a RANDOM rank-r subspace and its complement, identical machinery)
# at 0.905 -- near the 1.0 chance value the construction predicts, which validates the probe --
# and an absolute-only gate at 0.25 would then have been close to unmeetable by construction
# for a nonlinear MLP encoder, exactly the anchor-reachability failure mode. The load-bearing
# half is therefore the MARGIN BELOW THE NULL, which is measured on the same encoder with the
# same machinery and so cannot be unreachable by construction; the absolute ceiling is retained
# as a conjunct at a value the phrase "insensitive" can honestly carry.
INSENSITIVITY_NULL_MARGIN = 0.15   # C4a: null_ratio - measured_ratio, paired per seed.
                                   # RECORDED, NEVER SCORED (autopsy-forbidden: retention-
                                   # confounded). Kept only so C4a stays on the record.
# C4b: the ABSOLUTE ceiling is no longer a hand-set constant. It is computed PER SEED by
# `_c4b_ceiling` from two quantities this run MEASURES on that seed's own encoder (R3):
#   F = `jacobian_aligned_floor_ratio`  -- the attainable best-case routing reference
#   I = `isotropic_reference_ratio`     -- the no-routing reference from retention geometry
# The RULE is fixed here, before execution, and is not tunable after seeing data.
C4B_CEILING_RULE = "arithmetic_midpoint_of_measured_floor_and_isotropic"
INSENSITIVITY_RATIO_MAX_LEGACY_1043 = 0.50   # recorded for comparability; NOT scored
# ---- R1: the within-run permutation null for C2 -----------------------------------------
# N_PERMUTATIONS drives cost linearly (one RRR refit + one decoder fit per replicate), so it
# is the one knob here with a real budget consequence. 200 gives a minimum attainable
# one-sided p of 1/201 = 0.00498, comfortably below alpha; the staged design
# (evidence/planning/v3_exq_1043a_mech537_design_staged_20260918.md) pre-registers 100 as an
# acceptable fallback (min p 1/101 = 0.0099), and which of the two is in force is recorded.
N_PERMUTATIONS = 200
# One-sided. This is the conventional permutation-test alpha, NOT a domain effect-size
# constant -- and in particular it is NOT the withdrawn ORIENTATION_MARGIN, which was also
# 0.05. The collision is coincidental and is called out so no reader conflates them.
PERMUTATION_ALPHA = 0.05
# ---- R2: the parsimonious rank ----------------------------------------------------------
# The SMALLEST rank whose grouped-CV held-out R^2 is within this tolerance of the ladder
# maximum. The autopsy's own wording ("within 0.001 of its maximum"); replaying it over
# V3-EXQ-1043's landed rrr_heldout_r2_by_rank reproduces its stated 8 / 10 / 10.
PARSIMONIOUS_R2_TOL = 1.0e-3
# ---- R3: the attainable-floor probe -----------------------------------------------------
# States used for the finite-difference Jacobian. The Jacobian is RANK-INDEPENDENT, so it is
# built once per seed and the top-r eigenvectors are taken per rank.
N_JACOBIAN_STATES = 32
# Finite-difference step for the Jacobian, as a fraction of the mean held-out centred sender
# norm -- the SAME scale the sensitivity probe uses, so the floor is measured in the same
# regime as the quantity it anchors.
JACOBIAN_EPS_FRAC = 0.05
STABILITY_MARGIN_OVER_CHANCE = 0.15  # the cross-stratum basis overlap must clear the MEASURED
                               # chance level (two independent rank-r subspaces in this
                               # sender's live dims) by this margin. Self-calibrating: chance
                               # overlap scales with rank/dim, so an absolute bar would mean
                               # something different at every CV-selected rank.
SPAN_OVERLAP_MIN = 0.50        # a cross-stratum basis disagreement counts against the premise
                               # only when the two strata's own data spans overlap at least
                               # this much -- otherwise the bases differ because the visitation
                               # differs, not because the receiver state does (F7)
SUBSPACE_STABILITY_MIN = 0.50  # min mean_squared_cosine_overlap (cross-seed and cross-stratum)

# FROZEN POSITIVE-CONTROL REFERENCES for the readiness anchors, copied verbatim from
# `evidence/experiments/v3_exq_1008_zworld_adequacy_portfolio_ws250_rebasis_20260907T233826Z_v3.json`
# `arm_results[]` -- the SAME arms, protocol, oracle and trivial-predictor family this driver
# reuses. They are what makes each gate demonstrably REACHABLE rather than a hand-written
# predicate narrower than the state it anchors to (validate_experiments' anchor-reachability
# warning; failure_autopsy_SD-068-rem-fanout-cluster_2026-07-18 sec 2).
REF_1008_WS250_FULL_AGREEMENT = [0.9399441480636597, 0.933527410030365, 0.9424936175346375]
REF_1008_WS250_FULL_ELEVATION = [0.3738361597061157, 0.35322660207748413, 0.3704834580421448]
REF_1008_WS250_RANDPROJ_ELEVATION = [0.2202048897743225, 0.20621061325073242,
                                     0.17251908779144287]
REF_1008_SOURCE = ("v3_exq_1008_zworld_adequacy_portfolio_ws250_rebasis_"
                   "20260907T233826Z_v3.json arm_results[] seeds 42/43/44")

# 1..32. The shared instrument DEFAULTS to 1..min(16, dx, dy), and the mid-scale smoke showed
# cross-validation SATURATING at that 16 -- i.e. the ladder, not the data, was choosing the
# rank, which would make "the RRR rank selected by cross-validation" (MECH-537's own
# operationalisation) untrue of this run. 32 = dy, the largest meaningful RRR rank: at rank dy
# the rank constraint is inactive and the fit IS unconstrained OLS, so there is nothing beyond
# it to select and a selection AT 32 is a genuine cross-validated verdict rather than a ladder
# artefact. `rrr_rank_at_ladder_ceiling` is recorded per seed either way.
RRR_RANKS = list(range(1, 33))
RRR_FOLDS = 5
# RELATIVE ridge, scaled in-run to mean(diag(Xc^T Xc)). An ABSOLUTE 1e-6 is what the shared
# instrument defaults to and it is NOT usable on this sender: `world_state` is largely a 5x5x7
# one-hot local view, so many of its 250 dimensions are exactly constant on a split, land
# exactly at zero after the train-split z-score, and make Xc^T Xc EXACTLY SINGULAR -- a ridge
# eight orders of magnitude below the other diagonal entries does not rescue the LU. Caught by
# this driver's own --dry-run smoke (`torch.linalg.solve: the input matrix is singular`), which
# is why the smoke runs before anything is queued. Scaling the ridge to the design makes the
# solve well-conditioned at any sample size.
RRR_RIDGE_REL = 1.0e-4
# Dimensions whose TRAIN-SPLIT raw std sits at or below the standardiser's eps floor are
# exactly constant: they are identically zero after standardisation, carry no information for
# any decoder, and are pure padding in the RRR design. They are dropped from the RRR FIT ONLY
# and the fitted basis is embedded back into the full 250-dim ambient space with zero rows, so
# every arm still presents the decoder with 250 dims (the capacity match is untouched) and the
# dropped dims land, correctly and harmlessly, in the orthogonal complement.
SENDER_DIM_STD_FLOOR = STANDARDISER_EPS

N_SENSITIVITY_STATES = 64
N_SENSITIVITY_DIRECTIONS = 16
SENSITIVITY_EPS_FRACS = (0.05, 0.10)   # of the mean held-out centred sender norm

# n RAISED from 3 to 6 (R4). 42/43/44 are the 1002 / 1008 / 1010 / 1043 lineage seeds and are
# kept so the 1043 comparison is direct; 45/46/47 extend it. Not a reef config at this rung,
# so the seed-44 reef-instability rule does not apply (x1002 line ~557).
SEEDS = [42, 43, 44, 45, 46, 47]

DRY_RUN_SEEDS = [42]
DRY_RUN_ZWORLD_P0 = x1002.DRY_RUN_ZWORLD_P0
DRY_RUN_P0 = x1002.DRY_RUN_P0
DRY_RUN_P1 = x1002.DRY_RUN_P1
DRY_RUN_STEPS = x1002.DRY_RUN_STEPS
DRY_RUN_BC_EPISODES = x1002.DRY_RUN_BC_EPISODES
DRY_RUN_BC_RANDOM_EPISODES = x1002.DRY_RUN_BC_RANDOM_EPISODES
DRY_RUN_ADAPTER_PASSES = x1002.DRY_RUN_ADAPTER_PASSES
DRY_RUN_SENS_STATES = 4
DRY_RUN_SENS_DIRECTIONS = 3
DRY_RUN_PERMUTATIONS = 3
DRY_RUN_JACOBIAN_STATES = 2

# ---- arms --------------------------------------------------------------------------------
ARM_FULL = "ws250_full"
ARM_COMM = "ws250_comm"
ARM_PERP = "ws250_perp"
ARM_RAND = "ws250_randrank"
ARM_IDS = [ARM_FULL, ARM_COMM, ARM_PERP, ARM_RAND]

_ZG = ZGoalStreamAccumulator()


# ==========================================================================================
# helpers
# ==========================================================================================
def _mean(vals: Sequence[float]) -> float:
    vals = [float(v) for v in vals if v is not None and np.isfinite(float(v))]
    return float(np.mean(vals)) if vals else float("nan")


def _sd(vals: Sequence[float]) -> float:
    vals = [float(v) for v in vals if v is not None and np.isfinite(float(v))]
    return float(np.std(vals, ddof=0)) if len(vals) > 1 else 0.0


def _worst(vals: Sequence[float], cells: Sequence[str]) -> Tuple[float, str]:
    """The WORST (minimum) value and the cell id that carries it.

    Reported instead of a mean wherever a precondition's `met` is a worst-case claim, so the
    indexer's recompute reads the same statistic the claim makes (skill Step 3).
    """
    pairs = [(float(v), str(c)) for v, c in zip(vals, cells)
             if v is not None and np.isfinite(float(v))]
    if not pairs:
        return (float("nan"), "")
    return min(pairs, key=lambda p: p[0])


def _contrast_discriminated(per_seed: Sequence[float], eps: float = 1e-9) -> bool:
    """Did this contrast's two arms actually produce different agreements anywhere?

    An all-zero delta vector is a VACUOUS pass, not a null result: it means the projection
    changed nothing the decoder could see. Reported as `criteria_non_degenerate[...] = false`
    so the indexer flags it rather than scoring it.
    """
    vals = [float(v) for v in per_seed if v is not None and np.isfinite(float(v))]
    return bool(vals) and float(max(abs(v) for v in vals)) > float(eps)


def _random_orthonormal(in_dim: int, k: int, seed: int) -> torch.Tensor:
    """[in_dim, k] with orthonormal columns, deterministic in `seed`.

    Reimplemented locally (identically in substance to `interface_probe._random_orthonormal`
    and `x1008._random_orthonormal`) rather than reaching into another module's private name.
    A dedicated generator, so this draw never perturbs the cell's own RNG stream.
    """
    g = torch.Generator().manual_seed(int(seed))
    q, _r = torch.linalg.qr(torch.randn(int(in_dim), int(k), generator=g, dtype=torch.float64))
    return q.contiguous().float()


def _live_sender_dims(st: Dict[str, Any], d: int) -> torch.Tensor:
    """Indices of the sender dimensions that actually VARY on the train split.

    See SENDER_DIM_STD_FLOOR. Falls back to every dimension if no standardiser was fitted.
    """
    if not st.get("fitted"):
        return torch.arange(int(d))
    sd = st["std"].reshape(-1)
    keep = (sd > float(SENDER_DIM_STD_FLOOR)).nonzero(as_tuple=False).reshape(-1)
    return keep if int(keep.numel()) > 0 else torch.arange(int(d))


def _rrr_ridge_abs(x: torch.Tensor) -> float:
    """RRR_RIDGE_REL scaled to the design's mean diagonal energy (see RRR_RIDGE_REL)."""
    if int(x.shape[0]) < 2:
        return float(RRR_RIDGE_REL)
    xc = (x - x.mean(dim=0, keepdim=True)).to(torch.float64)
    diag_mean = float((xc * xc).sum(dim=0).mean())
    return float(RRR_RIDGE_REL) * max(diag_mean, 1.0)


def _embed_basis(basis_sub: torch.Tensor, keep: torch.Tensor, d: int) -> torch.Tensor:
    """Lift a [len(keep), k] basis back into the full [d, k] ambient sender space.

    Orthonormality is preserved exactly: the added rows are zeros, so the Gram matrix is
    unchanged. Asserted in --self-test.
    """
    out = torch.zeros(int(d), int(basis_sub.shape[1]), dtype=basis_sub.dtype)
    out[keep] = basis_sub
    return out


def _project(x: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    """Orthogonal projection of rows of `x` [N, d] onto the column space of `basis` [d, k].

    Returns a [N, d] AMBIENT tensor, not [N, k]: every arm must present the decoder with the
    same `in_dim`, so the capacity match holds by construction (docstring, THE FOUR ARMS).
    """
    if int(basis.shape[1]) == 0:
        return torch.zeros_like(x)
    return (x @ basis) @ basis.T


def _heldout_r2_linear(x_tr: torch.Tensor, y_tr: torch.Tensor,
                       x_te: torch.Tensor, y_te: torch.Tensor,
                       ridge: float = 1.0e-6) -> float:
    """Held-out R^2 of a ridge linear map x -> y, fitted on train, scored on test.

    CORROBORATING DIAGNOSTIC ONLY -- never a criterion. See the docstring's STATED LIMITATION.
    """
    if int(x_tr.shape[0]) < 2 or int(x_te.shape[0]) < 2:
        return float("nan")
    xt = x_tr.to(torch.float64)
    yt = y_tr.to(torch.float64)
    xm = xt.mean(dim=0, keepdim=True)
    ym = yt.mean(dim=0, keepdim=True)
    xc = xt - xm
    b = torch.linalg.solve(xc.T @ xc + ridge * torch.eye(xc.shape[1], dtype=torch.float64),
                           xc.T @ (yt - ym))
    pred = (x_te.to(torch.float64) - xm) @ b + ym
    yv = y_te.to(torch.float64)
    ss_res = float(((yv - pred) ** 2).sum())
    ss_tot = float(((yv - yv.mean(dim=0, keepdim=True)) ** 2).sum())
    if ss_tot <= 0.0:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def _ws250_features(episodes: List[Dict[str, Any]]
                    ) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    """(X [N, 250], y [N], episode_id [N]) in the SAME row order every extractor here walks.

    `x1002._zworld_features` walks episodes in order and steps in order; so does this, so the
    two are aligned row-for-row by construction and the episode ids double as the RRR's
    grouped-CV keys (never a per-row split: adjacent grid-world steps are strongly correlated).
    """
    xs: List[torch.Tensor] = []
    ys: List[int] = []
    gids: List[int] = []
    for ep_i, ep in enumerate(episodes):
        for obs, lab in zip(ep["obs"], ep["labels"]):
            w = obs.get("world_state")
            if w is None:
                raise KeyError("obs_dict has no 'world_state' -- this driver cannot run "
                               "without the encoder's own input tensor.")
            t = w if isinstance(w, torch.Tensor) else torch.as_tensor(w)
            xs.append(t.reshape(-1).float().to(DEVICE))
            ys.append(int(lab))
            gids.append(int(ep_i))
    if not xs:
        return (torch.zeros(0, WORLD_STATE_DIM), torch.zeros(0, dtype=torch.long), [])
    return (torch.stack(xs), torch.tensor(ys, dtype=torch.long), gids)


def _episode_ids(episodes: List[Dict[str, Any]]) -> List[int]:
    out: List[int] = []
    for ep_i, ep in enumerate(episodes):
        out.extend([int(ep_i)] * len(ep["labels"]))
    return out


def _trivial_agreement(episodes_tr: List[Dict[str, Any]],
                       episodes_te: List[Dict[str, Any]],
                       y_tr: torch.Tensor, y_te: torch.Tensor,
                       action_dim: int) -> Dict[str, Any]:
    """The STRONGEST TRIVIAL predictor on this task, and the majority-class one.

    x1002 measured previous-executed-action at 0.568-0.582 held-out -- far above the majority
    class -- so elevation is reported over the stronger of the two, exactly as 1002 does.
    """
    maj_te, maj_share = x1002._state_blind_agreement(y_tr, y_te, action_dim)
    prev_te = x1002._prev_action_vector(episodes_te)
    prev_agree = None
    if int(prev_te.shape[0]) == int(y_te.shape[0]) and int(y_te.shape[0]) > 0:
        prev_agree = float((prev_te == y_te).float().mean().item())
    cands = [v for v in (maj_te, prev_agree) if v is not None]
    strongest = max(cands) if cands else None
    return {"majority_class_agreement": maj_te,
            "majority_class_train_share": float(maj_share),
            "prev_action_agreement": prev_agree,
            "strongest_trivial_agreement": strongest}


# ==========================================================================================
# the sensitivity probe (the causal leg)
# ==========================================================================================
def _sensitivity(agent, obs_rows: List[Dict[str, Any]], basis_raw: torch.Tensor,
                 eps_abs: float, n_dirs: int, seed: int,
                 direction_pool: Optional[torch.Tensor] = None) -> Dict[str, float]:
    """Mean ||dz_world|| for unit perturbations drawn INSIDE and OUTSIDE `basis_raw`.

    Both conditions perturb by the SAME absolute raw-space magnitude `eps_abs` along a UNIT
    raw-space direction, so the comparison is norm-matched at the encoder's input. Every
    measurement is taken from the canonical post-reset recurrent state (docstring, THE CAUSAL
    LEG) -- the baseline is recomputed under the same reset immediately before each perturbed
    call, so the two never share a mutated state.

    ON-MANIFOLD DIRECTIONS (`direction_pool`, [M, d] of real held-out sender rows). The raw
    direction is drawn as a DIFFERENCE OF TWO REAL OBSERVATIONS, not as an isotropic Gaussian.
    This matters and the mid-scale smoke is what showed it: the RRR explained 99.9% of
    `z_world` variance on the trajectory while an isotropic-Gaussian complement perturbation
    still moved z_world 76% as much as a communication-subspace one. There is no contradiction
    -- R^2 is measured ON the data manifold and an isotropic direction leaves it -- but it
    means an isotropic probe scores the encoder's response in a regime it never operates in,
    which is not what "the consumer is insensitive to the complement" asserts. A difference of
    two observed states is on-manifold by construction and needs no PCA or threshold. The
    Gaussian fallback is kept only for a degenerate/empty pool.
    """
    d = int(basis_raw.shape[0])
    k = int(basis_raw.shape[1])
    g = torch.Generator().manual_seed(int(seed))
    pool_n = int(direction_pool.shape[0]) if direction_pool is not None else 0
    inside: List[float] = []
    outside: List[float] = []
    for obs in obs_rows:
        w = torch.as_tensor(obs["world_state"]).reshape(-1).float()
        # The baseline is a pure function of (reset state, obs), so it is IDENTICAL for every
        # direction at this state -- computed once. `agent.reset()` immediately before each
        # sense() is what makes that true and is what keeps the baseline and the perturbed
        # call from ever sharing a mutated recurrent state.
        agent.reset()
        z0 = x737._agent_zworld(agent, obs)
        for _j in range(int(n_dirs)):
            if pool_n >= 2:
                i0 = int(torch.randint(pool_n, (1,), generator=g).item())
                i1 = int(torch.randint(pool_n, (1,), generator=g).item())
                gv = (direction_pool[i0] - direction_pool[i1]).reshape(-1).float()
                if float(gv.norm()) <= 1e-12:
                    gv = torch.randn(d, generator=g)
            else:
                gv = torch.randn(d, generator=g)
            proj = basis_raw @ (basis_raw.T @ gv) if k > 0 else torch.zeros(d)
            din = proj
            dout = gv - proj
            for vec, sink in ((din, inside), (dout, outside)):
                nrm = float(vec.norm())
                if nrm <= 1e-12:
                    continue
                obs_p = dict(obs)
                obs_p["world_state"] = (w + float(eps_abs) * (vec / nrm))
                agent.reset()
                z1 = x737._agent_zworld(agent, obs_p)
                sink.append(float((z1 - z0).norm()))
    m_in = _mean(inside)
    m_out = _mean(outside)
    ratio = float(m_out / m_in) if (np.isfinite(m_in) and m_in > 1e-12) else float("nan")
    return {"mean_dz_inside": m_in, "mean_dz_outside": m_out,
            "sensitivity_ratio": ratio,
            "direction_source": ("observed_state_differences" if pool_n >= 2
                                 else "isotropic_gaussian_fallback"),
            "n_direction_pool": int(pool_n),
            "n_inside": len(inside), "n_outside": len(outside)}


def _decision_sensitivity(agent, obs_rows: List[Dict[str, Any]], basis_std: torch.Tensor,
                          std_vec: torch.Tensor, eps_abs: float) -> Dict[str, float]:
    """||dz_world|| for perturbations along the COMM and COMPLEMENT components of each of the
    FIVE coordinates the oracle actually reads. This is what C4 scores (see the F1 note).

    The split is taken in the STANDARDISED basis -- the same decomposition C1/C2/C3 use -- and
    each component is then mapped to raw sender space and normalised, so the two conditions are
    unit raw directions at matched magnitude AND refer to the same subspace the decode arms do.
    Deterministic: no random draw is involved at all, because the probe directions are named by
    the oracle's own definition rather than sampled.
    """
    d = int(basis_std.shape[0])
    k = int(basis_std.shape[1])
    inside: List[float] = []
    outside: List[float] = []
    per_index: Dict[str, Dict[str, float]] = {}
    for j in DECISION_INDICES:
        e_raw = torch.zeros(d)
        e_raw[int(j)] = 1.0
        e_std = e_raw / std_vec                       # raw -> standardised coordinates
        comp_in_std = basis_std @ (basis_std.T @ e_std) if k > 0 else torch.zeros(d)
        comp_out_std = e_std - comp_in_std
        # UN-NORMALISED components, deliberately. In raw space e_j = raw_in + raw_out exactly
        # (the split is orthogonal in the standardised basis, so a^2 + b^2 need not equal 1
        # here -- the decomposition, not the Pythagorean identity, is what is exact). The
        # consumer's response to a change in the decision coordinate is a*J(u_in) + b*J(u_out),
        # so a ratio taken between UNIT directions silently drops the weights a and b -- and
        # they are exactly what carries the routing statement: where the comm retention `a` is
        # LOW (which is the regime in which C1/C2 pass) a unit-normalised ratio is deflated by
        # b/a and C4 can read "insensitive" while the complement path in fact dominates what
        # the consumer sees. Perturbing by the components themselves keeps the weights inside
        # the measurement. The unit-normalised per-unit-direction ratio is still recorded, as a
        # diagnostic, so both readings are on the record.
        vals: Dict[str, float] = {}
        norms: Dict[str, float] = {}
        for name, comp_std, sink in (("in", comp_in_std, inside),
                                     ("out", comp_out_std, outside)):
            raw_dir = comp_std * std_vec              # standardised -> raw coordinates
            nrm = float(raw_dir.norm())
            norms[name] = nrm
            if nrm <= 1e-12:
                vals[name] = float("nan")
                vals[name + "_per_unit"] = float("nan")
                continue
            tot = 0.0
            n = 0
            for obs in obs_rows:
                w = torch.as_tensor(obs["world_state"]).reshape(-1).float()
                agent.reset()
                z0 = x737._agent_zworld(agent, obs)
                obs_p = dict(obs)
                obs_p["world_state"] = (w + float(eps_abs) * raw_dir)
                agent.reset()
                z1 = x737._agent_zworld(agent, obs_p)
                tot += float((z1 - z0).norm())
                n += 1
            m = float(tot / n) if n else float("nan")
            vals[name] = m
            vals[name + "_per_unit"] = (float(m / nrm) if nrm > 1e-12 else float("nan"))
            if np.isfinite(m):
                sink.append(m)
        per_index[str(j)] = {
            "mean_dz_comm_component": vals.get("in", float("nan")),
            "mean_dz_complement_component": vals.get("out", float("nan")),
            "mean_dz_comm_per_unit": vals.get("in_per_unit", float("nan")),
            "mean_dz_complement_per_unit": vals.get("out_per_unit", float("nan")),
            "raw_norm_comm_component": norms.get("in", float("nan")),
            "raw_norm_complement_component": norms.get("out", float("nan")),
        }
    m_in = _mean(inside)
    m_out = _mean(outside)
    ratio = float(m_out / m_in) if (np.isfinite(m_in) and m_in > 1e-12) else float("nan")
    pu_in = _mean([v["mean_dz_comm_per_unit"] for v in per_index.values()])
    pu_out = _mean([v["mean_dz_complement_per_unit"] for v in per_index.values()])
    return {"mean_dz_inside": m_in, "mean_dz_outside": m_out, "sensitivity_ratio": ratio,
            "per_unit_sensitivity_ratio_diagnostic": (float(pu_out / pu_in)
                                                      if (np.isfinite(pu_in) and pu_in > 1e-12)
                                                      else float("nan")),
            "mean_raw_norm_comm_component":
                _mean([v["raw_norm_comm_component"] for v in per_index.values()]),
            "mean_raw_norm_complement_component":
                _mean([v["raw_norm_complement_component"] for v in per_index.values()]),
            "per_decision_index": per_index, "n_decision_indices": len(DECISION_INDICES)}


# ==========================================================================================
# per-seed work
# ==========================================================================================
def _run_seed(seed: int, action_dim: int, env_kwargs: Dict[str, Any],
              cfg_slice: Dict[str, Any], zworld_p0: int, p0: int, p1: int, steps: int,
              bc_eps: int, bc_rand: int, passes: int, n_sens_states: int,
              n_sens_dirs: int, dry_run: bool) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """One seed: collect, warm up ONCE, freeze, estimate P_comm, run the four arms + legs."""
    print("Seed %d Condition %s:comm_subspace_routing" % (seed, RUNG_ID), flush=True)

    # ---- dataset (shared by every arm, step-for-step) -------------------------------------
    ep_oracle = x1002._collect_episodes(seed, env_kwargs, "oracle", bc_eps, steps)
    ep_random = x1002._collect_episodes(seed, env_kwargs, "random", bc_rand, steps)
    tr_eps, te_eps = x1002._split_episodes(ep_oracle)

    # ---- warm up ONCE, then FREEZE --------------------------------------------------------
    print("Seed %d Condition %s:warmup" % (seed, RUNG_ID), flush=True)
    warm_env = x734._make_env(seed, env_kwargs)
    agent = x1002._make_agent(warm_env)
    before = latent_stack_snapshot(agent)
    x734._train_all_on_agent(
        agent, warm_env, seed=seed, p0_episodes=p0, p1_episodes=p1,
        steps_per_episode=steps, rung_id=RUNG_ID, total_denominator=(p0 + p1),
        zworld_p0_episodes=zworld_p0,
        zworld_p0_env=(x734._make_env(seed, env_kwargs) if zworld_p0 > 0 else None),
        zworld_p0_dry_run=dry_run,
        zworld_p0_resource_field_weight=0.0,   # 978's OFF arm, exactly as 1002/1008/1010
    )
    guard = latent_stack_weight_delta(agent, before)
    _ZG.observe(agent)
    # ENCODER FROZEN from here: every read below is under torch.no_grad (via _agent_zworld)
    # and no optimiser ever touches the agent again.

    # ---- sender / receiver tensors, aligned row-for-row ------------------------------------
    x_tr, y_tr, g_tr = _ws250_features(tr_eps)
    x_te, y_te, _g_te = _ws250_features(te_eps)
    x_rd, y_rd, g_rd = _ws250_features(ep_random)
    z_tr, _yz = x1002._zworld_features(agent, tr_eps)
    z_te, _yz2 = x1002._zworld_features(agent, te_eps)
    z_rd, _yz3 = x1002._zworld_features(agent, ep_random)
    assert int(x_tr.shape[0]) == int(z_tr.shape[0]), "sender/receiver row misalignment (train)"
    assert int(x_te.shape[0]) == int(z_te.shape[0]), "sender/receiver row misalignment (test)"

    # ---- ONE standardiser, fitted on the TRAIN split, applied everywhere -------------------
    st = x1002._fit_standardiser(x_tr)
    xs_tr = x1002._apply_standardiser(x_tr, st)
    xs_te = x1002._apply_standardiser(x_te, st)
    xs_rd = x1002._apply_standardiser(x_rd, st)

    # ---- the communication subspace, fitted on TRAIN episodes only -------------------------
    keep = _live_sender_dims(st, WORLD_STATE_DIM)
    ridge_tr = _rrr_ridge_abs(xs_tr[:, keep])
    css = communication_subspace(xs_tr[:, keep], z_tr, RRR_RANKS, groups=g_tr,
                                 n_folds=RRR_FOLDS, ridge=ridge_tr, seed=seed)
    rank = int(css.selected_rank)
    b_std = _embed_basis(css.basis.float(), keep, WORLD_STATE_DIM)   # [250, rank], orthonormal
    # F6 FIX: the random control is drawn INSIDE THE LIVE DIMENSIONS and embedded, so it has
    # exactly the same support as the fitted basis (whose rows on the ~114 train-constant dims
    # are exactly zero by `_embed_basis`). Drawn over all 250 it would squander ~46% of its
    # rank on dimensions that are identically zero after standardisation -- which handicaps
    # ARM_RAND against ARM_COMM and biases C2, the load-bearing criterion.
    b_rand = _embed_basis(_random_orthonormal(int(keep.numel()), rank, seed=seed * 7919 + 13),
                          keep, WORLD_STATE_DIM)
    print("  [rrr] seed=%d selected_rank=%d heldout_r2=%.4f folds=%d rows=%d"
          % (seed, rank, css.selected_heldout_r2, css.n_folds_used, css.n_rows), flush=True)

    # cross-stratum stability: the SAME estimator on the random-driven visitation
    css_rd = communication_subspace(xs_rd[:, keep], z_rd, RRR_RANKS, groups=g_rd,
                                    n_folds=RRR_FOLDS, ridge=_rrr_ridge_abs(xs_rd[:, keep]),
                                    seed=seed)
    stratum = principal_angles(b_std, _embed_basis(css_rd.basis.float(), keep,
                                                   WORLD_STATE_DIM))
    # F7 FIX: a ridge RRR solution lies in the ROW SPACE OF THE VISITED DATA, so two strata
    # that visit different parts of the observation space yield different bases for the SAME
    # encoder. A low cross-stratum overlap is therefore only evidence about receiver-state
    # dependence when the two strata's data spans themselves overlap. Measured, not assumed:
    # the same principal-angle statistic between the strata's own top-r principal directions.
    def _top_pcs(x: torch.Tensor, k: int) -> torch.Tensor:
        xc = (x - x.mean(dim=0, keepdim=True)).to(torch.float64)
        _u, _s, vt = torch.linalg.svd(xc, full_matrices=False)
        return vt[:max(1, min(int(k), int(vt.shape[0]))), :].T.float()
    span_overlap = principal_angles(_top_pcs(xs_tr[:, keep], rank),
                                    _top_pcs(xs_rd[:, keep], rank))
    # CHANCE LEVEL for the overlap statistic at THIS rank in THIS sender's live dimensions,
    # measured rather than assumed: two independently drawn rank-r orthonormal bases. It is
    # what makes every overlap number above interpretable (they are not near 0 for large r --
    # two random rank-32 subspaces of a 136-dim space already overlap at ~0.24).
    _chance = _mean([
        float(principal_angles(
            _random_orthonormal(int(keep.numel()), rank, seed=seed * 13 + 1000 * t),
            _random_orthonormal(int(keep.numel()), rank, seed=seed * 13 + 1000 * t + 7)
        )["mean_squared_cosine_overlap"]) for t in range(3)])

    # ---- geometry: retained norm on the FIVE oracle decision coordinates -------------------
    # 1008 measured PCA-32 retaining 0.23-0.29 of each decision direction and a random
    # orthonormal projection 0.29-0.39, so these numbers are directly comparable to a
    # published reference rather than free-floating.
    def _retained(basis: torch.Tensor) -> Dict[str, Any]:
        vals = []
        for j in DECISION_INDICES:
            e = torch.zeros(int(basis.shape[0]))
            e[int(j)] = 1.0
            vals.append(float((basis.T @ e).norm()))
        return {"per_decision_index": {str(j): v for j, v in zip(DECISION_INDICES, vals)},
                "max": float(max(vals)) if vals else float("nan"),
                "mean": _mean(vals)}

    geom = {"comm_subspace": _retained(b_std), "randrank_control": _retained(b_rand)}

    # ---- the four arms ---------------------------------------------------------------------
    trivial = _trivial_agreement(tr_eps, te_eps, y_tr, y_te, action_dim)
    feats = {
        ARM_FULL: (xs_tr, xs_te),
        ARM_COMM: (_project(xs_tr, b_std), _project(xs_te, b_std)),
        ARM_PERP: (xs_tr - _project(xs_tr, b_std), xs_te - _project(xs_te, b_std)),
        ARM_RAND: (_project(xs_tr, b_rand), _project(xs_te, b_rand)),
    }
    arm_rows: List[Dict[str, Any]] = []
    agreements: Dict[str, float] = {}
    for arm_id in ARM_IDS:
        f_tr, f_te = feats[arm_id]
        with arm_cell(seed, config_slice=dict(cfg_slice, arm_id=arm_id,
                                              arm_selected_rank=(rank if arm_id in
                                                                 (ARM_COMM, ARM_PERP, ARM_RAND)
                                                                 else None)),
                      script_path=Path(__file__), config_slice_declared=True,
                      include_driver_script_in_hash=False,
                      extra_ineligible_reasons=[
                          "frozen_agent_and_fitted_subspace_shared_across_arms_within_seed"]
                      ) as cell:
            net, train_stats = x1002._train_adapter(f_tr, y_tr, action_dim, passes, seed, arm_id)
            agree_te = x1002._agreement(net, f_te, y_te)
            agree_tr = x1002._agreement(net, f_tr, y_tr)
            strongest = trivial["strongest_trivial_agreement"]
            row = {
                "cell_id": "%s|seed%d" % (arm_id, seed),
                "arm_id": arm_id,
                "seed": int(seed),
                "feature_dim": int(f_tr.shape[1]),
                "subspace_rank": (rank if arm_id in (ARM_COMM, ARM_PERP, ARM_RAND) else None),
                "oracle_action_agreement": agree_te,
                "train_agreement_capacity_witness": agree_tr,
                "agreement_elevation": (None if (agree_te is None or strongest is None)
                                        else float(agree_te - strongest)),
                "adapter_training": train_stats,
                "capacity_match": x1002._capacity_report(net, int(f_tr.shape[1]), action_dim),
                "heldout_steps": int(f_te.shape[0]),
            }
            cell.stamp(row)
        agreements[arm_id] = float(agree_te) if agree_te is not None else float("nan")
        arm_rows.append(row)
        print("  [arm] seed=%d %s heldout_agreement=%.4f (train %.4f)"
              % (seed, arm_id, agreements[arm_id], (agree_tr or float("nan"))), flush=True)

    # ---- corroborating variance routing (diagnostic, never a criterion) --------------------
    r2 = {
        "heldout_r2_from_full": _heldout_r2_linear(xs_tr, z_tr, xs_te, z_te),
        "heldout_r2_from_comm": _heldout_r2_linear(_project(xs_tr, b_std), z_tr,
                                                   _project(xs_te, b_std), z_te),
        "heldout_r2_from_complement": _heldout_r2_linear(xs_tr - _project(xs_tr, b_std), z_tr,
                                                         xs_te - _project(xs_te, b_std), z_te),
    }

    # ---- the causal leg --------------------------------------------------------------------
    # Map the standardised-basis subspace back to RAW sender space and re-orthonormalise, so
    # the two perturbation conditions really are orthogonal complements OF world_state and a
    # unit direction in either is a unit direction at the encoder's input.
    std_vec = (st["std"].reshape(-1) if st.get("fitted") else torch.ones(WORLD_STATE_DIM))
    # F5 NOTE: `diag(std)` is NOT orthogonal, so the RAW-orthogonal complement of `b_raw` is
    # not the image of the STANDARDISED-orthogonal complement the decode arms use. The
    # decision-targeted probe C4 actually scores avoids the mismatch entirely by splitting in
    # the standardised basis and only THEN mapping to raw; `b_raw` drives the generic probe,
    # which is a secondary diagnostic. The rotation is measured rather than assumed.
    b_raw, _r = torch.linalg.qr(torch.diag(std_vec) @ b_std)
    b_raw = b_raw.contiguous()
    std_basis_rotation = principal_angles(b_std, b_raw)
    b_rand_live = _embed_basis(
        _random_orthonormal(int(keep.numel()), rank, seed=seed * 104729 + 7),
        keep, WORLD_STATE_DIM)
    b_raw_null, _rn = torch.linalg.qr(torch.diag(std_vec) @ b_rand_live)
    b_raw_null = b_raw_null.contiguous()

    probe_obs: List[Dict[str, Any]] = []
    for ep in te_eps:
        probe_obs.extend(ep["obs"])
    stride = max(1, len(probe_obs) // max(1, int(n_sens_states)))
    probe_obs = probe_obs[::stride][:int(n_sens_states)]
    centred = (x_te - x_te.mean(dim=0, keepdim=True)) if int(x_te.shape[0]) else x_te
    base_norm = float(centred.norm(dim=1).mean()) if int(centred.shape[0]) else 1.0

    # ---- F1 FIX: the DECISION-TARGETED probe, which is the one C4 scores ----------------
    # A direction drawn generically from the complement puts ~1.5% of its energy on the five
    # coordinates the oracle actually reads (5 of ~136 live dims), so a generic ratio measures
    # "is the encoder's Jacobian isotropic off the RRR subspace", NOT "does the decision
    # content reach the receiver through the communication subspace". The claim is about the
    # latter, and this substrate hands us the five coordinates EXACTLY (x1008's
    # DECISION_WORLD_STATE_INDICES, derived from LocalViewGreedyPolicy's own move deltas), so
    # the probe can be aimed instead of sampled. For each decision coordinate e_j we split it
    # into its communication-subspace and complement components IN THE STANDARDISED BASIS (the
    # SAME decomposition C1/C2/C3 use -- see the F5 note at `b_raw`), map each component back
    # to raw sender space, normalise to a unit raw direction, and perturb at matched eps.
    dec_sens: Dict[str, Any] = {"by_eps": {}}
    sens: Dict[str, Any] = {"base_sender_norm": base_norm, "by_eps": {}}
    for frac in SENSITIVITY_EPS_FRACS:
        eps_abs = float(frac) * base_norm
        # The null control consumes the SAME direction pool and the same eps, so it differs
        # from the measured condition in exactly one thing: whether the subspace the direction
        # is split by was FITTED or drawn at random.
        meas = _sensitivity(agent, probe_obs, b_raw, eps_abs, n_sens_dirs, seed=seed,
                            direction_pool=x_te)
        null = _sensitivity(agent, probe_obs, b_raw_null, eps_abs, n_sens_dirs,
                            seed=seed + 50021, direction_pool=x_te)
        sens["by_eps"][("%.3f" % frac)] = {
            "eps_frac": float(frac), "eps_abs": eps_abs,
            "measured": meas, "null_control": null,
        }
        dec_sens["by_eps"][("%.3f" % frac)] = {
            "eps_frac": float(frac), "eps_abs": eps_abs,
            "measured": _decision_sensitivity(agent, probe_obs, b_std, std_vec, eps_abs),
            "null_control": _decision_sensitivity(agent, probe_obs, b_rand_live, std_vec,
                                                  eps_abs),
        }
    primary_key = "%.3f" % SENSITIVITY_EPS_FRACS[-1]
    sens["sensitivity_ratio"] = sens["by_eps"][primary_key]["measured"]["sensitivity_ratio"]
    sens["null_sensitivity_ratio"] = \
        sens["by_eps"][primary_key]["null_control"]["sensitivity_ratio"]
    dec_sens["sensitivity_ratio"] = \
        dec_sens["by_eps"][primary_key]["measured"]["sensitivity_ratio"]
    dec_sens["null_sensitivity_ratio"] = \
        dec_sens["by_eps"][primary_key]["null_control"]["sensitivity_ratio"]
    print("  [sens] seed=%d DECISION ratio=%.4f null=%.4f | generic ratio=%.4f null=%.4f"
          % (seed, dec_sens["sensitivity_ratio"], dec_sens["null_sensitivity_ratio"],
             sens["sensitivity_ratio"], sens["null_sensitivity_ratio"]), flush=True)

    seed_row = {
        "seed": int(seed),
        "selected_rank": rank,
        "rrr_heldout_r2": float(css.selected_heldout_r2),
        "rrr_heldout_r2_by_rank": {str(k): float(v) for k, v in css.heldout_r2_by_rank.items()},
        "rrr_n_folds_used": int(css.n_folds_used),
        "rrr_n_rows": int(css.n_rows),
        "rrr_ridge_abs": float(ridge_tr),
        "n_sender_dims_live": int(keep.numel()),
        "n_sender_dims_dropped_constant": int(WORLD_STATE_DIM - int(keep.numel())),
        "cross_stratum_selected_rank": int(css_rd.selected_rank),
        "rrr_rank_at_ladder_ceiling": bool(rank >= max(RRR_RANKS)),
        "agreements": agreements,
        "trivial_predictors": trivial,
        "heldout_steps": int(x_te.shape[0]),
        "zworld_weight_delta": guard,
        "zworld_participation_ratio": x1002._participation_ratio(z_tr),
        "feature_standardisation": x1002._standardiser_report(st),
        "decision_coordinate_retention": geom,
        "variance_routing": r2,
        "sensitivity_decision_targeted": dec_sens,
        "sensitivity_generic_diagnostic": sens,
        "std_to_raw_basis_rotation": std_basis_rotation,
        "cross_stratum_subspace": stratum,
        "cross_stratum_data_span_overlap": float(span_overlap["mean_squared_cosine_overlap"]),
        "subspace_overlap_chance_level": float(_chance),
        "delta_full_minus_comm": float(agreements[ARM_FULL] - agreements[ARM_COMM]),
        "delta_randrank_minus_comm": float(agreements[ARM_RAND] - agreements[ARM_COMM]),
        "delta_perp_minus_comm": float(agreements[ARM_PERP] - agreements[ARM_COMM]),
        "delta_full_minus_perp": float(agreements[ARM_FULL] - agreements[ARM_PERP]),
        "basis_std": b_std,   # stripped before the manifest write; used for cross-seed angles
    }
    verdict = bool(np.isfinite(agreements[ARM_FULL]) and agreements[ARM_FULL] >= AGREEMENT_BAR)
    print("verdict: %s" % ("PASS" if verdict else "FAIL"), flush=True)
    return seed_row, arm_rows


# ==========================================================================================
# 1043a INSTRUMENT ADDITIONS (R1 / R2 / R3) -- every one of these is a PURE function of its
# arguments except the two that need the frozen agent, and all of them are exercised by
# --self-test without a multi-hour run.
# ==========================================================================================
def _parsimonious_rank(heldout_r2_by_rank: Dict[Any, float],
                       tol: float = PARSIMONIOUS_R2_TOL) -> int:
    """R2: the SMALLEST rank whose grouped-CV held-out R^2 is within `tol` of the maximum.

    Not a hand-set rank: a within-run, cross-validated rule over the ladder this run already
    computes. Replaying it over V3-EXQ-1043's landed `rrr_heldout_r2_by_rank` returns
    8 / 10 / 10 on seeds 42/43/44, which is exactly what the autopsy states.

    Why it matters: at rank = dy the RRR rank constraint is INACTIVE and the fit is
    unconstrained OLS, so "the low-rank channel content must pass through" -- the premise the
    biology supplies -- is not instantiated there and C2 measured there cannot test it. The
    1043 run selected rank 32 = dy on 3/3 seeds off a flat asymptote (r2 gains 31 -> 32 of
    ~5e-7), so this is not a hypothetical.
    """
    items = [(int(k), float(v)) for k, v in heldout_r2_by_rank.items()
             if v is not None and np.isfinite(float(v))]
    if not items:
        return 1
    best = max(v for _k, v in items)
    ok = [k for k, v in items if (best - v) <= float(tol)]
    return int(min(ok)) if ok else int(min(k for k, _v in items))


def _permute_within_groups(groups: Sequence[Any], gen: np.random.Generator) -> np.ndarray:
    """R1: a row permutation that shuffles ONLY WITHIN each episode group.

    THE WITHIN/ACROSS DISTINCTION IS THE WHOLE POINT, not a detail. `_kfold_indices` builds a
    GROUPED k-fold in which every row of an episode lands entirely in one fold. A permutation
    that moved receiver rows ACROSS episodes would pair a test episode's sender rows with a
    training episode's receiver rows, so a test row's target would also literally be a
    training target: the permuted fit becomes optimistic, the null inflates, and the test is
    biased toward declaring the OBSERVED contrast unremarkable -- i.e. biased against the
    claim in a way no reader could see from the manifest.

    Permuting within the group keeps each row's group label AND its partner's group label
    identical, so the fold structure is bit-identical to the observed fit's, while the
    timestep-level sender-receiver correspondence -- which is where the routing signal lives
    -- is destroyed. That is the null this design wants.
    """
    g = np.asarray(groups)
    idx = np.arange(int(g.shape[0]))
    out = idx.copy()
    for val in np.unique(g):
        where = np.nonzero(g == val)[0]
        if where.size > 1:
            out[where] = where[gen.permutation(where.size)]
    return out


def _permutation_p_value(observed: float, nulls: Sequence[float]) -> float:
    """One-sided permutation p: P(null statistic >= observed), with the standard +1 correction.

    The +1 on both numerator and denominator is not cosmetic -- it is what keeps the test
    valid (a p of exactly 0 is not attainable from a finite reference set), and it is what
    sets the minimum attainable p at 1/(N+1).
    """
    vals = [float(v) for v in nulls if v is not None and np.isfinite(float(v))]
    if not vals or not np.isfinite(float(observed)):
        return float("nan")
    n_ge = sum(1 for v in vals if v >= float(observed))
    return float((1 + n_ge) / (1 + len(vals)))


def _jacobian_std_gram(agent, obs_rows: List[Dict[str, Any]], std_vec: torch.Tensor,
                       keep: torch.Tensor, eps_abs: float) -> torch.Tensor:
    """R3: the state-averaged Gram matrix of the encoder's Jacobian, in the STANDARDISED basis.

    Returns [len(keep), len(keep)] = mean over states of `J_std^T J_std`, where
    `J_std = d z_world / d (standardised sender)`.

    FINITE DIFFERENCES, NOT AUTOGRAD, and deliberately. `x737._agent_zworld` reads the frozen
    encoder entirely under `torch.no_grad()`, and the quantity this Gram is used to anchor --
    the C4b sensitivity ratio -- is ITSELF a finite-difference measurement at the same eps. An
    autograd Jacobian and a finite-difference ratio would disagree wherever the encoder is
    nonlinear, and the anchor would then be measuring a different object from the thing it
    anchors. Same modality, same eps, same canonical post-reset state: the anchor and the
    anchored quantity are commensurable by construction.

    Every measurement is taken from the canonical post-reset recurrent state, exactly as
    `_sensitivity` and `_decision_sensitivity` do -- `agent.reset()` immediately before each
    `sense()`, so the baseline and the perturbed call never share a mutated state.

    The perturbation is a unit RAW coordinate direction at `eps_abs` (the same construction
    the probes use); the column is then scaled by that dimension's standardiser std to express
    the Jacobian in standardised coordinates, which is the basis `_decision_sensitivity`
    splits `e_j` in.
    """
    d = int(std_vec.reshape(-1).shape[0])
    live = [int(i) for i in keep.reshape(-1).tolist()]
    k = len(live)
    gram = torch.zeros(k, k, dtype=torch.float64)
    n_states = 0
    for obs in obs_rows:
        w = torch.as_tensor(obs["world_state"]).reshape(-1).float()
        agent.reset()
        z0 = x737._agent_zworld(agent, obs).reshape(-1)
        cols = torch.zeros(int(z0.shape[0]), k, dtype=torch.float64)
        for c, i in enumerate(live):
            e = torch.zeros(d)
            e[i] = 1.0
            obs_p = dict(obs)
            obs_p["world_state"] = (w + float(eps_abs) * e)
            agent.reset()
            z1 = x737._agent_zworld(agent, obs_p).reshape(-1)
            # raw column, then -> standardised coordinates (chain rule through w = std * ws)
            cols[:, c] = ((z1 - z0).double() / float(eps_abs)) * float(std_vec.reshape(-1)[i])
        gram += cols.T @ cols
        n_states += 1
    return gram / max(1, n_states)


def _jacobian_aligned_basis(gram: torch.Tensor, keep: torch.Tensor, d: int,
                            r: int) -> torch.Tensor:
    """R3: the rank-r STANDARDISED-basis subspace the encoder is most sensitive to.

    Top-r eigenvectors of the state-averaged `J_std^T J_std`, embedded back into the full
    ambient sender exactly as the fitted basis is (`_embed_basis`, zero rows on the
    train-constant dims), so it is a drop-in substitute for `b_std` in
    `_decision_sensitivity` and the two are compared on identical footing.

    WHAT THIS IS, STATED PRECISELY so a later autopsy does not over-read it: the ratio
    measured through this subspace is the best-case ROUTING reference for this encoder at this
    rank -- if any rank-r subspace could carry the consumer's response to the decision
    coordinates, this is the one that does. It is an ATTAINABLE reference, not a certified
    infimum of the ratio over all rank-r subspaces: the ratio also depends on the
    un-normalised component weights a and b of each `e_j`, which this construction does not
    separately optimise. Recorded under the name `jacobian_aligned_floor_ratio` for that
    reason, never as "the minimum".
    """
    g = gram.to(torch.float64)
    g = 0.5 * (g + g.T)   # symmetrise against finite-difference asymmetry
    evals, evecs = torch.linalg.eigh(g)
    order = torch.argsort(evals, descending=True)
    r = max(1, min(int(r), int(evecs.shape[1])))
    basis_sub = evecs[:, order[:r]].contiguous().float()
    return _embed_basis(basis_sub, keep, int(d))


def _c4b_ceiling(floor_ratio: float, isotropic_ratio: float) -> float:
    """R3: THE PRE-REGISTERED FLOOR -> CEILING RULE. Fixed here, before execution.

    `C4B_CEILING_RULE = "arithmetic_midpoint_of_measured_floor_and_isotropic"`:

        ceiling(seed) = (F + I) / 2

    where, on that seed's own encoder and at the scored rank,
        F = `jacobian_aligned_floor_ratio`  -- the attainable best-case ROUTING reference
        I = `isotropic_reference_ratio`     -- the NO-ROUTING reference, taken from the
            FITTED decomposition's own component norms
            (`mean_raw_norm_complement_component / mean_raw_norm_comm_component`), and so
            retention-MATCHED by construction rather than retention-confounded the way the
            random-subspace null that C4a uses is.

    WHY THIS RULE AND NOT A CONSTANT. It introduces no new number at all: both endpoints are
    MEASURED in-run, by the same machinery, on the same encoder. The reading is direct -- the
    consumer counts as insensitive to the complement iff the fitted subspace lands on the
    ROUTING half of the range between the best routing this encoder can support and no routing
    at all. And it cannot be unmeetable by construction, which is exactly what the hand-set
    0.50 of V3-EXQ-1043 turned out to be: nothing measured anywhere in the corpus sat at or
    below 0.50 on this probe (lowest observed 0.8318).

    PER SEED, not pooled. The RULE is fixed; the realised value moves because the ENCODER
    moves. This is not a new liberty -- C5 already scores against a MEASURED per-seed chance
    level, for the same reason (an absolute bar means something different at every rank).

    REACHABILITY GUARD, which survives under any candidate rule: if F is not strictly below I
    the encoder shows no measurable headroom between best-case routing and no routing at this
    rank, so no ceiling drawn between them means anything. Returns NaN, and the caller scores
    that seed as UNREACHABLE rather than as a FAIL -- an un-anchorable criterion must not
    print a verdict.
    """
    f, i = float(floor_ratio), float(isotropic_ratio)
    if not (np.isfinite(f) and np.isfinite(i)):
        return float("nan")
    if not (0.0 <= f < i):
        return float("nan")
    return float((f + i) / 2.0)


# ==========================================================================================
# THE VERDICT GRID -- a PURE function of named booleans, exercised by --self-test without a
# multi-hour run. A grid that can only be reached by the full run is a grid nobody checks.
# ==========================================================================================
def _paired_positive(deltas: Sequence[float], floor: float) -> Dict[str, Any]:
    """The assay spec's 1.6 positivity rule, as ONE testable function.

    Positive iff mean(delta) >= `floor` AND mean(delta) >= DELTA_SD_MULTIPLE * SD(delta)
    across seeds AND at least SEED_MAJORITY seeds individually clear `floor`.
    """
    vals = [float(d) for d in deltas if d is not None and np.isfinite(float(d))]
    m, s = _mean(vals), _sd(vals)
    n_clear = sum(1 for v in vals if v >= floor)
    ok = bool(vals) and (m >= floor) and (m >= DELTA_SD_MULTIPLE * s) and (n_clear >= SEED_MAJORITY)
    return {"mean": m, "sd": s, "n_seeds_clearing": int(n_clear),
            "floor": float(floor), "sd_multiple": float(DELTA_SD_MULTIPLE),
            "seeds_required": int(SEED_MAJORITY), "passed": bool(ok),
            "per_seed": vals}


def _c2_falsified(per_seed: Sequence[float]) -> bool:
    """Is the rank-matched orientation contrast NON-POSITIVE -- the random subspace doing at
    least as badly as the fitted one?

    `not c2` is NOT this. `c2` is `_paired_positive`, a conjunction of an absolute floor, a
    2*SD consistency clause and a seed-majority clause, so it fails on data whose contrast is
    positive on EVERY seed but noisy across them (e.g. [0.10, 0.30, 0.02]: mean 0.14, 2*SD
    0.236 -> not positive). Routing that to "a RANDOM subspace of the same rank drops it as far
    or further" would print a falsification whose text asserts the opposite of the measured
    sign. The falsification therefore carries its own predicate: the mean contrast is at or
    below zero AND a seed majority is individually at or below zero. Everything in between --
    positive but not consistently so -- is `undetermined`, which is what it is.
    """
    vals = [float(v) for v in per_seed if v is not None and np.isfinite(float(v))]
    if not vals:
        return False
    n_nonpos = sum(1 for v in vals if v <= 0.0)
    return bool(_mean(vals) <= 0.0 and n_nonpos >= SEED_MAJORITY)


def _adjudicate(premise_ok: bool, c1: bool, c2: bool, c3: bool, c4: bool,
                equivalent: bool, c2_falsified: bool
                ) -> Tuple[str, str, str, str]:
    """(outcome, label, evidence_direction, hypothesis_verdict).

    ORDER IS PRE-REGISTERED. The PREMISE is adjudicated first, because a claim about *which*
    subspace carries the content is unanswerable if there is no stable single subspace to talk
    about. The premise route is SUBSPACE STABILITY ONLY -- exactly the falsifier MECH-537
    registers ("the subspace estimate is unstable across frame or receiver-state strata") --
    and it routes to MECH-547 / MECH-555 as `non_contributory`, never `mixed`, since nothing
    about MECH-537 was measured in that case.

    An earlier draft also let a high complement-sensitivity ratio trigger the premise route.
    That was withdrawn: it is a second, driver-invented operationalisation of a falsifier the
    claim states in terms of strata, and it could fire while the bases were demonstrably stable
    and the RRR explained the receiver input at R^2 ~ 0.999 -- a verdict nobody could attribute.
    A high ratio now simply fails C4 and lands `undetermined`, with every continuous margin
    recorded.

    TWO REACHABLE FALSIFICATIONS, not one. The claim's own registered falsifier (the target is
    as decodable inside the subspace as in the full sender) is, given V3-EXQ-1010's H-F result,
    unlikely to be reached on this source -- so relying on it alone would leave a grid that can
    confirm but not falsify. The reachable falsification is `c1 and c2_falsified`: the
    phenotype IS present (the target drops inside the communication subspace) but a RANDOM
    subspace of the SAME RANK drops it as far or further, so the drop is DIMENSIONALITY, not
    orientation -- precisely the distinctive content of MECH-537, and not merely a null.

    `c2_falsified` is a POSITIVE predicate of its own (`_c2_falsified`), never `not c2`. See
    that function: the negation of a conjunctive positivity test is satisfied by data whose
    contrast is positive on every seed, and routing that to a falsification would print a
    verdict contradicting its own numbers.
    """
    if not premise_ok:
        return ("PASS",
                "single_subspace_premise_fails_route_mech547_mech555",
                "non_contributory",
                "MECH-537 not adjudicated: the communication-subspace estimate is not stable "
                "across seeds and/or across receiver-state strata whose data spans do overlap, "
                "so the single-subspace premise this claim is operationalised on does not "
                "hold. Routes to MECH-547 / MECH-555.")
    if equivalent:
        return ("PASS",
                "no_routing_failure_target_survives_the_subspace",
                "weakens",
                "MECH-537 FALSIFIED on this source: the target is as decodable inside the "
                "estimated communication subspace as in the full sender, within the "
                "pre-registered equivalence band. No routing failure here -- route to F3 "
                "(consumer insensitivity) or F1 (target absent from sender).")
    if c1 and c2 and c3 and c4:
        return ("PASS",
                "communication_subspace_routing_failure_confirmed",
                "supports",
                "MECH-537 CONFIRMED on this source: the oracle target is decodable from the "
                "full sender and poorly decodable from the estimated communication subspace, "
                "worse than from a RANDOM subspace of the SAME RANK (so the drop is "
                "orientation, not dimensionality), deleting the subspace from the sender costs "
                "essentially no decodability, and the receiver is insensitive to the "
                "complement components of the five coordinates the oracle actually reads.")
    if c1 and (not c2) and c2_falsified:
        return ("PASS",
                "routing_drop_explained_by_rank_not_orientation",
                "weakens",
                "MECH-537 FALSIFIED in its distinctive content: the phenotype is present -- the "
                "target does drop inside the estimated communication subspace -- but a RANDOM "
                "subspace of the SAME RANK drops it as far or further, so the drop is "
                "DIMENSIONALITY, not a subspace oriented away from the decision directions. "
                "That is a reduction in what reaches the receiver, not a routing failure in "
                "MECH-537's sense; route to F1 / MECH-532 (compression without a trained "
                "decompression stage) rather than to a re-exposure repair.")
    return ("FAIL",
            "routing_signature_incomplete_undetermined",
            "mixed",
            "MECH-537 UNDETERMINED: the measured contrasts are neither the full confirming "
            "signature, nor equivalent to the full sender within the band, nor a "
            "discriminated rank-explains-it falsification. The continuous per-seed margins "
            "are recorded; no verdict is claimed.")


# ==========================================================================================
def _config(dry_run: bool, zworld_p0: int, p0: int, p1: int, steps: int, bc_eps: int,
            bc_rand: int, passes: int, n_sens_states: int, n_sens_dirs: int) -> Dict[str, Any]:
    return {
        "env_kwargs": x734._env_kwargs_for_rung(RUNG),
        "rung_id": RUNG_ID,
        "level_id": LEVEL_ID,
        "sender": "world_state_250",
        "receiver_input": "sense_time_z_world",
        "zworld_p0_episodes": int(zworld_p0),
        "p0_warmup_episodes": int(p0),
        "p1_reinforce_episodes": int(p1),
        "steps_per_episode": int(steps),
        "p0a_field_weight_on": 0.0,
        "use_resource_field_head": True,
        "resource_field_dim": int(x1002.RESOURCE_FIELD_DIM),
        "bc_episodes": int(bc_eps),
        "bc_random_episodes": int(bc_rand),
        "bc_train_frac": float(BC_TRAIN_FRAC),
        "adapter_passes": int(passes),
        "adapter_batch": int(x1002.ADAPTER_BATCH),
        "adapter_lr": float(x1002.ADAPTER_LR),
        "adapter_trunk_hidden": int(x734.PPO_TRUNK_HIDDEN),
        "feature_standardisation": "train_split_zscore_once_before_projection",
        "standardiser_eps": float(STANDARDISER_EPS),
        "rrr_ranks": list(RRR_RANKS),
        "rrr_folds": int(RRR_FOLDS),
        "rrr_ridge_relative": float(RRR_RIDGE_REL),
        "sender_dim_std_floor": float(SENDER_DIM_STD_FLOOR),
        "rrr_group_key": "episode",
        "n_sensitivity_states": int(n_sens_states),
        "n_sensitivity_directions": int(n_sens_dirs),
        "sensitivity_eps_fracs": list(SENSITIVITY_EPS_FRACS),
        "dry_run": bool(dry_run),
    }


def run_experiment(seeds: List[int], dry_run: bool = False) -> Dict[str, Any]:
    t0 = time.perf_counter()
    zworld_p0 = DRY_RUN_ZWORLD_P0 if dry_run else ZWORLD_P0_EPISODES
    p0 = DRY_RUN_P0 if dry_run else P0_EPISODES
    p1 = DRY_RUN_P1 if dry_run else P1_EPISODES
    steps = DRY_RUN_STEPS if dry_run else STEPS_PER_EPISODE
    bc_eps = DRY_RUN_BC_EPISODES if dry_run else BC_EPISODES
    bc_rand = DRY_RUN_BC_RANDOM_EPISODES if dry_run else BC_RANDOM_EPISODES
    passes = DRY_RUN_ADAPTER_PASSES if dry_run else ADAPTER_PASSES
    n_sens_states = DRY_RUN_SENS_STATES if dry_run else N_SENSITIVITY_STATES
    n_sens_dirs = DRY_RUN_SENS_DIRECTIONS if dry_run else N_SENSITIVITY_DIRECTIONS

    # ---- ANCHOR REACHABILITY, asserted BEFORE any compute is spent -----------------------
    # Each shipped predicate is scored against the frozen 1008 positive control. A gate the
    # known-good control cannot clear is a guaranteed false negative -- it would report
    # met=false on every run forever and mislabel an instrument-specification gap as a
    # substrate verdict. Raising here costs seconds; discovering it after the warmup costs
    # the whole run.
    anchor_reachability = [
        assert_anchor_reachable(
            anchor_name="source_adequacy_ws250_full",
            reference_cells=REF_1008_WS250_FULL_AGREEMENT,
            score_fn=(lambda v: float(v) >= float(AGREEMENT_BAR)),
            threshold=1.0, reference_source=REF_1008_SOURCE),
        assert_anchor_reachable(
            anchor_name="source_elevation_over_strongest_trivial",
            reference_cells=REF_1008_WS250_FULL_ELEVATION,
            score_fn=(lambda v: float(v) >= float(AGREEMENT_ELEVATION_MIN)),
            threshold=1.0, reference_source=REF_1008_SOURCE),
        assert_anchor_reachable(
            # NOTE ON RANK: the 1008 reference is a random projection at rank 32, while this
            # driver's control sits at the RRR's CV-selected rank (<= 32), so its elevation
            # will be lower. That is the gate working as intended, not a reachability defect:
            # if a random rank-r subspace cannot beat the trivial predictor, C2's comparator
            # genuinely cannot move and the run must refuse rather than read a low D_comm as
            # orientation. The reference establishes the gate is clearable in this family.
            anchor_name="randrank_control_supra_trivial",
            reference_cells=REF_1008_WS250_RANDPROJ_ELEVATION,
            score_fn=(lambda v: float(v) >= float(RANDRANK_CONTROL_MARGIN)),
            threshold=1.0, reference_source=REF_1008_SOURCE),
    ]

    env_kwargs = x734._env_kwargs_for_rung(RUNG)
    probe_env = x734._make_env(seeds[0], env_kwargs)
    action_dim = int(probe_env.action_dim)
    cfg = _config(dry_run, zworld_p0, p0, p1, steps, bc_eps, bc_rand, passes,
                  n_sens_states, n_sens_dirs)
    _ts = datetime.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

    seed_rows: List[Dict[str, Any]] = []
    arm_results: List[Dict[str, Any]] = []
    for seed in seeds:
        srow, arows = _run_seed(seed, action_dim, env_kwargs, cfg, zworld_p0, p0, p1, steps,
                                bc_eps, bc_rand, passes, n_sens_states, n_sens_dirs, dry_run)
        seed_rows.append(srow)
        arm_results.extend(arows)

    cells = ["seed%d" % r["seed"] for r in seed_rows]

    # ---- cross-seed subspace stability -----------------------------------------------------
    cross_seed: List[Dict[str, Any]] = []
    for i in range(len(seed_rows)):
        for j in range(i + 1, len(seed_rows)):
            pa = principal_angles(seed_rows[i]["basis_std"], seed_rows[j]["basis_std"])
            cross_seed.append({"pair": "seed%d|seed%d" % (seed_rows[i]["seed"],
                                                          seed_rows[j]["seed"]),
                               "mean_squared_cosine_overlap":
                                   float(pa["mean_squared_cosine_overlap"]),
                               "k1": pa["k1"], "k2": pa["k2"]})
    cross_seed_overlaps = [c["mean_squared_cosine_overlap"] for c in cross_seed]
    cross_stratum_overlaps = [float(r["cross_stratum_subspace"]["mean_squared_cosine_overlap"])
                              for r in seed_rows]
    for r in seed_rows:
        r.pop("basis_std", None)   # tensors never reach the manifest

    # ---- PRE-REGISTERED PRECONDITIONS ------------------------------------------------------
    full_ag = [r["agreements"][ARM_FULL] for r in seed_rows]
    elevations = [float(r["agreements"][ARM_FULL]
                        - (r["trivial_predictors"]["strongest_trivial_agreement"] or 0.0))
                  for r in seed_rows]
    rand_over_trivial = [float(r["agreements"][ARM_RAND]
                               - (r["trivial_predictors"]["strongest_trivial_agreement"] or 0.0))
                         for r in seed_rows]
    rrr_r2 = [r["rrr_heldout_r2"] for r in seed_rows]
    held = [float(r["heldout_steps"]) for r in seed_rows]
    # KEY NAME VERIFIED against zworld_encoder_guard.latent_stack_weight_delta's own return
    # dict: it emits `world_encoder_max_abs_delta`. A wrong key here would read 0.0 and fail
    # the precondition on every run for a typo rather than a finding.
    zdelta = [float((r["zworld_weight_delta"] or {}).get("world_encoder_max_abs_delta", 0.0)
                    or 0.0) for r in seed_rows]
    prat = [float(r["zworld_participation_ratio"]) for r in seed_rows]

    w_full, c_full = _worst(full_ag, cells)
    w_elev, c_elev = _worst(elevations, cells)
    w_rand, c_rand = _worst(rand_over_trivial, cells)
    w_r2, c_r2 = _worst(rrr_r2, cells)
    w_held, c_held = _worst(held, cells)
    w_zd, c_zd = _worst(zdelta, cells)
    w_pr, c_pr = _worst(prat, cells)

    checks = [
        {"name": "source_adequacy_ws250_full", "measured": w_full,
         "threshold": float(AGREEMENT_BAR), "direction": "lower", "offending_cell": c_full,
         "control": "POSITIVE CONTROL and the source-adequacy gate EXP-1403 requires: the "
                    "capacity-matched decoder must reproduce the oracle from the FULL 250-dim "
                    "world_state, the tensor the encoder itself reads. Below this floor the "
                    "target is not in the sender and no routing question is askable "
                    "(V3-EXQ-1008 measured 0.940 here). Worst seed."},
        {"name": "source_elevation_over_strongest_trivial", "measured": w_elev,
         "threshold": float(AGREEMENT_ELEVATION_MIN), "direction": "lower",
         "offending_cell": c_elev,
         "control": "The full-sender decode must beat the strongest TRIVIAL predictor "
                    "(previous executed action, 0.568-0.582 in this lineage) by the margin "
                    "1002 pre-registered, else the adequacy above is a trivial-predictor "
                    "artefact. Worst seed."},
        {"name": "rrr_heldout_r2_supra_floor", "measured": w_r2,
         "threshold": float(RRR_R2_FLOOR), "direction": "lower", "offending_cell": c_r2,
         "control": "The RRR must actually predict the receiver input on held-out episodes. "
                    "Below this floor 'the estimated communication subspace' is not a subspace "
                    "of anything and every arm built from it is noise. Worst seed. SELF-"
                    "ANCHORING and so not subject to the reachability failure mode: the "
                    "measured value is the instrument's own BEST cross-validated score over "
                    "the whole rank ladder (`max heldout_r2_by_rank`), not a hand-written "
                    "predicate that could be narrower than the state it anchors to. NOTE: this "
                    "is NOT comparable to `variance_routing.heldout_r2_from_*`, which this "
                    "driver computes with PER-COLUMN means while interface_probe._r2 "
                    "normalises against a single GLOBAL mean -- the two differ by z_world's "
                    "between-column variance and must never be compared to each other."},
        {"name": "randrank_control_supra_trivial", "measured": w_rand,
         "threshold": float(RANDRANK_CONTROL_MARGIN), "direction": "lower",
         "offending_cell": c_rand,
         "control": "READINESS ASSERT FOR THE LOAD-BEARING CRITERION, reporting the SAME "
                    "statistic C2 routes on (held-out top-1 agreement of a matched-rank "
                    "subspace decode). If a RANDOM rank-r projection is itself at the trivial "
                    "predictor, C2's comparator cannot move and a low D_comm would carry no "
                    "information about orientation. Worst seed."},
        {"name": "heldout_steps_sufficient", "measured": w_held,
         "threshold": float(HELDOUT_MIN_STEPS), "direction": "lower", "offending_cell": c_held,
         "control": "1002's own held-out sample floor. Worst seed."},
        {"name": "zworld_encoder_trained_in_p0", "measured": w_zd,
         "threshold": float(ZWORLD_DELTA_FLOOR), "direction": "lower", "offending_cell": c_zd,
         "control": "The receiver side must be a TRAINED encoder. At zero weight delta "
                    "z_world is a frozen random projection and the communication subspace "
                    "describes an untrained map. Worst seed."},
        {"name": "zworld_not_collapsed", "measured": w_pr,
         "threshold": float(PARTICIPATION_RATIO_FLOOR), "direction": "lower",
         "offending_cell": c_pr,
         "control": "x808's participation-ratio floor: a collapsed z_world makes any RRR onto "
                    "it degenerate. Worst seed."},
    ]

    manifest: Dict[str, Any] = {
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "run_id": "%s_%s_v3" % (EXPERIMENT_TYPE, _ts),
        "architecture_epoch": ARCHITECTURE_EPOCH,
        "timestamp_utc": _ts,
        "config": cfg,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "claim_ids": list(CLAIM_IDS),
        "bears_on": list(BEARS_ON),
        "sleep_driver_pattern": "none",
        "rung_id": RUNG_ID,
        "level_id": LEVEL_ID,
        "dry_run": bool(dry_run),
        "per_seed_results": seed_rows,
        "arm_results": arm_results,
        "subspace_stability": {
            "premise_note": "The premise is the WITHIN-ENCODER cross-STRATUM test only, and "
                            "counts only where the two strata's own data spans overlap "
                            "(>= SPAN_OVERLAP_MIN): a ridge RRR basis lies in the row space of "
                            "the visited data, so strata visiting different regions yield "
                            "different bases for an identical encoder. CROSS-SEED overlap is a "
                            "DIAGNOSTIC and is not a premise input: each seed trains its own "
                            "encoder, so a cross-seed disagreement is encoder "
                            "non-identifiability across training replicates, which is not the "
                            "falsifier MECH-537 registers. Both are reported against the "
                            "measured chance level for this rank and live dimension count.",
            "chance_level_per_seed": [
                {"seed": r["seed"], "chance_overlap": float(r["subspace_overlap_chance_level"]),
                 "selected_rank": int(r["selected_rank"])} for r in seed_rows],
            "cross_seed": cross_seed,
            "cross_seed_min_overlap": (min(cross_seed_overlaps) if cross_seed_overlaps
                                       else float("nan")),
            "cross_stratum_per_seed": [
                {"seed": r["seed"],
                 "mean_squared_cosine_overlap":
                     float(r["cross_stratum_subspace"]["mean_squared_cosine_overlap"])}
                for r in seed_rows],
            "cross_stratum_min_overlap": (min(cross_stratum_overlaps)
                                          if cross_stratum_overlaps else float("nan")),
            "stratum_definition": "oracle-driven visitation vs random-driven visitation, the "
                                  "two episode families x1002._collect_episodes already "
                                  "produces; the RRR is re-fitted independently on each.",
        },
    }

    try:
        preconditions = p0_readiness_gate(checks)
        gate_green = True
        gate_payload = preconditions
    except P0NotReady as e:
        gate_green = False
        gate_payload = e.preconditions

    # ---- CONTRASTS + CRITERIA ---------------------------------------------------------------
    d_full_comm = _paired_positive([r["delta_full_minus_comm"] for r in seed_rows],
                                   ROUTING_DROP_MIN)
    d_rand_comm = _paired_positive([r["delta_randrank_minus_comm"] for r in seed_rows],
                                   ORIENTATION_MARGIN)
    perp_losses = [float(r["delta_full_minus_perp"]) for r in seed_rows]
    n_perp_retains = sum(1 for v in perp_losses
                         if np.isfinite(v) and v <= COMPLEMENT_RETENTION_TOL)
    # C4 scores the DECISION-TARGETED probe (F1). The generic whole-subspace probe is kept as
    # a recorded diagnostic and is deliberately NOT a criterion.
    sens_ratios = [float(r["sensitivity_decision_targeted"]["sensitivity_ratio"])
                   for r in seed_rows]
    null_ratios = [float(r["sensitivity_decision_targeted"]["null_sensitivity_ratio"])
                   for r in seed_rows]
    generic_ratios = [float(r["sensitivity_generic_diagnostic"]["sensitivity_ratio"])
                      for r in seed_rows]
    null_margins = [float(n - m) for n, m in zip(null_ratios, sens_ratios)]
    d_null_margin = _paired_positive(null_margins, INSENSITIVITY_NULL_MARGIN)
    n_insensitive = sum(1 for v in sens_ratios
                        if np.isfinite(v) and v <= INSENSITIVITY_RATIO_MAX)

    n_equivalent = sum(1 for r in seed_rows
                       if abs(float(r["delta_full_minus_comm"])) <= EQUIVALENCE_BAND)

    # CROSS-SEED OVERLAP IS A DIAGNOSTIC, NOT A PREMISE INPUT. Each seed warms up its OWN
    # agent (`_run_seed` builds a fresh one and trains it), so the per-seed bases belong to
    # THREE DIFFERENT TRAINED ENCODERS. A disagreement between them is encoder
    # non-identifiability across training replicates -- a real and interesting fact, but not
    # MECH-537's registered falsifier, which is about instability "across frame or
    # receiver-state strata" WITHIN a receiver. Counting it against the premise would let a run
    # whose every seed individually shows the full C1..C4 routing signature emit
    # `non_contributory / premise fails` for a reason no reader could recover from the label.
    # It is recorded, alongside the measured chance level, and left to the reader.
    cross_seed_min = float(min([v for v in cross_seed_overlaps if np.isfinite(v)])
                           if any(np.isfinite(v) for v in cross_seed_overlaps)
                           else float("nan"))
    # Cross-STRATUM instability counts only where the two strata's own data spans overlap
    # (F7): a ridge RRR basis lies in the row space of the visited data, so strata that visit
    # different regions produce different bases for an identical encoder, and reading that as
    # receiver-state dependence would be a mis-attribution.
    span_overlaps = [float(r["cross_stratum_data_span_overlap"]) for r in seed_rows]
    informative_stratum = [ov for ov, sp in zip(cross_stratum_overlaps, span_overlaps)
                           if np.isfinite(ov) and np.isfinite(sp)
                           and sp >= SPAN_OVERLAP_MIN]
    cross_stratum_min_informative = (float(min(informative_stratum)) if informative_stratum
                                     else float("nan"))
    # The premise is the WITHIN-ENCODER cross-stratum test, and only that -- exactly the
    # falsifier the claim registers. Scored against the MEASURED CHANCE LEVEL for two
    # independent rank-r subspaces in this sender's live dimensions rather than an absolute
    # bar: chance overlap falls as 1/rank-ish (r=32 -> ~0.24, r=8 -> ~0.06 at 136 live dims),
    # so a fixed absolute threshold silently changes meaning with the CV-selected rank.
    chance_overlaps = [float(r["subspace_overlap_chance_level"]) for r in seed_rows]
    chance_mean = _mean(chance_overlaps)
    stability_required = float(chance_mean + STABILITY_MARGIN_OVER_CHANCE)
    stability_min = cross_stratum_min_informative
    premise_ok = bool((not np.isfinite(stability_min))
                      or stability_min >= stability_required)
    c1 = bool(d_full_comm["passed"])
    c2 = bool(d_rand_comm["passed"])
    c3 = bool(n_perp_retains >= SEED_MAJORITY)
    # C4 is the ABSOLUTE, un-normalised ratio and nothing else. The null-margin leg is
    # RECORDED but is NOT a conjunct: once the component weights are inside the measurement
    # (which is what makes the ratio a statement about the consumer's actual response to a
    # change in a decision coordinate), the random-subspace null is no longer matched on
    # RETENTION -- a fitted subspace that retains the decision coordinates better than chance
    # deflates the measured ratio for a retention reason, not a coupling one. A confounded
    # conjunct in the confirm path would be one fewer independent leg than the verdict text
    # implies, so it is reported instead of scored.
    c4 = bool(n_insensitive >= SEED_MAJORITY)
    equivalent = bool(n_equivalent >= SEED_MAJORITY)

    criteria = [
        {"name": "C1_target_drops_inside_comm_subspace", "load_bearing": False,
         "rank_confounded": True,
         "passed": c1, "measured": d_full_comm["mean"], "threshold": float(ROUTING_DROP_MIN),
         "sd": d_full_comm["sd"], "n_seeds": d_full_comm["n_seeds_clearing"],
         "seeds_required": int(SEED_MAJORITY), "per_seed": d_full_comm["per_seed"],
         "detail": "D_full - D_comm, paired per seed."},
        {"name": "C2_drop_is_orientation_not_rank", "load_bearing": True,
         "passed": c2, "measured": d_rand_comm["mean"], "threshold": float(ORIENTATION_MARGIN),
         "sd": d_rand_comm["sd"], "n_seeds": d_rand_comm["n_seeds_clearing"],
         "seeds_required": int(SEED_MAJORITY), "per_seed": d_rand_comm["per_seed"],
         "detail": "D_randrank - D_comm at the SAME rank. THE load-bearing criterion: a "
                   "rank-r projection loses information whatever its orientation, so only "
                   "decoding WORSE than a random subspace of equal rank shows the "
                   "communication subspace is oriented away from the decision directions."},
        {"name": "C3_complement_retains_the_decodability", "load_bearing": False,
         "expected_to_pass_by_dimensionality": True,
         "passed": c3, "measured": _mean(perp_losses),
         "threshold": float(COMPLEMENT_RETENTION_TOL), "direction": "upper",
         "n_seeds": int(n_perp_retains), "seeds_required": int(SEED_MAJORITY),
         "per_seed": perp_losses,
         "detail": "D_full - D_perp, an UPPER bound: deleting the communication subspace from "
                   "the sender must cost essentially nothing, which is the claim's own 'no "
                   "information having been destroyed'. Stated against the FULL sender rather "
                   "than against D_comm because the complement's rank is (live - r) against "
                   "the subspace's r, so a perp-minus-comm difference would be confounded by "
                   "dimensionality exactly as C1 is. NOT load-bearing, and deliberately so: a "
                   "rank-(live - r) complement retains each decision coordinate at norm "
                   ">= sqrt(1 - retention_comm^2) whatever its orientation, so a PASS here is "
                   "close to automatic and carries no orientation information. It is included "
                   "because its FAILURE would be informative -- it would mean deleting the "
                   "communication subspace destroys decodability, which is not a routing story "
                   "at all. C2 is the only rank-matched contrast and the only one that "
                   "licenses an orientation reading."},
        {"name": "C4a_decision_complement_coupling_below_matched_null",
         "load_bearing": False, "scored_as_conjunct": False, "retention_confounded": True,
         "passed": bool(d_null_margin["passed"]), "measured": d_null_margin["mean"],
         "threshold": float(INSENSITIVITY_NULL_MARGIN), "sd": d_null_margin["sd"],
         "n_seeds": d_null_margin["n_seeds_clearing"], "seeds_required": int(SEED_MAJORITY),
         "per_seed": d_null_margin["per_seed"],
         "detail": "null_ratio - measured_ratio, paired per seed, on the DECISION-TARGETED "
                   "probe: for each of the five coordinates the oracle actually reads, e_j is "
                   "split into its communication-subspace and complement components (in the "
                   "standardised basis, the same decomposition the decode arms use), each "
                   "component is mapped to raw sender space, normalised, and applied at "
                   "matched eps, and the ratio is mean||dz_world|| complement-component over "
                   "comm-component. A GENERIC complement direction puts only ~1.5% of its "
                   "energy on those five coordinates, so a generic ratio would measure whether "
                   "the encoder's Jacobian is isotropic off the subspace -- not whether the "
                   "decision content reaches the receiver -- which is why C4 scores the aimed "
                   "probe and the generic one is recorded as a diagnostic only. The null "
                   "repeats the identical construction on a RANDOM subspace of the same rank "
                   "AND the same live-dimension support, so the gate is calibrated on this "
                   "encoder by this machinery and cannot be unmeetable by construction."},
        {"name": "C4b_decision_complement_coupling_below_absolute_ceiling",
         "load_bearing": True, "scored_as_conjunct": True,
         "passed": bool(n_insensitive >= SEED_MAJORITY), "measured": _mean(sens_ratios),
         "threshold": float(INSENSITIVITY_RATIO_MAX), "direction": "upper",
         "n_seeds": int(n_insensitive), "seeds_required": int(SEED_MAJORITY),
         "per_seed": sens_ratios, "null_control_per_seed": null_ratios,
         "detail": "THE C4 CONJUNCT. Ratio of mean||dz_world|| along the COMPLEMENT COMPONENT "
                   "of each oracle decision coordinate to that along its COMMUNICATION-SUBSPACE "
                   "COMPONENT, components taken UN-NORMALISED so their weights in e_j stay "
                   "inside the measurement -- it is therefore the share of the consumer's "
                   "ACTUAL response to a change in that coordinate that arrives via the "
                   "complement rather than via the subspace, which is exactly what the claim "
                   "asserts is small. Control-free and reachable BY HYPOTHESIS rather than by "
                   "assumption: under a genuine routing failure the encoder Jacobian is "
                   "supported on the communication subspace, the complement term goes to ~0 "
                   "and the ratio to ~0; under no routing failure the ratio is ~b/a > 1. The "
                   "dry-run smoke measured 6.46 at that no-routing end, so the statistic "
                   "demonstrably has the dynamic range this criterion needs."},
        {"name": "C5_single_subspace_premise_holds", "load_bearing": True,
         "passed": premise_ok, "measured": stability_min,
         "threshold": stability_required, "direction": "lower",
         "chance_level_measured": chance_mean,
         "margin_over_chance_required": float(STABILITY_MARGIN_OVER_CHANCE),
         "detail": "min mean_squared_cosine_overlap between the ORACLE-driven and "
                   "RANDOM-driven basis WITHIN each seed's own encoder, over the seeds whose "
                   "two strata's data spans overlap. Scored against the MEASURED chance "
                   "overlap for two independent rank-r subspaces in this sender's live "
                   "dimensions, because chance overlap scales with rank and an absolute bar "
                   "would mean something different at every CV-selected rank. CROSS-SEED "
                   "overlap is deliberately NOT an input: each seed trains its OWN encoder, so "
                   "a cross-seed disagreement is encoder non-identifiability across training "
                   "replicates, not the receiver-state-strata instability this claim "
                   "registers; it is recorded as a diagnostic instead. Its FAILURE routes to "
                   "MECH-547 / MECH-555 and is not a MECH-537 conjunct."},
    ]

    outcome, label, direction, hypothesis_verdict = _adjudicate(
        premise_ok, c1, c2, c3, c4, equivalent, _c2_falsified(d_rand_comm["per_seed"]))
    if not gate_green:
        outcome = "FAIL"
        label = "substrate_not_ready_requeue"
        direction = "non_contributory"
        hypothesis_verdict = ("A pre-registered readiness precondition is unmet, so no "
                              "scientific leg is adjudicated. This is a refusal with a "
                              "record, not a verdict about MECH-537.")

    degeneracy = check_degeneracy({
        "C2_drop_is_orientation_not_rank": d_rand_comm["per_seed"],
        "C3_complement_retains_the_decodability": perp_losses,
        "C4_receiver_insensitive_to_decision_complement":
            [v for v in sens_ratios if np.isfinite(v)],
        "arm_agreement_spread": {
            "groups": [[float(r["agreements"][a]) for a in ARM_IDS] for r in seed_rows]},
    })

    def _flat(v: Any) -> Any:
        try:
            f = float(v)
        except (TypeError, ValueError):
            return None
        return f if np.isfinite(f) else None

    readout: Dict[str, Any] = {}
    for k, v in (
        ("d_full_mean", _mean(full_ag)),
        ("d_comm_mean", _mean([r["agreements"][ARM_COMM] for r in seed_rows])),
        ("d_perp_mean", _mean([r["agreements"][ARM_PERP] for r in seed_rows])),
        ("d_randrank_mean", _mean([r["agreements"][ARM_RAND] for r in seed_rows])),
        ("delta_full_minus_comm_mean", d_full_comm["mean"]),
        ("delta_randrank_minus_comm_mean", d_rand_comm["mean"]),
        ("delta_perp_minus_comm_mean",
         _mean([r["delta_perp_minus_comm"] for r in seed_rows])),
        ("delta_full_minus_perp_mean", _mean(perp_losses)),
        ("decision_sensitivity_ratio_mean", _mean(sens_ratios)),
        ("decision_null_sensitivity_ratio_mean", _mean(null_ratios)),
        ("generic_sensitivity_ratio_mean", _mean(generic_ratios)),
        ("cross_seed_stability_min", cross_seed_min),
        ("sensitivity_null_margin_mean", d_null_margin["mean"]),
        ("cross_stratum_stability_min", stability_min),
        ("subspace_overlap_chance_level_mean", chance_mean),
        ("stability_required", stability_required),
        ("rrr_heldout_r2_worst_seed", w_r2),
        ("selected_rank_mean", _mean([float(r["selected_rank"]) for r in seed_rows])),
        ("n_seeds_rank_at_ladder_ceiling",
         sum(1 for r in seed_rows if r["rrr_rank_at_ladder_ceiling"])),
        ("source_adequacy_worst_seed", w_full),
        ("decision_retention_comm_max_mean",
         _mean([r["decision_coordinate_retention"]["comm_subspace"]["max"] for r in seed_rows])),
        ("decision_retention_randrank_max_mean",
         _mean([r["decision_coordinate_retention"]["randrank_control"]["max"]
                for r in seed_rows])),
        ("c1_passed", int(c1)), ("c2_passed", int(c2)), ("c3_passed", int(c3)),
        ("c4_passed", int(c4)), ("c5_premise_holds", int(premise_ok)),
        ("gate_green", int(gate_green)),
    ):
        fv = _flat(v)
        if fv is not None:
            readout[k] = fv

    manifest.update({
        "outcome": outcome,
        "evidence_direction": direction,
        "hypothesis_verdict": hypothesis_verdict,
        "criteria": criteria,
        "combination_rule": (
            "PREMISE FIRST: not C5 (WITHIN-ENCODER cross-stratum subspace stability against "
            "the MEASURED chance level -- the claim's own registered falsifier -- counted only "
            "where the two strata's data spans overlap >= %.2f; CROSS-SEED overlap is a "
            "diagnostic and NOT a premise input, because each seed trains its own encoder) -> "
            "single_subspace_premise_fails (non_contributory, routes MECH-547/MECH-555). Else "
            "FALSIFY iff |D_full - D_comm| <= %.2f on >= %d seeds (weakens). Else CONFIRM iff "
            "C1 AND C2 AND C3 AND C4 (supports), where C4 is C4b alone -- C4a is recorded but "
            "retention-confounded and not scored. Else FALSIFY iff C1 AND the C2 contrast is "
            "genuinely NON-POSITIVE (mean <= 0 and a seed majority <= 0, never merely `not "
            "C2`) (weakens: the drop is rank, not orientation). Else undetermined (mixed). C2 "
            "is the only RANK-MATCHED contrast and the only one that licenses an ORIENTATION "
            "reading. A red readiness gate overrides everything with "
            "substrate_not_ready_requeue."
            % (SPAN_OVERLAP_MIN, EQUIVALENCE_BAND, SEED_MAJORITY)),
        "contrasts": {"full_minus_comm": d_full_comm,
                      "randrank_minus_comm": d_rand_comm,
                      "full_minus_perp_per_seed": perp_losses,
                      "n_seeds_complement_retains": int(n_perp_retains),
                      "n_seeds_equivalent_within_band": int(n_equivalent),
                      "complement_coupling_null_margin": d_null_margin,
                      "n_seeds_insensitive": int(n_insensitive),
                      "generic_sensitivity_ratio_per_seed": generic_ratios,
                      "cross_seed_stability_min": cross_seed_min,
                      "cross_stratum_stability_min_informative": cross_stratum_min_informative,
                      "cross_stratum_data_span_overlap_per_seed": span_overlaps},
        "readout": readout,
        "pre_registered": {
            "agreement_bar": float(AGREEMENT_BAR),
            "agreement_elevation_min": float(AGREEMENT_ELEVATION_MIN),
            "rrr_r2_floor": float(RRR_R2_FLOOR),
            "randrank_control_margin": float(RANDRANK_CONTROL_MARGIN),
            "routing_drop_min": float(ROUTING_DROP_MIN),
            "orientation_margin": float(ORIENTATION_MARGIN),
            "complement_retention_tol": float(COMPLEMENT_RETENTION_TOL),
            "equivalence_band": float(EQUIVALENCE_BAND),
            "insensitivity_null_margin": float(INSENSITIVITY_NULL_MARGIN),
            "insensitivity_ratio_max": float(INSENSITIVITY_RATIO_MAX),
            "span_overlap_min": float(SPAN_OVERLAP_MIN),
            "stability_margin_over_chance": float(STABILITY_MARGIN_OVER_CHANCE),
            "delta_sd_multiple": float(DELTA_SD_MULTIPLE),
            "seed_majority": int(SEED_MAJORITY),
            "heldout_min_steps": int(HELDOUT_MIN_STEPS),
        },
        "interpretation": {
            "label": label,
            "preconditions": gate_payload,
            "criteria_non_degenerate": {
                # A contrast criterion DISCRIMINATED only if the arms it compares actually
                # produced different agreements. All-zero deltas mean the projections changed
                # nothing (e.g. a rank that made P_comm the identity, or a decoder collapsed
                # to one class on every arm) -- that is a vacuous pass, not a null result.
                "C1_target_drops_inside_comm_subspace": _contrast_discriminated(
                    d_full_comm["per_seed"]),
                "C2_drop_is_orientation_not_rank": _contrast_discriminated(
                    d_rand_comm["per_seed"]),
                "C3_complement_retains_the_decodability": _contrast_discriminated(perp_losses),
                # The ratio is meaningless if the COMM perturbations themselves moved z_world
                # by nothing -- then the denominator is noise and so is the ratio.
                "C4_receiver_insensitive_to_decision_complement": bool(
                    [v for v in sens_ratios if np.isfinite(v)]
                    and all(float(r["sensitivity_decision_targeted"]["by_eps"][
                        "%.3f" % SENSITIVITY_EPS_FRACS[-1]]["measured"]["mean_dz_inside"])
                        > 1e-9 for r in seed_rows)),
                "C5_single_subspace_premise_holds": bool(np.isfinite(stability_min)),
            },
            "gate_green": bool(gate_green),
        },
        "guards": {"gate_green": bool(gate_green),
                   "anchor_reachability": anchor_reachability},
    })
    manifest.update(degeneracy)
    manifest["_elapsed_seconds_measured"] = float(time.perf_counter() - t0)
    manifest["_started_at"] = t0
    return manifest


# ==========================================================================================
def _run_self_test() -> int:
    """Exercise the verdict grid and the geometry helpers without a multi-hour run."""
    fails: List[str] = []

    def chk(name: str, cond: bool) -> None:
        print("  [selftest] %-52s %s" % (name, "ok" if cond else "FAIL"), flush=True)
        if not cond:
            fails.append(name)

    o, lab, d, _h = _adjudicate(True, True, True, True, True, False, False)
    chk("confirm -> supports", (o, d) == ("PASS", "supports")
        and lab == "communication_subspace_routing_failure_confirmed")
    o, lab, d, _h = _adjudicate(True, False, False, False, False, True, False)
    chk("equivalent -> weakens", (o, d) == ("PASS", "weakens"))
    o, lab, d, _h = _adjudicate(False, True, True, True, True, False, False)
    chk("unstable subspace -> non_contributory", (o, d) == ("PASS", "non_contributory"))
    o, lab, d, _h = _adjudicate(True, True, True, True, True, False, True)
    chk("an inconsistent (c2 and c2_falsified) caller still cannot print a falsification",
        lab == "communication_subspace_routing_failure_confirmed")
    o, lab, d, _h = _adjudicate(True, True, False, True, True, False, True)
    chk("C1 but not C2, discriminated -> weakens (rank, not orientation)",
        (o, d) == ("PASS", "weakens")
        and lab == "routing_drop_explained_by_rank_not_orientation")
    o, lab, d, _h = _adjudicate(True, True, False, True, True, False, False)
    chk("C1, C2 fails but is NOT falsified -> undetermined, never a falsification",
        (o, d) == ("FAIL", "mixed"))
    # The exact shape the second red-team pass found: positive on every seed, but failing the
    # 2*SD consistency clause. It must NOT read as "rank explains it".
    noisy = [0.10, 0.30, 0.02]
    chk("noisy-but-positive C2 contrast is not `passed`",
        not _paired_positive(noisy, ORIENTATION_MARGIN)["passed"])
    chk("noisy-but-positive C2 contrast is NOT falsified either", not _c2_falsified(noisy))
    o, lab, d, _h = _adjudicate(True, True, False, True, True, False,
                                _c2_falsified(noisy))
    chk("...so it routes to undetermined, not to a rank-explains-it weakens",
        lab == "routing_signature_incomplete_undetermined")
    chk("genuinely non-positive C2 contrast IS falsified",
        _c2_falsified([-0.04, 0.0, -0.11]))
    o, lab, d, _h = _adjudicate(True, True, True, True, False, False, False)
    chk("C4 false -> undetermined, never premise-fail and never supports",
        lab == "routing_signature_incomplete_undetermined")
    # c2 and c2_falsified are mutually exclusive on real data; the grid is exercised with
    # CONSISTENT pairs only, and the branch itself additionally requires `not c2`.
    labels = {_adjudicate(*a)[2] for a in (
        (True, True, True, True, True, False, False),    # confirm
        (True, False, False, False, False, True, False),  # equivalent
        (False, True, True, True, True, False, False),    # premise fails
        (True, True, False, True, True, False, True),     # rank, not orientation
        (True, True, True, True, False, False, False))}   # undetermined
    chk("grid reaches supports AND weakens AND non_contributory AND mixed",
        labels == {"supports", "weakens", "non_contributory", "mixed"})

    pp = _paired_positive([0.20, 0.21, 0.19], 0.05)
    chk("paired_positive: consistent large deltas pass", pp["passed"])
    pp = _paired_positive([0.20, -0.18, 0.21], 0.05)
    chk("paired_positive: seed-disagreeing deltas fail", not pp["passed"])
    pp = _paired_positive([0.02, 0.02, 0.02], 0.05)
    chk("paired_positive: below the absolute floor fails", not pp["passed"])

    sub_b = _random_orthonormal(40, 5, seed=3)
    keep_idx = torch.arange(0, 200, 5)[:40]
    emb = _embed_basis(sub_b, keep_idx, 250)
    chk("embed_basis preserves orthonormality",
        bool(torch.allclose(emb.T @ emb, torch.eye(5), atol=1e-4)))
    chk("embed_basis is zero off the kept dimensions",
        float(emb[[i for i in range(250) if i not in set(keep_idx.tolist())]].abs().max()) == 0.0)

    b = _random_orthonormal(250, 8, seed=1)
    chk("random_orthonormal is orthonormal",
        bool(torch.allclose(b.T @ b, torch.eye(8), atol=1e-4)))
    x = torch.randn(64, 250)
    p = _project(x, b)
    chk("projection is idempotent", bool(torch.allclose(_project(p, b), p, atol=1e-4)))
    chk("projection + complement reconstructs x",
        bool(torch.allclose(p + (x - p), x, atol=1e-5)))
    chk("projection and complement are orthogonal",
        float((p * (x - p)).sum(dim=1).abs().max()) < 1e-3)

    # The one runtime premise this driver's whole design rests on.
    env = x734._make_env(42, x734._env_kwargs_for_rung(RUNG))
    _flat_obs, obs = env.reset()
    ws = torch.as_tensor(obs["world_state"]).reshape(-1).float()
    rf = torch.as_tensor(obs["resource_field_view"]).reshape(-1).float()
    chk("world_state is 250-dim", int(ws.shape[0]) == WORLD_STATE_DIM)
    chk("resource_field_view == world_state[225:250]",
        bool(torch.allclose(ws[RESOURCE_FIELD_OFFSET:RESOURCE_FIELD_OFFSET + 25], rf)))
    chk("decision indices lie inside the resource field slice",
        all(RESOURCE_FIELD_OFFSET <= j < RESOURCE_FIELD_OFFSET + 25 for j in DECISION_INDICES))

    print("SELF-TEST: %s" % ("PASS" if not fails else ("FAIL -- " + ", ".join(fails))),
          flush=True)
    return 0 if not fails else 1


def _parse_args():
    ap = argparse.ArgumentParser(description=EXPERIMENT_TYPE)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    return ap.parse_args()


def main():
    """Run and write the manifest. Returns (outcome, manifest_path) for the __main__ block,
    which is where `emit_outcome` must be CALLED (validate_experiments asserts that literally,
    and the runner reads the sentinel it writes)."""
    args = _parse_args()

    if args.self_test:
        sys.exit(_run_self_test())

    seeds = args.seeds if args.seeds else (DRY_RUN_SEEDS if args.dry_run else SEEDS)
    result = run_experiment(list(seeds), dry_run=bool(args.dry_run))
    started_at = result.pop("_started_at", None)
    result.pop("_elapsed_seconds_measured", None)

    out_path = write_flat_manifest(
        result,
        dry_run=bool(args.dry_run),
        config=result["config"],
        seeds=list(seeds),
        script_path=Path(__file__),
        started_at=started_at,
        z_goal_stream_stats=_ZG.stats(),
    )
    print("manifest: %s" % out_path, flush=True)
    print("outcome: %s  label: %s  direction: %s"
          % (result["outcome"], result["interpretation"]["label"],
             result["evidence_direction"]), flush=True)

    _outcome_raw = str(result["outcome"]).upper()
    return (_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
            out_path, bool(args.dry_run))


if __name__ == "__main__":
    _outcome, _out_path, _dry = main()
    emit_outcome(outcome=_outcome, manifest_path=_out_path, dry_run=_dry)
