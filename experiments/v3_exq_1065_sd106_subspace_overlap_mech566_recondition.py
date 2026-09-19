"""V3-EXQ-1065 -- SD-106 ENCODER-PLANE SUBSPACE-OVERLAP PROBE + MECH-566 RECONDITIONING
FALSIFIER, on ONE shared SD-106 P0a warmup at epochs=40 (user decision 2026-09-19: one run,
shared warm-up, the two halves kept separately pre-registered).

EXPERIMENT_PURPOSE = "diagnostic". Two halves, each with its own pre-registered criterion,
its own claim, and its own per-claim evidence direction. Neither half re-measures the SD-106
acceptance target (V3-EXQ-1023a already did, FAIL, ratified 2026-09-17) and this run opens no
leg on the frozen question `zworld_actor_adequacy_locus` (GOV-FROZEN-1 / GFLAG-0312).

SLEEP DRIVER: not applicable -- no sleep flag is set. Recorded as sleep_driver_pattern="none".

=== HALF A: THE ENCODER-PLANE PROBE (claim SD-106; chip-20260919-sd106-subspace-overlap-...) ===

User instruction (AskUserQuestion 2026-09-19T00:49:12Z, option C): "principal angles /
subspace overlap between the SD-106 code and PCA-32 on the SAME P0a buffer, reported beside the
existing post-hoc R^2". V3-EXQ-1023a closed the (a) transfer-amplification / (b) which-directions
discrimination in favour of (b) BY ELIMINATION; this half measures (b) DIRECTLY.

  MANIPULATION: none on this half -- it is a measurement on the SD-106@epochs40 code that
                V3-EXQ-1041 executed six times and V3-EXQ-1023a ran its acceptance on.
  INSTRUMENT:   the P0a rollout buffer is CAPTURED in situ (x1041._capturing_target -- a
                replay does NOT reproduce it, measured 13 percent off), the trainer's own
                train/holdout split is reproduced (x1041._reproduce_split), the code is read
                through the substrate's own world path (x1041._z_world -> _z_world_path) and
                the post-hoc OLS R^2 is x1041._ols_holdout_r2 unchanged.
  THE NEW READOUT: principal angles (experiments/_lib/interface_probe.principal_angles, the
                existing harness -- not a second implementation) between two 32-dim subspaces
                of the 250-dim world_state space:
                  U_code = column space of the OLS decoder code -> world_obs (the directions
                           the code RECONSTRUCTS INTO; "which directions are preserved"),
                  V_pca  = the top-32 principal directions of the same train split.
                Scalar summary = mean squared cosine of the principal angles (1.0 identical,
                32/250 = 0.128 for a random 32-subspace).
  FLOOR / CEILING ON THE SAME RUN (the chip asked for both), scored OUT OF SAMPLE against
                ONE reference: PCA-32 fitted on the HOLDOUT split of the same buffer. The code
                basis and the train-split PCA-32 are both fitted on the train split and both
                scored against that holdout reference, so the CEILING (train PCA-32 vs holdout
                PCA-32) and the code face the same sampling noise; overlap_rel = code / ceiling.
                floor = a random orthonormal 32-subspace vs the holdout PCA-32 (analytic
                32/250 = 0.128) and the SD-106-OFF (shipped epochs=12, preservation_weight 0.0)
                code as the negative control. The in-sample form (code vs train PCA-32) is
                recorded as context only -- the smoke showed it EXCEEDING the split ceiling,
                which is why the out-of-sample form is the pre-registered one.
  NON-DEGENERACY CONTROL FOR P1: an UNTRAINED (random-init) SD-106 world path is read through
                the identical probe on the identical buffer and split. An OLS decoder's range
                can align with the top principal directions for almost ANY code (the smoke
                showed the shipped-budget OFF code matching PCA-32 as well as SD-106 did), so
                a HIGH reading is informative ONLY if the untrained control reads below the
                HIGH band on a seed majority. P1's `criteria_non_degenerate` is keyed to that;
                if the control does not discriminate, the P1 class is recorded but flagged
                degenerate and the decision-cell R^2 supplement carries the question instead.
  DECISION-RELEVANT SUPPLEMENT (recorded, not routed on): per-coordinate holdout R^2 of the
                oracle's five decision cells (x1008.DECISION_WORLD_STATE_INDICES) and of the
                25-cell resource-field block, decoded linearly from the SD-106 code and from
                the PCA-32 coordinates -- the "which directions" question asked of the exact
                coordinates the oracle reads.

  THE P1 STATISTIC (user decision 2026-09-19 at /queue-experiment Step 4.5, replacing the
                mean-cos^2 overlap after the cross-model red-team's BLOCKING finding F1: on the
                dry buffer a RANDOM-INIT encoder already read 0.95 of the overlap ceiling, so
                that statistic could never return LOW). P1 routes on the ABSOLUTE
                PC-RECOVERABILITY of the SD-106 code:
                    rec(code) = unweighted mean over the 32 train-split principal directions of
                                the 0-clipped HOLDOUT R^2 with which the code linearly recovers
                                each direction's score (which principal directions the code
                                carries, the low-variance tail counting as much as PC1),
                against the EXACT ceiling 1.0 (PCA-32's own coordinates recover every score by
                construction). The second red-team pass (RT2-1/RT2-2) showed why there is NO
                control-anchored normalisation: world_obs is effectively 32-dimensional, so any
                32-dim LINEAR projection recovers the PC scores almost exactly and is not a
                content-free floor, and a floor that reads negative drags a normalised score
                toward HIGH. What rec measures is what the code's NONLINEARITY and precision
                gate discard, per direction. The in-run random-init encoder is the
                "can-it-read-LOW" guard (non-degeneracy), the random orthonormal projection is
                recorded as the "linear keeps everything" reference (~0.98 expected at full
                scale), principal angles are still REPORTED for every code (the chip's request),
                and the same readouts are computed for the OFF code and for the oracle's
                decision cells.

  PRE-REGISTERED READING TABLE (the user asked for this to be stated BEFORE the run):
    parity_ratio = posthoc_ols_r2 / pca32_ols_r2 on the same buffer (V3-EXQ-1041 measured
    0.980 at this budget).
      HIGH parity (>= PARITY_HIGH_FLOOR) and LOW recoverability (rec <= P1_LOW_CEIL, 0.75)
        -> `which_directions_confirmed_directly`: the retained directions are NOT the
           PCA-32 directions. Confirms directly what 1023a established by elimination.
           Routes: /failure-autopsy -> the SD-106 objective-choice decision (user-held;
           this run does NOT change the objective).
      HIGH parity and HIGH recoverability (rec >= P1_HIGH_FLOOR, 0.90)
        -> `retained_directions_are_pca32`: the genuinely surprising result. The consumer
           shortfall is then NOT a which-directions effect and the question 1023a closed by
           elimination is RE-OPENED. Routes, decided now: the shortfall lies at the consumer's
           access to a code that spans the right directions -- exactly MECH-566 (conditioning)
           / MECH-567 (target-carrying head) territory -- so Half B's verdict becomes the
           load-bearing reading and /failure-autopsy adjudicates the pair together.
      HIGH parity and rec between the two bands -> `recoverability_indeterminate`
           (recorded; no routing; a wider band is a design change, not a re-run).
      parity BELOW the floor -> `probe_premise_parity_unmet`: the premise this table rests on
           (0.98 of the PCA-32 ceiling, V3-EXQ-1041) did not reproduce; the table does not
           apply and Half A reports the reproduction failure instead of a reading. Half B is
           UNAFFECTED (the two halves are independent; red-team F3).
      the untrained encoder itself reads ABOVE P1_LOW_CEIL on a seed majority
        -> `probe_cannot_read_low`: the statistic cannot read LOW even for a code that carries
           nothing, so P1 is DEGENERATE (flagged in non_degenerate_per_claim, SD-106
           direction unknown) and the decision-cell readouts carry the which-directions
           question as recorded context.
    Each half is gated by ITS OWN readiness preconditions (red-team RT2-4): a Half-B instrument
    failure (raw-field floor, PCA anchor, invertibility, 1023a reproduction) voids MECH-566's
    direction only; a Half-A failure (buffer capture) voids SD-106's only; the shared SD-106
    warmup checks (bypass trained, encoder trained, PR floor, step floor) void both.
    HONESTY NOTE: every framing of V3-EXQ-1023a's OUTCOME above is POST-HOC (1023a has run and
    been adjudicated). Only THIS run's own criteria (P1, M1 below) are pre-registered.

=== HALF B: THE MECH-566 RECONDITIONING FALSIFIER (claim MECH-566; proposal EXP-0250) ===

MECH-566 (claims.yaml, candidate): the SD-106 preservation objective is EXACTLY invariant under
GL(32) acting on the code, so it cannot prefer a well-conditioned representative of its orbit;
the fixed-capacity consumer is NOT invariant, so transfer fails for a conditioning reason
rather than a content reason. Its registered falsifier, verbatim in substance: a FIXED
invertible linear reconditioning of the FROZEN code before the consumer -- which changes NO
information and leaves preservation R^2 EXACTLY unchanged -- raises held-out
oracle_action_agreement by >= 0.05 on a seed majority (CONFIRMING) or moves it < 0.02 under
every member of the family (FALSIFYING).

WHAT THE EXISTING RECORD ALREADY SAYS, SO THIS RUN IS NARROWER THAN THE CLAIM'S FALSIFIER:
  * per-component standardisation is ALREADY the baseline: every arm in the x1002/x1010/x1023
    lineage z-scores on the train split (x1002.STANDARDISE_FEATURES = True), so the 0.7273
    consumer-rung agreement V3-EXQ-1023a recorded was measured WITH the diagonal member of the
    family applied. The diagonal member is therefore already exonerated as the fix, and only
    the full-covariance members (ZCA, within-class/LDA whitening) remain untested on the
    SD-106 code.
  * on the SD-106-OFF code, V3-EXQ-1008 measured ZCA at +0.040 / +0.030 / +0.048 over diag
    (seeds 42/43/44) -- inside MECH-566's own grey zone (0.02..0.05) -- and LDA-whitening at
    +0.042 / +0.033 / +0.052. That is prior evidence on a DIFFERENT code; the SD-106 code is
    what MECH-566 is about, and what this run measures.
  * capacity is not the limit: V3-EXQ-1023a's deep2048x4 rung reaches train agreement ~1.0
    with held-out ~0.74 (memorisation); the sample-saturation witness costs SD-106 0.02-0.05
    at half the rows vs PCA-32 0.01-0.017.

  MANIPULATION: the coordinate system the FROZEN SD-106 code is presented to the consumer in.
                Members: diag (train-split z-score; the lineage's baseline, replicated in-run),
                zca (x1008._ZCAWhiten, full-covariance), lda (x1008._LDAWhiten, within-class
                whitening; supervised by the oracle's action classes on the TRAIN split only).
                All three are invertible affine maps fitted on the train split.
  HELD FIXED:   the frozen code, the 1002 dataset recipe (episode-disjoint split), the decoder
                (x1010's capacity ladder at the consumer rung mlp128 = x734.PPOPolicyNet at
                PPO_TRUNK_HIDDEN, plus the linear rung), the fit protocol (Adam, ADAPTER_LR,
                ADAPTER_BATCH, ADAPTER_PASSES, grad clip), the decoder init draw per seed and
                rung (x1010: same draw for every arm), the seeds [42, 43, 44].
  TWO FEATURE SOURCES, both run: `sense` = sense()-time z_world (what E1/E2/E3 actually read;
                V3-EXQ-1023a's instrument) and `path` = ZWorldP0Trainer._z_world_path(world_state)
                (the quantity P0a trained). SD-ZWORLD-SENSE-PATH-PARITY (substrate_queue,
                degrading, pending_implementation) records ~0.10 lower decodability on the
                sense path; running both makes that gap visible at the consumer rung on this
                code. M1 routes on the SENSE source (the consumer's real input); the path
                source is recorded context.
  INSTRUMENT CHECK (precondition, Half B): the OLS holdout R^2 of world_obs from the
                transformed code must equal that from the raw code to within
                INVARIANCE_TOL -- the transform is an element of the very group the preservation
                objective is invariant to, so a change here means the transform is NOT
                invertible (a bug), never a finding.
  CRITERION M1 (load-bearing for MECH-566), on the SENSE source at the consumer rung:
                lift_m = agree(sd106_sense_<m>__mlp128) - agree(sd106_sense_diag__mlp128)
                best_lift = max over m in {zca, lda}.
                CONFIRM  (MECH-566 supports): best_lift >= MECH566_CONFIRM_LIFT (0.05) on a
                         seed majority.
                FALSIFY  (MECH-566 weakens): every member's lift in [-0.02, 0.02) on a seed
                         majority -- "moves < 0.02" in the claim's own words means NO lift.
                HARMFUL  (MECH-566 unknown): any member's lift <= -0.02. A whitened code that
                         HURTS the consumer is the signature of near-null code directions being
                         amplified into the float32 MLP (red-team F4), or of a genuinely harmful
                         conditioning; `consumer_input_column_rms` per member separates the two.
                         Not a falsification: the falsifier names the absence of a lift.
                else     `conditioning_partial` (MECH-566 mixed): a real but insufficient
                         conditioning effect -- the claim's own grey zone, recorded as such and
                         an ACCEPTABLE TERMINAL READING (V3-EXQ-1008 put 5 of 6 OFF-code lifts
                         there; `gap_fraction_closed_by_best_member` says how much of the
                         SD-106 -> PCA-32 gap the best member closed).
                ZCA uses a RELATIVE ridge (ZCA_RIDGE_FRAC x lambda_max, x1008's LDA convention),
                not x1008's absolute 1e-6 floor, so a near-null direction is damped rather than
                amplified; the `--self-test` proves both that and that the invertibility
                precondition is BLIND to conditioning (it certifies invertibility only).
  GL-INVARIANT CONTENT CHECK (recorded, not routed on): a CONVERGED multinomial logistic
                regression (L-BFGS, near-zero ridge) is invariant to any invertible affine
                reconditioning of its input up to optimisation, so its held-out agreement on
                the SD-106 code vs on PCA-32 is a content comparison that conditioning cannot
                touch. V3-EXQ-1023a's 60-pass Adam linear rung read 0.61 (SD-106) vs 0.83
                (PCA-32) but had NOT converged (CE still falling). If the converged probe keeps
                a gap of that size, the two codes are not in one GL(32) orbit with respect to
                action-relevant content, which is a content difference -- MECH-566's premise
                that they are does not hold at the linear rung. If the gap closes, the 1023a
                linear-rung gap was an optimisation artefact.
  DV-SYMMETRY, per arm: the DV is held-out oracle-action agreement, an argmax over a
                freshly-fitted decoder's logits. The manipulation changes the decoder's INPUT
                coordinates, hence the reachable function under a fixed fit budget -- it is
                not a broadcast constant, a monotone rescaling of candidate scores, or a
                permutation of interchangeable units. At the LINEAR rung a converged fit IS
                invariant to it by design, which is why the linear rung is an instrument check
                here and NOT a MECH-566 criterion. The PCA-32 anchor and raw-field arms carry no
                manipulation; they locate the parity bar and prove the labels are readable.

=== WHY P0a-ONLY, AND HOW THAT IS CHECKED ===

V3-EXQ-1023a's 18.7 h were the x734 all-ON P0b/P1 warmup (290 episodes per agent) and the
deep decoder rungs. `experiments/_lib/allon_training.py` records that P0b/P1 cover NO
latent_stack parameter, so the z_world path is trained by P0a alone; V3-EXQ-1041 ran P0a only
on that basis. This run does the same. It is CHECKED rather than assumed: the in-run
`sd106_sense_diag__mlp128` cell reproduces V3-EXQ-1023a's consumer-rung instrument on the same
seeds, and the READINESS PRECONDITION `sd106_consumer_diag_reproduces_1023a` requires
|in-run - 1023a| <= REPRO_TOL (0.05) on every seed (scoped out under --dry-run). Unmet routes
`substrate_not_ready_requeue`, NOT a verdict: MECH-566's registered 0.05 lift is denominated
against 1023a's 0.7273 baseline, and a P0a-only baseline that drifts from it is a different
instrument (red-team F5) -- it would also mean P0b/P1 DO shape sense-time z_world, in which case
the successor letter runs the full x734 warmup and pays the 18 h.

=== STEP 2.5c (substrate-path overlap), measured on origin/master 2026-09-19 ===

Open `degrading` entries overlapping this driver: SD-106 / SD-018 (zworld_p0.py, stack.py --
the run's own subject), SD-ZWORLD-SENSE-PATH-PARITY (agent.sense, _z_world_path,
zworld_p0_warmup.py -- this run's two feature sources straddle exactly that gap and record it),
sd061-resume-progress-ecology and sd-allon-training-signal-absorption-telemetry
(allon_training -- NOT exercised: P0a only, `_train_all_on_agent` is never called). Recorded,
not blocking. The open `corrupting` entries (MECH-320 tonic_vigor, sd105 selection_entropy_floor,
sd_blocked_agency, contextmemory-write-path) are config-disabled on the x724 all-ON stack or
sit on paths this driver never runs: it constructs REEAgents, runs the P0a recipe under a
RandomPolicy, and calls sense() only for the frozen replay the sense-source features need.
sense() does not write ContextMemory (V3-EXQ-1023a's carried exception covered P1, which this
run omits).

=== ETHICS PREFLIGHT ===
ethics_preflight: involves_negative_valence false, involves_suffering_like_state false,
involves_self_model false, involves_inescapability_or_helplessness false,
involves_offline_replay_over_harm false, involves_social_mind_or_language false,
involves_human_data_or_clinical_context false, decision allow (SENT-0; hazard-free rung D3).

red-team: see the RED-TEAM RECORD at the end of this docstring and the queue entry note.

=== RED-TEAM RECORD (/queue-experiment Step 4.5; session model Fable 5.1, reviewer Opus) ===
PASS 1 -- red-team (opus): BLOCKING. 7 findings.
  F1 BLOCKING  the chip's mean-cos^2 decoder-range overlap has no dynamic range (untrained encoder
               0.953 of ceiling on the dry buffer) -> FIXED by USER DECISION (P1 statistic replaced;
               principal angles kept as a recorded readout).
  F2           P1's control failure never reached outcome / non_degenerate -> FIXED
               (non_degenerate_per_claim; degenerate half -> direction unknown).
  F3           a Half-A parity miss voided Half-B's MECH-566 evidence -> FIXED (halves independent).
  F4           invertibility check blind to conditioning; absolute ZCA ridge -> FIXED (relative
               ridge; consumer-input RMS + spectrum recorded; HARMFUL branch; self-test proves the
               blindness rather than pretending otherwise).
  F5           no gate on the in-run diag baseline -> FIXED (sd106_consumer_diag_reproduces_1023a).
  F6           modal M1 outcome is the claim's grey zone -> DISMISSED as a design change: those are
               MECH-566's REGISTERED bands (claims.yaml what_would_answer); PARTIAL is pre-registered
               as an acceptable terminal reading and gap_fraction_closed_by_best_member is recorded.
  F7           OFF probe slice asserted preservation 200 / skip on -> FIXED (per-arm slice values).
PASS 2 (the one permitted re-spawn: F1 changed the DV) -- red-team (opus): BLOCKING. 6 findings.
  RT2-1 BLOCKING a control-anchored score is fragile: a random orthonormal projection recovers PC
               scores ~exactly (world_obs is effectively 32-dim) and is not a content-free floor
               -> FIXED: P1 is the ABSOLUTE recoverability against the exact 1.0 ceiling; the
               untrained encoder is only a can-it-read-LOW guard.
  RT2-2        unclipped negative R^2 lets a broken floor drag s toward HIGH -> FIXED (0-clipped
               per-column R^2, no normalisation).
  RT2-3        the whitened consumer input's own conditioning was unrecorded -> FIXED
               (consumer_input_spectrum per member).
  RT2-4        Half-B-only preconditions voided Half-A's direction at the gate layer -> FIXED
               (gate_a / gate_b; each half voided by its own instrument only).
  RT2-5        FALSIFY band narrow vs seed spread; HARMFUL at -0.02 could mask a null -> PARTLY
               FIXED (HARMFUL now symmetric with CONFIRM at -0.05, so FALSIFY spans (-0.05, 0.02));
               "no null distribution measured" DISMISSED: the paired same-init design and seed
               majority are the lineage's instrument (x1010 red-team F8), and the bands are the
               claim's own.
  RT2-6        best_lift = max(zca, lda) let the label-supervised LDA carry CONFIRM -> FIXED
               (M1 routes on ZCA alone; LDA recorded as the supervised upper reference).
No third pass (the skill forbids iterating to CLEAR). Verdict recorded in the queue entry note.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from experiment_protocol import emit_outcome  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell, reset_all_rng  # noqa: E402
from experiments._lib.capability_eval import RandomPolicy  # noqa: E402
from experiments._lib.interface_probe import principal_angles  # noqa: E402
from experiments._lib.readiness_anchor import assert_anchor_reachable  # noqa: E402
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator  # noqa: E402
from experiments._lib.zworld_p0_warmup import run_zworld_p0  # noqa: E402
from experiments._lib.zworld_encoder_guard import (  # noqa: E402
    latent_stack_snapshot, latent_stack_weight_delta,
)
from experiments._metrics import check_degeneracy, p0_readiness_gate, P0NotReady  # noqa: E402
from ree_core.latent.zworld_p0 import ZWorldP0Config  # noqa: E402

import experiments.v3_exq_734_env_difficulty_competence_recovery_sweep as x734  # noqa: E402
import experiments.v3_exq_1002_zworld_actor_adequacy_oracle_adapter as x1002  # noqa: E402
import experiments.v3_exq_1008_zworld_adequacy_portfolio_ws250_rebasis as x1008  # noqa: E402
import experiments.v3_exq_1010_zworld_overcapacity_decoder_sweep as x1010  # noqa: E402
import experiments.v3_exq_1023_sd106_bottleneck_preservation_validation as x1023  # noqa: E402
import experiments.v3_exq_1023a_sd106_preservation_parity_epochs40 as x1023a  # noqa: E402
import experiments.v3_exq_1041_sd106_preservation_step_budget_metric_diagnostic as x1041  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_1065_sd106_subspace_overlap_mech566_recondition"
QUEUE_ID = "V3-EXQ-1065"
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"
EXPERIMENT_PURPOSE = "diagnostic"
CLAIM_IDS = ["SD-106", "MECH-566"]

DEVICE = x1002.DEVICE
RUNG = x1002.RUNG
RUNG_ID = x1002.RUNG_ID
LEVEL_ID = x1002.LEVEL_ID
STEPS_PER_EPISODE = x1002.STEPS_PER_EPISODE
BC_EPISODES = x1002.BC_EPISODES
BC_RANDOM_EPISODES = x1002.BC_RANDOM_EPISODES
ADAPTER_PASSES = x1002.ADAPTER_PASSES
ADAPTER_LR = x1002.ADAPTER_LR
ADAPTER_BATCH = x1002.ADAPTER_BATCH
SEED_MAJORITY = x1002.SEED_MAJORITY
RAW_FIELD_CONTROL_FLOOR = x1002.RAW_FIELD_CONTROL_FLOOR
ZWORLD_P0_EPISODES = x1002.ZWORLD_P0_EPISODES          # 60 -- the ONLY training phase here
PROJECTION_DIM = x1010.PROJECTION_DIM                   # 32
CONSUMER_RUNG = x1010.CONSUMER_RUNG                     # "mlp128"
LINEAR_RUNG = "linear"
RUNGS = [LINEAR_RUNG, CONSUMER_RUNG]
ARM_RAW = x1010.ARM_RAW                                 # "rawfield_ceiling"
SEEDS = list(x1023.SEEDS)                               # [42, 43, 44]
DRY_RUN_SEEDS = list(x1023.DRY_RUN_SEEDS)
# The lineage's dry-run P0a (2 episodes x 20 steps = 40 buffered rows) cannot fit a 32-subspace
# on each side of the trainer's 80/20 split (needs >= 34 rows per split). The smoke MUST
# exercise the probe's code path (V3-EXQ-591g: a short-circuited smoke is blind), so this
# driver's dry-run P0a runs 10 episodes (200 rows -> 160/40). resolve_p0a_config still forces
# epochs=2 / batch_size=8, so the smoke stays seconds long.
DRY_RUN_ZWORLD_P0 = 10
DRY_RUN_STEPS = x1023.DRY_RUN_STEPS
DRY_RUN_BC_EPISODES = x1023.DRY_RUN_BC_EPISODES
DRY_RUN_BC_RANDOM_EPISODES = x1023.DRY_RUN_BC_RANDOM_EPISODES
DRY_RUN_ADAPTER_PASSES = x1023.DRY_RUN_ADAPTER_PASSES

# SD-106 P0a budget and objective: V3-EXQ-1023a's, IMPORTED so this run cannot drift from the
# cell it re-measures (epochs=40, preservation_weight=200.0, skip on, resource field 0.0).
P0A_EPOCHS = int(x1023a.P0A_EPOCHS)                     # 40
P0A_STEP_FLOOR = int(x1023a.P0A_STEP_FLOOR)             # 1000
P0A_CONFIG = x1023a.P0A_CONFIG
SHIPPED_P0A_EPOCHS = int(ZWorldP0Config().epochs)      # 12 -- the OFF control's budget
PRESERVATION_WEIGHT = float(x1023.PRESERVATION_WEIGHT)  # 200.0
USE_WORLD_ENCODER_SKIP = bool(x1023.USE_WORLD_ENCODER_SKIP)
SD106_PARITY_BAR = float(x1023.SD106_PARITY_BAR)        # 0.85 -- the anchor's readiness bar
PARTICIPATION_RATIO_FLOOR = float(x1023.PARTICIPATION_RATIO_FLOOR)  # 2.0

# ---- PRE-REGISTERED THRESHOLDS (constants; never derived from this run's statistics) ---
# Half A. parity_ratio = posthoc_ols_r2 / pca32_ols_r2 on the captured buffer. V3-EXQ-1041
# measured 0.980 (epochs=40) / 0.951 (shipped) -- the premise the reading table rests on is the
# epochs=40 figure, so the floor sits between the two budgets.
PARITY_HIGH_FLOOR = 0.95
# P1 routes on the ABSOLUTE PC-RECOVERABILITY of the SD-106 code:
#     rec = unweighted mean over the 32 train-split principal directions of the (0-clipped)
#           holdout R^2 with which the code linearly recovers each direction's score.
# The ceiling is EXACT and needs no measurement: PCA-32's own coordinates recover every score
# with R^2 = 1 by construction. There is deliberately NO control-anchored normalisation (red-team
# RT2-1/RT2-2): world_obs is effectively 32-dimensional (PCA-32 explains 0.976 dry / 0.998
# full-scale of its variance), so ANY 32-dim LINEAR projection recovers the PC scores almost
# exactly and is not a content-free floor, while a normalising floor that reads negative drags
# the score toward HIGH. What rec measures is therefore what the code's NONLINEARITY and
# precision gate discard, per direction, with the low-variance tail counting as much as PC1.
# HIGH: the retained directions ARE the PCA-32 directions. LOW: at least a quarter of the
# directions are (on average) not linearly readable -- the which-directions signature, since
# parity_ratio (variance-WEIGHTED) sits at 0.98 for this cell. Between: indeterminate.
# Non-degeneracy guard: the statistic must be ABLE to read LOW -- the in-run random-init encoder
# must read <= P1_LOW_CEIL on a seed majority. The random orthonormal projection is recorded as
# the "linear keeps everything" reference (expected ~0.98 at full scale), never as a floor.
P1_HIGH_FLOOR = 0.90
P1_LOW_CEIL = 0.75
PC_TAIL_START = 10        # PCs 11..32: the low-variance tail, reported separately as context
# Principal-angle overlap (the chip's original statistic) is RECORDED beside P1 for every code,
# with these bands kept only so the recorded classification is readable; nothing routes on it
# (red-team F1: a random-init encoder already reads ~0.95 of ceiling on it).
OVERLAP_HIGH_FLOOR = 0.80
OVERLAP_LOW_CEIL = 0.60
# Half B. MECH-566's own registered numbers. A lift BELOW -MECH566_FALSIFY_LIFT is NOT a
# falsification: the claim's falsifier says "moves < 0.02", meaning no lift, and a large
# NEGATIVE move is the signature of whitening amplifying near-null code directions into the
# consumer (red-team F4) -- classified HARMFUL and left indeterminate for the falsifier.
MECH566_CONFIRM_LIFT = 0.05
MECH566_FALSIFY_LIFT = 0.02
MECH566_HARMFUL_LIFT = 0.05      # symmetric with CONFIRM: FALSIFY band is (-0.05, 0.02)
ZCA_RIDGE_FRAC = float(x1008.LDA_RIDGE_FRAC)     # 1e-3 of lambda_max, x1008's LDA convention
# Instrument: an invertible affine reconditioning cannot change the OLS holdout R^2 of
# world_obs from the code (float64 gelsd; the LDA ridge is on the WHITENING matrix, which stays
# invertible, not on the OLS).
INVARIANCE_TOL = 1.0e-4
# Reproduction of V3-EXQ-1023a's consumer-rung diag cell on the P0a-only agent (context).
REPRO_TOL = 0.05
REF_1023A_CONSUMER_DIAG: Dict[int, float] = {
    42: 0.7541899681091309, 43: 0.7355652451515198, 44: 0.6921119689941406,
}
REF_1023A_PCA_CONSUMER: Dict[int, float] = {
    42: 0.8770949840545654, 43: 0.8578360080718994, 44: 0.8702290058135986,
}
REF_1041_PARITY_RATIO_EPOCHS40 = 0.980   # design premise, cited not re-registered
# Converged linear probe (GL-invariant content check). Recorded, not routed on.
CONVERGED_LINEAR_L2 = 1.0e-6
CONVERGED_LINEAR_LBFGS_STEPS = 6          # outer steps of max_iter each
CONVERGED_LINEAR_LBFGS_MAX_ITER = 100
DRY_RUN_CONVERGED_LINEAR_LBFGS_STEPS = 1

# ---- ARMS -------------------------------------------------------------------------------
TRACK_PCA = x1010.TRACK_PCA                              # "ws250_pca" (anchor, diag only)
TRACK_SD106_SENSE = "sd106_sense"
TRACK_SD106_PATH = "sd106_path"
MEMBERS = ["diag", "zca", "lda"]
PROBE_SD106 = "probe_sd106"
PROBE_OFF = "probe_off"


def _arm_id(track: str, rung: str) -> str:
    return "%s__%s" % (track, rung)


def _member_track(source: str, member: str) -> str:
    return "%s_%s" % (source, member)


PCA_ARM_IDS = [_arm_id(TRACK_PCA, r) for r in RUNGS]
SENSE_ARM_IDS = [_arm_id(_member_track(TRACK_SD106_SENSE, m), r) for m in MEMBERS for r in RUNGS]
PATH_ARM_IDS = [_arm_id(_member_track(TRACK_SD106_PATH, m), r) for m in MEMBERS for r in RUNGS]
CONSUMER_ARM_IDS = [ARM_RAW] + PCA_ARM_IDS + SENSE_ARM_IDS + PATH_ARM_IDS   # 1 + 2 + 6 + 6 = 15
PROBE_ARM_IDS = [PROBE_SD106, PROBE_OFF]
ARM_IDS = PROBE_ARM_IDS + CONSUMER_ARM_IDS                                  # 17 per seed

_EXTRA_SUBSTRATE_PATHS = [Path(m.__file__) for m in (x1002, x1008, x1010, x1023, x1023a, x1041, x734)]
_ZG = ZGoalStreamAccumulator()


def _ctx(arm_id: str) -> Dict[str, Any]:
    if arm_id in PROBE_ARM_IDS:
        return {"id": arm_id, "arm_id": arm_id, "track": arm_id, "rung": None,
                "source": None, "member": None, "is_probe": True}
    track = ARM_RAW if arm_id == ARM_RAW else arm_id.split("__", 1)[0]
    rung = None if arm_id == ARM_RAW else arm_id.split("__", 1)[1]
    source, member = None, None
    for src in (TRACK_SD106_SENSE, TRACK_SD106_PATH):
        for m in MEMBERS:
            if track == _member_track(src, m):
                source, member = src, m
    return {"id": arm_id, "arm_id": arm_id, "track": track, "rung": rung,
            "source": source, "member": member, "is_probe": False,
            "is_anchor": bool(track == TRACK_PCA), "is_raw": bool(arm_id == ARM_RAW)}


# --------------------------------------------------------------------------------------
# WARMUP: P0a ONLY, buffer captured in situ (x1041's construction order)
# --------------------------------------------------------------------------------------
def _warm_p0a(track: str, seed: int, env_kwargs: Dict[str, Any], episodes: int,
              steps_per_episode: int, dry_run: bool, sink: List[torch.Tensor]):
    """Build the agent exactly as its lineage does, then run the SD-070 P0a recipe with the
    world_obs tap on the target. SD-106: x1023._make_sd106_agent + x1023a.P0A_CONFIG (epochs=40,
    preservation 200.0). OFF: x1002._make_agent at the shipped default (epochs=12,
    preservation 0.0) -- the negative control for the probe, budget-UNMATCHED and said so."""
    warm_env = x734._make_env(seed, env_kwargs)
    if track == PROBE_SD106:
        agent = x1023._make_sd106_agent(warm_env)
        cfg = P0A_CONFIG
    else:
        agent = x1002._make_agent(warm_env)
        cfg = ZWorldP0Config(preservation_weight=0.0, resource_field_weight=0.0,
                             epochs=int(SHIPPED_P0A_EPOCHS))
    before = latent_stack_snapshot(agent)
    p0_env = x734._make_env(seed, env_kwargs)
    stats = run_zworld_p0(
        agent, p0_env, seed, int(episodes), int(steps_per_episode),
        policy=RandomPolicy(seed), label="ree_allon rung=%s %s" % (RUNG_ID, track),
        dry_run=dry_run, config=cfg, target_fn=x1041._capturing_target(sink),
    )
    guard = latent_stack_weight_delta(agent, before)
    _ZG.observe(agent)
    skip = getattr(agent.latent_stack.split_encoder, "world_encoder_skip", None)
    skip_norm = float(skip.weight.detach().norm()) if skip is not None else 0.0
    return agent, stats, guard, skip_norm


# --------------------------------------------------------------------------------------
# ENCODER-PLANE PROBE HELPERS
# --------------------------------------------------------------------------------------
def _ols_fit(x_tr: torch.Tensor, y_tr: torch.Tensor) -> torch.Tensor:
    """Affine OLS x -> y (min-norm gelsd, float64). Returns W [d_x + 1, d_y]; last row = bias."""
    ones = torch.ones(x_tr.shape[0], 1, dtype=torch.float64)
    a = torch.cat([x_tr.double(), ones], dim=1)
    return torch.linalg.lstsq(a, y_tr.double(), driver="gelsd").solution


def _ols_apply(w: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    ones = torch.ones(x.shape[0], 1, dtype=torch.float64)
    return torch.cat([x.double(), ones], dim=1) @ w


def _r2_per_column(y_true: torch.Tensor, y_pred: torch.Tensor, y_tr_mean: torch.Tensor
                   ) -> torch.Tensor:
    """Per-column holdout R^2 against the TRAIN-split mean predictor (x1041's definition,
    columnwise). A train-constant column has no variance to explain and reads nan; callers
    drop nan before summarising."""
    mse = ((y_pred.double() - y_true.double()) ** 2).mean(dim=0)
    base = ((y_tr_mean.double().expand_as(y_true) - y_true.double()) ** 2).mean(dim=0)
    # CLIPPED at 0: a negative holdout R^2 means "not recovered" (worse than the train mean),
    # and letting it go negative lets a broken floor drag a normalised score toward HIGH
    # (red-team RT2-2). The unclipped value is not needed by any criterion.
    out = (1.0 - mse / base).clamp(min=0.0)
    out[base <= 0.0] = float("nan")
    return out


def _decoder_range_basis(z_tr: torch.Tensor, obs_tr: torch.Tensor) -> torch.Tensor:
    """Column space of the OLS decoder code -> world_obs, as a [250, 32] basis: the directions
    of observation space the code reconstructs INTO. `principal_angles` QR-orthonormalises."""
    w = _ols_fit(z_tr, obs_tr)                  # [33, 250]
    return w[:-1, :].T.contiguous()             # [250, 32]


def _encoder_domain_basis(obs_tr: torch.Tensor, z_tr: torch.Tensor) -> torch.Tensor:
    """Column space of the OLS map world_obs -> code, [250, 32]: the directions of observation
    space the code is LINEARLY SENSITIVE to. Secondary to the decoder-range basis."""
    w = _ols_fit(obs_tr, z_tr)                  # [251, 32]
    return w[:-1, :].contiguous()               # [250, 32]


def _pca_basis(obs: torch.Tensor, k: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
    W, stats = x1008._world_state_pca_stats(obs, int(k))
    return W, stats


def _spectrum(z: torch.Tensor) -> Dict[str, Any]:
    if z.ndim != 2 or int(z.shape[0]) < 2:
        return {"fitted": False}
    zc = (z - z.mean(dim=0, keepdim=True)).double()
    s = torch.linalg.svdvals(zc)
    var = (s ** 2) / float(zc.shape[0] - 1)
    s_min = float(s.min()) if s.numel() else 0.0
    return {
        "fitted": True,
        "singular_values": [float(v) for v in s],
        "condition_number": (float(s.max()) / s_min) if s_min > 0.0 else None,
        "participation_ratio": x1002._participation_ratio(z),
        "variance_top1_share": float(var[0] / var.sum()) if float(var.sum()) > 0 else None,
    }


def _overlap_block(z_tr: torch.Tensor, z_te: torch.Tensor, obs_tr: torch.Tensor,
                   obs_te: torch.Tensor, V_tr: torch.Tensor, V_te: torch.Tensor,
                   p_tr: torch.Tensor, p_te: torch.Tensor) -> Dict[str, Any]:
    """The subspace-overlap + decision-cell readouts for ONE code on ONE buffer split. Shared by
    the trained probe and the untrained control so the two are computed identically."""
    U_dec = _decoder_range_basis(z_tr, obs_tr)
    U_enc = _encoder_domain_basis(obs_tr, z_tr)
    pa_dec = principal_angles(U_dec, V_te)
    pa_dec_ins = principal_angles(U_dec, V_tr)
    pa_enc = principal_angles(U_enc, V_te)
    out: Dict[str, Any] = {
        "principal_angles_decoder_range_vs_pca32_holdout": pa_dec,
        "principal_angles_decoder_range_vs_pca32_train_insample": pa_dec_ins,
        "principal_angles_encoder_domain_vs_pca32_holdout": pa_enc,
        "overlap_decoder_range": float(pa_dec["mean_squared_cosine_overlap"]),
        "overlap_decoder_range_insample": float(pa_dec_ins["mean_squared_cosine_overlap"]),
        "overlap_encoder_domain": float(pa_enc["mean_squared_cosine_overlap"]),
    }
    ols = x1041._ols_holdout_r2(z_tr, obs_tr, z_te, obs_te)
    pca = x1041._ols_holdout_r2(p_tr, obs_tr, p_te, obs_te)
    out["posthoc_ols_r2"] = ols["r2"]
    out["parity_ratio"] = (None if (ols["r2"] is None or not pca["r2"])
                           else float(ols["r2"]) / float(pca["r2"]))
    idx_dec = [int(j) for j in x1008.DECISION_WORLD_STATE_INDICES]
    idx_field = list(range(int(x1008.RESOURCE_FIELD_OFFSET),
                           int(x1008.RESOURCE_FIELD_OFFSET) + int(x1008.RESOURCE_FIELD_DIM)))
    mu_tr = obs_tr.double().mean(dim=0, keepdim=True)
    r2_code = _r2_per_column(obs_te, _ols_apply(_ols_fit(z_tr, obs_tr), z_te), mu_tr)

    def _mean_over(idx: List[int], v: torch.Tensor) -> Optional[float]:
        vals = [float(v[j]) for j in idx if j < v.numel() and v[j] == v[j]]
        return (float(np.mean(vals)) if vals else None)

    out["decision_cells_r2"] = {str(j): (None if r2_code[j] != r2_code[j] else float(r2_code[j]))
                                for j in idx_dec if j < r2_code.numel()}
    out["decision_cells_r2_mean"] = _mean_over(idx_dec, r2_code)
    out["field_block_r2_mean"] = _mean_over(idx_field, r2_code)
    pc_te = (obs_te.double() - mu_tr) @ V_tr.double()
    pc_tr = (obs_tr.double() - mu_tr) @ V_tr.double()
    r2_pc = _r2_per_column(pc_te, _ols_apply(_ols_fit(z_tr, pc_tr.float()), z_te),
                           pc_tr.mean(dim=0, keepdim=True))
    out["pc_recoverability_r2"] = [(None if v != v else float(v)) for v in r2_pc]
    finite = [float(v) for v in r2_pc if v == v]
    # THE P1 STATISTIC (user decision 2026-09-19, replacing mean-cos^2 after the red-team's
    # BLOCKING finding): the UNWEIGHTED mean over the 32 train-split principal directions of the
    # holdout R^2 with which the code linearly recovers each direction's score. Unweighted, so
    # the low-variance tail -- where a variance-preserving objective is free to discard -- counts
    # as much as PC1. PCA-32's own coordinates recover every score exactly (R^2 = 1 by
    # construction), which is the ceiling; the in-run untrained encoder is the floor.
    out["pc_recoverability_mean"] = (float(np.mean(finite)) if finite else None)
    out["pc_recoverability_tail_mean"] = (float(np.mean(finite[PC_TAIL_START:]))
                                          if len(finite) > PC_TAIL_START else None)
    out["z_spectrum"] = _spectrum(z_tr)
    return out


def _probe_encoder_plane(track: str, seed: int, agent: Any, stats: Dict[str, Any],
                         sink: List[torch.Tensor], row: Dict[str, Any],
                         env_kwargs: Dict[str, Any]) -> None:
    """Half A on one warmed agent. Mutates `row`. All quantities on the CAPTURED buffer and the
    trainer's OWN split, exactly as V3-EXQ-1041 did; the new readouts sit beside the old."""
    cfg_dict = dict(stats.get("p0a_config") or {})
    obs = torch.stack(sink)
    row["capture_n_delta_abs"] = abs(int(obs.shape[0]) - int(row.get("n_buffered") or 0))
    row["world_obs_dim"] = int(obs.shape[1])
    tr_idx, te_idx = x1041._reproduce_split(int(obs.shape[0]), cfg_dict)
    obs_tr, obs_te = obs[tr_idx], obs[te_idx]
    cfg_obj = ZWorldP0Config(**cfg_dict)
    z_tr = x1041._z_world(agent, cfg_obj, obs_tr)
    z_te = x1041._z_world(agent, cfg_obj, obs_te)
    if int(z_tr.shape[1]) != int(PROJECTION_DIM):
        raise RuntimeError("z_world width %d != PROJECTION_DIM %d; the PCA-32 anchor would not "
                           "be at the encoder's own width. Refusing to run."
                           % (int(z_tr.shape[1]), int(PROJECTION_DIM)))
    row["z_dim"] = int(z_tr.shape[1])
    row["z_all_finite"] = bool(torch.isfinite(z_tr).all() and torch.isfinite(z_te).all())
    row["z_spectrum"] = _spectrum(z_tr)
    row["z_participation_ratio"] = row["z_spectrum"].get("participation_ratio")

    # ---- the EXISTING post-hoc OLS R^2, unchanged (x1041) ---------------------------------
    ols = x1041._ols_holdout_r2(z_tr, obs_tr, z_te, obs_te)
    row["posthoc_ols"] = ols
    row["posthoc_ols_r2"] = ols["r2"]
    p_tr, p_te = x1041._pca_project(obs_tr, obs_te, PROJECTION_DIM)
    pca = x1041._ols_holdout_r2(p_tr, obs_tr, p_te, obs_te)
    row["pca32_ols"] = pca
    row["pca32_ols_r2"] = pca["r2"]
    row["parity_ratio"] = (None if (ols["r2"] is None or not pca["r2"])
                           else float(ols["r2"]) / float(pca["r2"]))

    # ---- the NEW readout: principal angles / subspace overlap -----------------------------
    n_tr, n_te = int(obs_tr.shape[0]), int(obs_te.shape[0])
    if n_tr < PROJECTION_DIM + 2 or n_te < PROJECTION_DIM + 2:
        # A dry run buffers too few rows for a 32-subspace estimate on each split. Record the
        # shortfall honestly; every overlap reads None and the probe precondition scopes out.
        row["overlap_fitted"] = False
        for k in ("overlap_decoder_range", "overlap_encoder_domain", "overlap_ceiling_split",
                  "overlap_chance_random", "overlap_rel", "decision_cells_r2_code_mean",
                  "decision_cells_r2_pca_mean", "field_block_r2_code_mean",
                  "field_block_r2_pca_mean", "pc_recoverability_r2_from_code"):
            row[k] = None
        return
    row["overlap_fitted"] = True
    # OUT-OF-SAMPLE SCORING, SAME REFERENCE FOR CODE AND CEILING. Both the code's basis (the
    # OLS decoder, fitted on the train split) and the PCA-32 reference (fitted on the train
    # split) are scored against PCA-32 of the HOLDOUT split. The ceiling is therefore "how
    # well the train-split PCA-32 itself matches the holdout PCA-32", and the code faces the
    # same sampling noise it does. (An in-sample form -- code vs train-split PCA -- is recorded
    # as context; it is NOT comparable to the split ceiling and the smoke showed it exceeding
    # it, which is what forced this form.)
    V_tr, pca_stats = _pca_basis(obs_tr, PROJECTION_DIM)
    V_te, _ = _pca_basis(obs_te, PROJECTION_DIM)
    R = x1008._random_orthonormal(int(obs.shape[1]), PROJECTION_DIM, seed)
    pa_ceiling = principal_angles(V_tr, V_te)
    pa_chance = principal_angles(R, V_te)
    row["pca_train_stats"] = pca_stats
    row["pc_variance_explained_train"] = pca_stats.get("variance_explained_at_k")
    row["principal_angles_pca32_train_vs_holdout"] = pa_ceiling
    row["principal_angles_random32_vs_pca32_holdout"] = pa_chance
    row["overlap_ceiling_split"] = float(pa_ceiling["mean_squared_cosine_overlap"])
    row["overlap_chance_random"] = float(pa_chance["mean_squared_cosine_overlap"])
    row["overlap_chance_analytic"] = float(PROJECTION_DIM) / float(obs.shape[1])
    ceil = row["overlap_ceiling_split"]

    blk = _overlap_block(z_tr, z_te, obs_tr, obs_te, V_tr, V_te, p_tr, p_te)
    row.update({k: v for k, v in blk.items() if k not in ("posthoc_ols_r2", "parity_ratio",
                                                            "z_spectrum")})
    row["overlap_rel"] = (row["overlap_decoder_range"] / ceil) if ceil > 0.0 else None
    row["decision_cells_r2_code"] = blk["decision_cells_r2"]
    row["decision_cells_r2_code_mean"] = blk["decision_cells_r2_mean"]
    row["field_block_r2_code_mean"] = blk["field_block_r2_mean"]
    row["pc_recoverability_r2_from_code"] = blk["pc_recoverability_r2"]
    row["pc_recoverability_mean"] = blk["pc_recoverability_mean"]
    row["pc_recoverability_tail_mean"] = blk["pc_recoverability_tail_mean"]
    # A second floor: a random orthonormal 32-projection of world_obs read through the same
    # recoverability probe (recorded; the untrained ENCODER is the anchoring floor).
    rp_tr = (obs_tr.double() - obs_tr.double().mean(dim=0, keepdim=True)) @ R.double()
    rp_te = (obs_te.double() - obs_tr.double().mean(dim=0, keepdim=True)) @ R.double()
    rblk = _overlap_block(rp_tr.float(), rp_te.float(), obs_tr, obs_te, V_tr, V_te, p_tr, p_te)
    row["random_projection_control"] = {k: rblk[k] for k in (
        "pc_recoverability_mean", "pc_recoverability_tail_mean", "overlap_decoder_range",
        "decision_cells_r2_mean", "parity_ratio")}
    # The same decision-cell / field-block readouts for the PCA-32 coordinates (the anchor).
    mu_tr = obs_tr.double().mean(dim=0, keepdim=True)
    r2_pca = _r2_per_column(obs_te, _ols_apply(_ols_fit(p_tr, obs_tr), p_te), mu_tr)
    idx_dec = [int(j) for j in x1008.DECISION_WORLD_STATE_INDICES]
    idx_field = list(range(int(x1008.RESOURCE_FIELD_OFFSET),
                           int(x1008.RESOURCE_FIELD_OFFSET) + int(x1008.RESOURCE_FIELD_DIM)))
    vals_d = [float(r2_pca[j]) for j in idx_dec if j < r2_pca.numel() and r2_pca[j] == r2_pca[j]]
    vals_f = [float(r2_pca[j]) for j in idx_field if j < r2_pca.numel() and r2_pca[j] == r2_pca[j]]
    row["decision_cells_r2_pca"] = {str(j): (None if r2_pca[j] != r2_pca[j] else float(r2_pca[j]))
                                    for j in idx_dec if j < r2_pca.numel()}
    row["decision_cells_r2_pca_mean"] = float(np.mean(vals_d)) if vals_d else None
    row["field_block_r2_pca_mean"] = float(np.mean(vals_f)) if vals_f else None

    # ---- THE UNTRAINED-ENCODER CONTROL, on the SAME buffer and split (SD-106 cell only) ----
    # A random-init world path read through the identical probe. If ITS decoder range also
    # overlaps PCA-32 in the HIGH band, the overlap statistic does not discriminate a trained
    # code from noise and a HIGH reading on SD-106 is vacuous: P1's non-degeneracy is keyed to
    # this control reading BELOW the HIGH band. No warmup is needed -- the buffer is the SD-106
    # cell's own capture, so this costs one agent construction and two encoder passes.
    if track == PROBE_SD106:
        reset_all_rng(seed + 40_000)
        unt_env = x734._make_env(seed, env_kwargs)
        unt_agent = x1023._make_sd106_agent(unt_env)
        zu_tr = x1041._z_world(unt_agent, cfg_obj, obs_tr)
        zu_te = x1041._z_world(unt_agent, cfg_obj, obs_te)
        ublk = _overlap_block(zu_tr, zu_te, obs_tr, obs_te, V_tr, V_te, p_tr, p_te)
        ublk["overlap_rel"] = (ublk["overlap_decoder_range"] / ceil) if ceil > 0.0 else None
        ublk["participation_ratio"] = x1002._participation_ratio(zu_tr)
        row["untrained_control"] = ublk
        rec_u = ublk.get("pc_recoverability_mean")
        rec_c = row.get("pc_recoverability_mean")
        row["p1_score"] = rec_c                      # ABSOLUTE; ceiling 1.0 exact
        row["p1_untrained_reads_low"] = (None if rec_u is None
                                         else bool(float(rec_u) <= P1_LOW_CEIL))
        dec_c, dec_u, dec_p = (row.get("decision_cells_r2_code_mean"),
                               ublk.get("decision_cells_r2_mean"),
                               row.get("decision_cells_r2_pca_mean"))
        row["decision_cells_score"] = (
            None if (dec_c is None or dec_u is None or dec_p is None
                     or float(dec_p) - float(dec_u) <= 0.0)
            else (float(dec_c) - float(dec_u)) / (float(dec_p) - float(dec_u)))


# --------------------------------------------------------------------------------------
# CONSUMER-RUNG HELPERS (Half B)
# --------------------------------------------------------------------------------------
def _path_feats(agent: Any, feats: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """The P0a-trained quantity (`_z_world_path(world_state)`) on the 1002 dataset's rows."""
    cfg_obj = P0A_CONFIG
    return {k: x1041._z_world(agent, cfg_obj, feats["ws"][k]) for k in ("tr", "te", "r")}


class _ZCAWhitenRel(x1008._ZCAWhiten):
    """x1008's ZCA with a RELATIVE ridge (ZCA_RIDGE_FRAC x the largest covariance eigenvalue),
    matching the convention x1008._LDAWhiten already uses, instead of the ABSOLUTE 1e-6 floor.
    On a code with ~14 near-null directions (PR 17.7 of 32 at this budget) an absolute floor
    lets a near-null direction be amplified without bound into the float32 consumer while the
    invertibility check stays green (red-team F4); a relative ridge damps a direction whose
    eigenvalue is below the ridge toward zero instead. Same class contract (`__call__`,
    `condition_number`, `report`)."""
    kind = "zca_whitening_relative_ridge"

    def __init__(self, x_tr: torch.Tensor) -> None:
        n = int(x_tr.shape[0])
        self.mean = x_tr.mean(dim=0, keepdim=True) if n else None
        if n >= 2:
            c = x_tr - self.mean
            cov = (c.T @ c) / float(n - 1)
            lam_max = float(torch.linalg.eigvalsh(cov).max()) if int(cov.shape[0]) else 0.0
            ridge = float(ZCA_RIDGE_FRAC) * lam_max + float(x1008.WHITEN_EPS)
            self.T, self._st = x1008._inv_sqrt_psd(cov, ridge)
            self._st["ridge"] = ridge
            self._st["ridge_frac"] = float(ZCA_RIDGE_FRAC)
        else:
            self.T, self._st = torch.eye(int(x_tr.shape[1])), {"condition_number": 1.0}


def _make_transform(member: str, x_tr: torch.Tensor, y_tr: torch.Tensor, action_dim: int):
    if member == "diag":
        return x1008._DiagZ(x_tr)
    if member == "zca":
        return _ZCAWhitenRel(x_tr)
    if member == "lda":
        return x1008._LDAWhiten(x_tr, y_tr, action_dim)
    raise KeyError("unknown member: %s" % member)


def _column_rms_report(x: torch.Tensor) -> Dict[str, Any]:
    """Per-dimension RMS of the transformed consumer input: the amplification witness the
    invertibility check cannot carry (it is invariant to any full-rank map)."""
    if x.ndim != 2 or int(x.shape[0]) < 2:
        return {"fitted": False}
    rms = torch.sqrt((x.double() ** 2).mean(dim=0))
    lo = float(rms.min())
    return {"fitted": True, "rms_max": float(rms.max()), "rms_min": lo,
            "rms_ratio": (float(rms.max()) / lo if lo > 0.0 else None)}


def _invariance_check(transform: Any, z_tr: torch.Tensor, z_te: torch.Tensor,
                      ws_tr: torch.Tensor, ws_te: torch.Tensor) -> Dict[str, Any]:
    """OLS holdout R^2 of world_obs from the RAW code vs from the TRANSFORMED code. Equal to
    within INVARIANCE_TOL for any invertible affine transform -- the instrument check."""
    raw = x1041._ols_holdout_r2(z_tr, ws_tr, z_te, ws_te)
    tx = x1041._ols_holdout_r2(transform(z_tr), ws_tr, transform(z_te), ws_te)
    delta = (abs(float(raw["r2"]) - float(tx["r2"]))
             if (raw["r2"] is not None and tx["r2"] is not None) else None)
    return {"raw_r2": raw["r2"], "transformed_r2": tx["r2"], "abs_delta": delta,
            "numerical_rank_raw": raw["numerical_rank"],
            "numerical_rank_transformed": tx["numerical_rank"]}


def _converged_linear_agreement(x_tr: torch.Tensor, y_tr: torch.Tensor, x_te: torch.Tensor,
                                y_te: torch.Tensor, action_dim: int, seed: int,
                                lbfgs_steps: int) -> Dict[str, Any]:
    """Multinomial logistic regression fitted to (near) convergence with L-BFGS, tiny L2.
    Invariant to invertible affine reconditioning of the input up to optimisation, so the
    SD-106-vs-PCA-32 gap it reports is a CONTENT gap. Recorded, never routed on."""
    if int(x_tr.shape[0]) == 0 or int(x_te.shape[0]) == 0:
        return {"fitted": False}
    reset_all_rng(seed + 30_000)
    d = int(x_tr.shape[1])
    W = torch.zeros(d, int(action_dim), dtype=torch.float64, requires_grad=True)
    b = torch.zeros(int(action_dim), dtype=torch.float64, requires_grad=True)
    xt, yt = x_tr.double(), y_tr.long()
    opt = torch.optim.LBFGS([W, b], lr=1.0, max_iter=int(CONVERGED_LINEAR_LBFGS_MAX_ITER),
                            line_search_fn="strong_wolfe", tolerance_grad=1e-9,
                            tolerance_change=1e-12, history_size=50)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(xt @ W + b, yt) + \
            float(CONVERGED_LINEAR_L2) * (W ** 2).sum()
        loss.backward()
        return loss

    # `opt.step(closure)` returns the FIRST closure evaluation of that step (the loss BEFORE
    # the step's iterations), so it is recorded as `loss_before_step`, not as a final loss.
    losses = []
    for _ in range(int(lbfgs_steps)):
        losses.append(float(opt.step(closure).detach()))
    with torch.no_grad():
        pred_te = (x_te.double() @ W + b).argmax(dim=1)
        pred_tr = (xt @ W + b).argmax(dim=1)
    # final loss + gradient norm, evaluated AFTER the last step, as the convergence witness
    opt.zero_grad()
    loss = torch.nn.functional.cross_entropy(xt @ W + b, yt) + float(CONVERGED_LINEAR_L2) * (W ** 2).sum()
    loss.backward()
    grad_norm = float(torch.sqrt((W.grad ** 2).sum() + (b.grad ** 2).sum()))
    return {
        "fitted": True,
        "heldout_agreement": float((pred_te == y_te.long()).double().mean()),
        "train_agreement": float((pred_tr == yt).double().mean()),
        "final_loss": float(loss.detach()),
        "loss_before_step": losses,
        "final_grad_norm": grad_norm,
        "l2": float(CONVERGED_LINEAR_L2),
        "lbfgs_steps": int(lbfgs_steps),
        "max_iter_per_step": int(CONVERGED_LINEAR_LBFGS_MAX_ITER),
    }


def _assert_gates_reachable() -> List[Dict[str, Any]]:
    """Refuse, at setup, any readiness gate its own recorded positive control cannot clear.
    The two anchors with a recorded reference are certified with THE SHIPPED PREDICATES against
    x1023's frozen V3-EXQ-1010 cells (the same reference x1023 / x1023a certify against). The
    remaining preconditions are definitional and reachable by construction (||W_skip|| > 0 =
    "the bypass trained at all"; n_changed >= 1 = "the encoder trained at all"; PR >= 2.0 is
    SD-070's own absolute floor measured 17.7 at this budget; capture delta == 0 and the
    invariance delta < tol are exact-arithmetic identities) -- stated rather than silently
    omitted."""
    out = []
    out.append(assert_anchor_reachable(
        anchor_name="anchor_pca32_reaches_parity_bar_on_majority",
        reference_cells=list(x1023.REF_PCA_CONSUMER_1010),
        score_fn=lambda v: float(v) >= SD106_PARITY_BAR,   # THE SHIPPED PREDICATE
        threshold=float(SEED_MAJORITY) / 3.0,
        reference_source=("V3-EXQ-1010 ws250_pca at mlp128, seeds 42/43/44 (0.8836 / 0.8578 / "
                          "0.8702); V3-EXQ-1023a reproduced 0.8771 / 0.8578 / 0.8702")))
    out.append(assert_anchor_reachable(
        anchor_name="instrument_rawfield_control_supra_floor",
        reference_cells=list(x1023.REF_RAWFIELD_1010),
        score_fn=lambda v: float(v) >= RAW_FIELD_CONTROL_FLOOR,   # THE SHIPPED PREDICATE
        threshold=1.0,
        reference_source="V3-EXQ-1010 rawfield_ceiling 0.9735-0.9832"))
    return out


def _config_slice(base: Dict[str, Any], arm_id: str) -> Dict[str, Any]:
    ctx = _ctx(arm_id)
    d = dict(base)
    d.update({
        "arm_id": arm_id, "arm_track": ctx["track"], "capacity_rung": ctx["rung"],
        "feature_source": ctx.get("source"), "recondition_member": ctx.get("member"),
        "projection_dim": int(PROJECTION_DIM),
        # The SD-106 manipulation belongs IN the slice and must say what the cell actually ran:
        # the OFF probe trains at preservation 0.0 with no bypass on x1002's agent (red-team F7 --
        # a slice asserting 200.0/True on that cell is a false-HIT reuse key).
        "sd106_preservation_weight": (0.0 if arm_id == PROBE_OFF else float(PRESERVATION_WEIGHT)),
        "sd106_use_world_encoder_skip": (False if arm_id == PROBE_OFF else bool(USE_WORLD_ENCODER_SKIP)),
        "agent_family": ("x1002_all_on_off" if arm_id == PROBE_OFF else "x1023_sd106_all_on"),
        "zworld_p0_epochs": (int(SHIPPED_P0A_EPOCHS) if arm_id == PROBE_OFF else int(P0A_EPOCHS)),
        "warmup_phases": "p0a_only",
        "feature_standardisation": "member-defined",
        # Readout-affecting constants the cell's call graph reads (validate_experiments
        # config_slice_declaration): the converged-linear probe's ridge and iteration budget.
        "converged_linear_l2": float(CONVERGED_LINEAR_L2),
        "converged_linear_lbfgs_max_iter": int(CONVERGED_LINEAR_LBFGS_MAX_ITER),
        "converged_linear_lbfgs_steps": int(base.get("converged_linear_lbfgs_steps", 0) or 0),
        "pc_tail_start": int(PC_TAIL_START),
        "zca_ridge_frac": float(ZCA_RIDGE_FRAC),
        "p1_low_ceil": float(P1_LOW_CEIL),
    })
    return d


# --------------------------------------------------------------------------------------
# CELLS
# --------------------------------------------------------------------------------------
def run_probe_cell(arm_id: str, seed: int, sched: Dict[str, int], env_kwargs: Dict[str, Any],
                   cfg_base: Dict[str, Any], frozen: Dict[str, Any], dry_run: bool
                   ) -> Dict[str, Any]:
    """Warm one agent (P0a only, buffer captured) and run Half A on it. The SD-106 agent is
    kept in `frozen` for the consumer cells; the OFF agent is the probe's negative control."""
    print("Seed %d Condition %s:%s" % (seed, RUNG_ID, arm_id), flush=True)
    with arm_cell(seed, config_slice=_config_slice(cfg_base, arm_id), script_path=Path(__file__),
                  config_slice_declared=True, include_driver_script_in_hash=False,
                  extra_substrate_paths=_EXTRA_SUBSTRATE_PATHS) as cell:
        sink: List[torch.Tensor] = []
        agent, stats, guard, skip_norm = _warm_p0a(
            arm_id, seed, env_kwargs, sched["zworld_p0"], sched["steps"], dry_run, sink)
        row: Dict[str, Any] = {
            "cell_id": "%s|seed%d" % (arm_id, seed), "arm_id": arm_id, "track": arm_id,
            "capacity_rung": None, "seed": int(seed), "is_probe": True,
            "p0a_ran": bool(stats.get("p0a_ran")), "p0a_reason": stats.get("p0a_reason"),
            "n_buffered": stats.get("p0a_n_buffered"), "p0a_n_steps": stats.get("p0a_n_steps"),
            "p0a_epochs": int((stats.get("p0a_config") or {}).get("epochs") or 0),
            "used_preservation_head": stats.get("p0a_used_preservation_head"),
            "used_world_encoder_skip": stats.get("p0a_used_world_encoder_skip"),
            "skip_weight_norm": float(skip_norm),
            "world_encoder_weight_delta": guard,
            "grounding_label_balance": stats.get("p0a_grounding_label_balance"),
            "p0a_holdout": stats.get("p0a_holdout"),
            "p0a_config": stats.get("p0a_config"),
        }
        pres = stats.get("p0a_preservation_holdout")
        row["sgd_head_r2"] = pres.get("r2") if isinstance(pres, dict) else None
        if row["p0a_ran"]:
            _probe_encoder_plane(arm_id, seed, agent, stats, sink, row, env_kwargs)
        else:
            row["overlap_fitted"] = False
            for k in ("posthoc_ols_r2", "pca32_ols_r2", "parity_ratio", "overlap_decoder_range",
                      "overlap_rel", "overlap_ceiling_split", "capture_n_delta_abs"):
                row[k] = None
        if arm_id == PROBE_SD106:
            frozen["sd106_agent"] = agent
        cell.stamp(row)
    print("  [probe] %s seed=%d ols_r2=%s pca32=%s parity=%s overlap=%s ceiling=%s chance=%s"
          % (arm_id, seed, x1041._fmt(row.get("posthoc_ols_r2")), x1041._fmt(row.get("pca32_ols_r2")),
             x1041._fmt(row.get("parity_ratio")), x1041._fmt(row.get("overlap_decoder_range")),
             x1041._fmt(row.get("overlap_ceiling_split")), x1041._fmt(row.get("overlap_chance_random"))),
          flush=True)
    print("verdict: %s" % ("PASS" if row.get("overlap_fitted") else "FAIL"), flush=True)
    return row


def run_consumer_cell(arm_id: str, seed: int, data: Dict[str, Any], feats: Dict[str, Any],
                      frozen: Dict[str, Any], action_dim: int, sched: Dict[str, int],
                      cfg_base: Dict[str, Any], dry_run: bool) -> Dict[str, Any]:
    ctx = _ctx(arm_id)
    print("Seed %d Condition %s:%s" % (seed, RUNG_ID, arm_id), flush=True)
    passes = sched["passes"]
    # Every SD-106 cell reads the seed's ONE frozen agent (and the two sources share it), so the
    # cells are not independent -- stamped reuse-ineligible for exactly that reason.
    ineligible = (["frozen_agent_shared_across_members_rungs_and_sources"]
                  if ctx.get("source") else [])
    with arm_cell(seed, config_slice=_config_slice(cfg_base, arm_id), script_path=Path(__file__),
                  config_slice_declared=True, include_driver_script_in_hash=False,
                  extra_substrate_paths=_EXTRA_SUBSTRATE_PATHS,
                  extra_ineligible_reasons=ineligible) as cell:
        y_tr, y_te, yr_te = feats["y_tr"], feats["y_te"], feats["yr_te"]
        f_tr, f_te = feats["field"]["tr"], feats["field"]["te"]
        if ctx["is_raw"]:
            x_tr, x_te, xr_te = f_tr, f_te, feats["field"]["r"]
            row = x1010._fit_track_cell(arm_id, seed, data, action_dim, passes,
                                        x_tr, y_tr, x_te, y_te, xr_te, yr_te,
                                        x1008._DiagZ(x_tr), None, None)
        elif ctx["is_anchor"]:
            x_tr, x_te, xr_te = feats["ws"]["tr"], feats["ws"]["te"], feats["ws"]["r"]
            if "pca" not in frozen:
                W, stats = x1008._world_state_pca_stats(x_tr, PROJECTION_DIM)
                frozen["pca"] = x1008._LinearProjection(x_tr, W, "pca_32", extra=stats)
            row = x1010._fit_track_cell(arm_id, seed, data, action_dim, passes,
                                        x_tr, y_tr, x_te, y_te, xr_te, yr_te,
                                        frozen["pca"], f_tr, f_te)
            if ctx["rung"] == LINEAR_RUNG:
                row["converged_linear"] = _converged_linear_agreement(
                    frozen["pca"](x_tr), y_tr, frozen["pca"](x_te), y_te, action_dim, seed,
                    sched["lbfgs_steps"])
        else:
            agent = frozen["sd106_agent"]
            src = ctx["source"]
            if src not in frozen:
                frozen[src] = (x1008._z_feats(agent, data) if src == TRACK_SD106_SENSE
                               else _path_feats(agent, feats))
            z = frozen[src]
            x_tr, x_te, xr_te = z["tr"], z["te"], z["r"]
            transform = _make_transform(ctx["member"], x_tr, y_tr, action_dim)
            row = x1010._fit_track_cell(arm_id, seed, data, action_dim, passes,
                                        x_tr, y_tr, x_te, y_te, xr_te, yr_te,
                                        transform, f_tr, f_te)
            row["zworld_participation_ratio"] = x1002._participation_ratio(x_tr)
            row["z_spectrum"] = _spectrum(x_tr)
            row["invariance_check"] = _invariance_check(
                transform, x_tr, x_te, feats["ws"]["tr"], feats["ws"]["te"])
            row["transform_condition_number"] = getattr(transform, "condition_number", None)
            row["consumer_input_column_rms"] = _column_rms_report(transform(x_tr))
            # The conditioning of the decoder's ACTUAL input (red-team RT2-3): cond(T) and the
            # column RMS do not report it (ZCA mixes directions), and the raw z_spectrum is the
            # pre-transform code. This is the number a FALSIFY reading must be read against.
            row["consumer_input_spectrum"] = _spectrum(transform(x_tr))
            if ctx["rung"] == LINEAR_RUNG and ctx["member"] == "diag":
                row["converged_linear"] = _converged_linear_agreement(
                    transform(x_tr), y_tr, transform(x_te), y_te, action_dim, seed,
                    sched["lbfgs_steps"])
        row["feature_source"] = ctx.get("source")
        row["recondition_member"] = ctx.get("member")
        row["is_probe"] = False
        cell.stamp(row)
    x1010._print_verdict(row)
    return row


# --------------------------------------------------------------------------------------
# THE RUN
# --------------------------------------------------------------------------------------
def _cell(rows: List[Dict[str, Any]], arm_id: str, seed: int) -> Optional[Dict[str, Any]]:
    for r in rows:
        if r.get("arm_id") == arm_id and int(r.get("seed", -1)) == int(seed):
            return r
    return None


def _majority_class(classes: List[str], majority: int) -> str:
    for c in ("CONFIRM", "FALSIFY", "HIGH", "LOW"):
        if sum(1 for x in classes if x == c) >= majority:
            return c
    return "INDETERMINATE"


def run_experiment(seeds: List[int], dry_run: bool = False) -> Dict[str, Any]:
    sched = {
        "zworld_p0": DRY_RUN_ZWORLD_P0 if dry_run else ZWORLD_P0_EPISODES,
        "steps": DRY_RUN_STEPS if dry_run else STEPS_PER_EPISODE,
        "bc_eps": DRY_RUN_BC_EPISODES if dry_run else BC_EPISODES,
        "bc_rand": DRY_RUN_BC_RANDOM_EPISODES if dry_run else BC_RANDOM_EPISODES,
        "passes": DRY_RUN_ADAPTER_PASSES if dry_run else ADAPTER_PASSES,
        "lbfgs_steps": (DRY_RUN_CONVERGED_LINEAR_LBFGS_STEPS if dry_run
                        else CONVERGED_LINEAR_LBFGS_STEPS),
    }
    seeds_sufficient = bool(len(seeds) >= SEED_MAJORITY)
    majority = int(SEED_MAJORITY)
    env_kwargs = x734._env_kwargs_for_rung(RUNG)
    cfg_base = x1002._off_path_config_slice(
        dry_run, sched["zworld_p0"], 0, 0, sched["steps"],
        sched["bc_eps"], sched["bc_rand"], sched["passes"], 0)
    cfg_base["converged_linear_lbfgs_steps"] = int(sched["lbfgs_steps"])
    # Refuse at setup any readiness anchor its own recorded positive control cannot clear
    # (x1023's certificate, imported: the SAME predicates -- PCA-32 majority at the parity bar,
    # raw-field floor -- score the SAME V3-EXQ-1010 reference cells).
    anchor_reachability = _assert_gates_reachable()
    probe_env = x734._make_env(seeds[0], env_kwargs)
    action_dim = int(probe_env.action_dim)

    rows: List[Dict[str, Any]] = []
    per_seed: List[Dict[str, Any]] = []
    for s in seeds:
        # ---- the 1002 dataset, re-collected from its deterministic recipe ---------------
        torch.manual_seed(s)
        np.random.seed(s)
        oracle_eps = x1002._collect_episodes(s, env_kwargs, "oracle", sched["bc_eps"], sched["steps"])
        rand_eps = x1002._collect_episodes(s, env_kwargs, "random", sched["bc_rand"], sched["steps"])
        tr, te = x1002._split_episodes(oracle_eps)
        data = {"train": tr, "test": te, "random": rand_eps}
        f_tr, y_tr = x1002._rawfield_features(tr)
        f_te, y_te = x1002._rawfield_features(te)
        fr_te, yr_te = x1002._rawfield_features(rand_eps)
        w_tr, _ = x1008._world_state_features(tr)
        w_te, _ = x1008._world_state_features(te)
        wr_te, _ = x1008._world_state_features(rand_eps)
        feats = {"y_tr": y_tr, "y_te": y_te, "yr_te": yr_te,
                 "field": {"tr": f_tr, "te": f_te, "r": fr_te},
                 "ws": {"tr": w_tr, "te": w_te, "r": wr_te}}
        frozen: Dict[str, Any] = {}

        # ---- Half A: the two probe cells (SD-106 warm is shared with Half B) ------------
        seed_rows = [run_probe_cell(a, s, sched, env_kwargs, cfg_base, frozen, dry_run)
                     for a in PROBE_ARM_IDS]
        # ---- Half B: the consumer cells -------------------------------------------------
        for a in CONSUMER_ARM_IDS:
            seed_rows.append(run_consumer_cell(a, s, data, feats, frozen, action_dim, sched,
                                               cfg_base, dry_run))
        rows.extend(seed_rows)
        frozen.clear()

        # ---- per-seed readouts -------------------------------------------------------------
        pr = _cell(seed_rows, PROBE_SD106, s) or {}
        po = _cell(seed_rows, PROBE_OFF, s) or {}
        cons = {m: (_cell(seed_rows, _arm_id(_member_track(TRACK_SD106_SENSE, m), CONSUMER_RUNG), s)
                    or {}).get("oracle_action_agreement") for m in MEMBERS}
        cons_path = {m: (_cell(seed_rows, _arm_id(_member_track(TRACK_SD106_PATH, m), CONSUMER_RUNG), s)
                         or {}).get("oracle_action_agreement") for m in MEMBERS}
        lin = {m: (_cell(seed_rows, _arm_id(_member_track(TRACK_SD106_SENSE, m), LINEAR_RUNG), s)
                   or {}).get("oracle_action_agreement") for m in MEMBERS}
        pca_c = (_cell(seed_rows, _arm_id(TRACK_PCA, CONSUMER_RUNG), s) or {}).get("oracle_action_agreement")
        pca_l = _cell(seed_rows, _arm_id(TRACK_PCA, LINEAR_RUNG), s) or {}
        sd_l = _cell(seed_rows, _arm_id(_member_track(TRACK_SD106_SENSE, "diag"), LINEAR_RUNG), s) or {}
        diag = cons.get("diag")
        lifts = {m: (None if (cons.get(m) is None or diag is None) else float(cons[m]) - float(diag))
                 for m in ("zca", "lda")}
        lifts_path = {m: (None if (cons_path.get(m) is None or cons_path.get("diag") is None)
                          else float(cons_path[m]) - float(cons_path["diag"])) for m in ("zca", "lda")}
        # M1 ROUTES ON ZCA ALONE -- the UNSUPERVISED member MECH-566's falsifier names. LDA is
        # fitted on the oracle's own action labels, so a lift it produces would attribute a
        # supervision effect to conditioning (red-team RT2-6); it is recorded as the supervised
        # upper reference, never routed on.
        zca_lift = lifts.get("zca")
        best_lift = zca_lift
        if zca_lift is None:
            m1_class = "INDETERMINATE"
        elif float(zca_lift) >= MECH566_CONFIRM_LIFT:
            m1_class = "CONFIRM"
        elif float(zca_lift) <= -MECH566_HARMFUL_LIFT:
            # a whitened member that HURTS by as much as CONFIRM would help is amplification /
            # a harmful conditioning, not the "no lift" the falsifier names (red-team F4); the
            # band is symmetric with CONFIRM so a merely-null negative cannot be masked (RT2-5)
            m1_class = "HARMFUL"
        elif float(zca_lift) < MECH566_FALSIFY_LIFT:
            m1_class = "FALSIFY"
        else:
            m1_class = "PARTIAL"
        gap = (None if (pca_c is None or diag is None) else float(pca_c) - float(diag))
        gap_closed = (None if (gap is None or best_lift is None or gap <= 0.0)
                      else float(best_lift) / float(gap))
        parity = pr.get("parity_ratio")
        orel = pr.get("overlap_rel")
        p1s = pr.get("p1_score")
        unt_low = pr.get("p1_untrained_reads_low")
        if parity is None or p1s is None or unt_low is None:
            p1_class = "INDETERMINATE"
        elif float(parity) < PARITY_HIGH_FLOOR:
            p1_class = "PARITY_UNMET"
        elif not bool(unt_low):
            p1_class = "DEGENERATE"          # the statistic cannot read LOW even for noise
        elif float(p1s) >= P1_HIGH_FLOOR:
            p1_class = "HIGH"
        elif float(p1s) <= P1_LOW_CEIL:
            p1_class = "LOW"
        else:
            p1_class = "INDETERMINATE"
        # The chip's original statistic, classified for the record only (nothing routes on it).
        if orel is None:
            overlap_class = "INDETERMINATE"
        elif float(orel) >= OVERLAP_HIGH_FLOOR:
            overlap_class = "HIGH"
        elif float(orel) <= OVERLAP_LOW_CEIL:
            overlap_class = "LOW"
        else:
            overlap_class = "INDETERMINATE"
        conv_sd = (sd_l.get("converged_linear") or {}).get("heldout_agreement")
        conv_pca = (pca_l.get("converged_linear") or {}).get("heldout_agreement")
        per_seed.append({
            "seed": int(s),
            # Half A
            "parity_ratio": parity,
            "posthoc_ols_r2": pr.get("posthoc_ols_r2"), "pca32_ols_r2": pr.get("pca32_ols_r2"),
            "overlap_sd106": pr.get("overlap_decoder_range"),
            "overlap_sd106_encoder_domain": pr.get("overlap_encoder_domain"),
            "overlap_off": po.get("overlap_decoder_range"),
            "overlap_ceiling": pr.get("overlap_ceiling_split"),
            "overlap_chance": pr.get("overlap_chance_random"),
            "overlap_rel": orel,
            "overlap_rel_off": (None if (po.get("overlap_decoder_range") is None
                                         or not pr.get("overlap_ceiling_split"))
                                else float(po["overlap_decoder_range"]) / float(pr["overlap_ceiling_split"])),
            "overlap_untrained": (pr.get("untrained_control") or {}).get("overlap_decoder_range"),
            "overlap_rel_untrained": (pr.get("untrained_control") or {}).get("overlap_rel"),
            "overlap_class_recorded_only": overlap_class,
            "parity_untrained": (pr.get("untrained_control") or {}).get("parity_ratio"),
            # THE P1 STATISTIC and its anchors
            "pc_recoverability_sd106": pr.get("pc_recoverability_mean"),
            "pc_recoverability_tail_sd106": pr.get("pc_recoverability_tail_mean"),
            "pc_recoverability_off": po.get("pc_recoverability_mean"),
            "pc_recoverability_untrained": (pr.get("untrained_control") or {}).get("pc_recoverability_mean"),
            "pc_recoverability_tail_untrained": (pr.get("untrained_control") or {}).get("pc_recoverability_tail_mean"),
            "pc_recoverability_random_projection": (pr.get("random_projection_control") or {}).get("pc_recoverability_mean"),
            "p1_score": p1s,
            "p1_untrained_reads_low": unt_low,
            "p1_score_off": po.get("pc_recoverability_mean"),
            "decision_cells_r2_code_mean": pr.get("decision_cells_r2_code_mean"),
            "decision_cells_r2_pca_mean": pr.get("decision_cells_r2_pca_mean"),
            "decision_cells_r2_untrained_mean": (pr.get("untrained_control") or {}).get("decision_cells_r2_mean"),
            "decision_cells_score": pr.get("decision_cells_score"),
            "field_block_r2_code_mean": pr.get("field_block_r2_code_mean"),
            "field_block_r2_pca_mean": pr.get("field_block_r2_pca_mean"),
            "p1_class": p1_class,
            # Half B
            "consumer_agreement_sense": cons, "consumer_agreement_path": cons_path,
            "linear_agreement_sense": lin,
            "pca_consumer_agreement": pca_c,
            "lift_sense": lifts, "lift_path": lifts_path, "best_lift_sense": best_lift,
            "sd106_to_pca_gap": gap, "gap_fraction_closed_by_best_member": gap_closed,
            "m1_class": m1_class,
            "sense_minus_path_diag": (None if (diag is None or cons_path.get("diag") is None)
                                      else float(diag) - float(cons_path["diag"])),
            "converged_linear_sd106_sense": conv_sd, "converged_linear_pca32": conv_pca,
            "converged_linear_gap": (None if (conv_sd is None or conv_pca is None)
                                     else float(conv_pca) - float(conv_sd)),
            "repro_1023a_consumer_diag_delta": (
                None if (diag is None or int(s) not in REF_1023A_CONSUMER_DIAG)
                else abs(float(diag) - REF_1023A_CONSUMER_DIAG[int(s)])),
            "repro_1023a_pca_consumer_delta": (
                None if (pca_c is None or int(s) not in REF_1023A_PCA_CONSUMER)
                else abs(float(pca_c) - REF_1023A_PCA_CONSUMER[int(s)])),
            "invariance_abs_delta_worst": max(
                [float((r.get("invariance_check") or {}).get("abs_delta") or 0.0)
                 for r in seed_rows if r.get("invariance_check")] or [0.0]),
            "sd106_skip_weight_norm": pr.get("skip_weight_norm"),
            "sd106_p0a_n_steps": pr.get("p0a_n_steps"),
            "sd106_participation_ratio": pr.get("z_participation_ratio"),
        })

    # ---- aggregation -------------------------------------------------------------------
    m1_classes = [p["m1_class"] for p in per_seed]
    p1_classes = [p["p1_class"] for p in per_seed]
    m1 = _majority_class(m1_classes, majority)
    for extra in ("PARTIAL", "HARMFUL"):
        if m1 == "INDETERMINATE" and sum(1 for c in m1_classes if c == extra) >= majority:
            m1 = extra
    p1 = _majority_class(p1_classes, majority)
    for extra in ("PARITY_UNMET", "DEGENERATE"):
        if p1 == "INDETERMINATE" and sum(1 for c in p1_classes if c == extra) >= majority:
            p1 = extra

    # ---- preconditions (readiness -- the run cannot mean anything if any is red) --------
    raw_rows = [r for r in rows if r.get("arm_id") == ARM_RAW]
    raw_worst, raw_worst_cell = x1002._worst_cell(raw_rows, "oracle_action_agreement", "min")
    pca_vals = [(p["seed"], p["pca_consumer_agreement"]) for p in per_seed
                if p.get("pca_consumer_agreement") is not None]
    pca_n_clear = sum(1 for _s, v in pca_vals if float(v) >= SD106_PARITY_BAR)
    pca_worst_seed = (min(pca_vals, key=lambda t: float(t[1]))[0] if pca_vals else None)
    skip_norms = [p["sd106_skip_weight_norm"] for p in per_seed if p.get("sd106_skip_weight_norm") is not None]
    prs = [p["sd106_participation_ratio"] for p in per_seed if p.get("sd106_participation_ratio") is not None]
    sd_probe_rows = [r for r in rows if r.get("arm_id") == PROBE_SD106]
    n_changed = None
    for r in sd_probe_rows:
        g = r.get("world_encoder_weight_delta")
        v = g.get("n_world_encoder_changed") if isinstance(g, dict) else None
        if v is not None:
            n_changed = int(v) if n_changed is None else min(n_changed, int(v))
    cap_worst = max([int(r.get("capture_n_delta_abs") or 0) for r in rows if r.get("is_probe")] or [0])
    inv_worst = max([float(p["invariance_abs_delta_worst"]) for p in per_seed] or [0.0])
    steps = [int(p["sd106_p0a_n_steps"]) if p.get("sd106_p0a_n_steps") is not None else 0 for p in per_seed]
    checks = [
        {"name": "instrument_rawfield_control_supra_floor",
         "description": "The raw 25-dim resource field decoded at the consumer's width must clear "
                        "RAW_FIELD_CONTROL_FLOOR, else the dataset/labels are blind.",
         "measured": raw_worst, "threshold": float(RAW_FIELD_CONTROL_FLOOR), "direction": "lower",
         "control": "rawfield_ceiling arm, worst seed (x1002's imported positive control)",
         "offending_cell": raw_worst_cell},
        {"name": "anchor_pca32_reaches_parity_bar_on_majority",
         "description": "PCA-32 at the consumer rung must clear SD106_PARITY_BAR on a seed majority: "
                        "the anchor locates the band MECH-566's CONFIRM reading points toward.",
         "measured": int(pca_n_clear), "threshold": int(majority), "direction": "lower",
         "control": "ws250_pca__mlp128; V3-EXQ-1023a measured 0.8771 / 0.8578 / 0.8702",
         "offending_cell": ("seed%s" % pca_worst_seed if pca_worst_seed is not None else None)},
        {"name": "sd106_bypass_trained_off_zero",
         "description": "SD-106's zero-initialised bypass must have moved: ||W_skip|| > 0 on the worst seed.",
         "measured": (min(skip_norms) if skip_norms else 0.0), "threshold": 1e-8,
         "direction": "lower", "comparator": ">",
         "control": "SD-106 arm's world_encoder_skip weight norm after P0a, worst seed"},
        {"name": "sd106_encoder_trained_in_p0a",
         "description": "The SD-106 z_world path must have CHANGED during P0a (V3-EXQ-783 signature).",
         "measured": (float(n_changed) if n_changed is not None else 0.0), "threshold": 1.0,
         "direction": "lower", "control": "latent_stack_weight_delta n_world_encoder_changed, worst seed"},
        {"name": "sd106_latent_not_collapsed",
         "description": "Participation ratio of the SD-106 code on the P0a buffer >= SD-070's floor.",
         "measured": (float(min(prs)) if prs else 0.0), "threshold": float(PARTICIPATION_RATIO_FLOOR),
         "direction": "lower", "control": "V3-EXQ-1041 measured 17.7 at epochs=40"},
        {"name": "p0a_buffer_captured_exactly",
         "description": "The captured buffer must have EXACTLY the trainer's row count, else the "
                        "probe's split is not the trainer's split (x1041 capture_n_delta gate).",
         "measured": float(cap_worst), "threshold": 0.0, "direction": "upper",
         "control": "abs(len(sink) - trainer.n_buffered), worst probe cell"},
        {"name": "recondition_transforms_invertible",
         "description": "OLS holdout R^2 of world_obs from the transformed code equals that from the "
                        "raw code within INVARIANCE_TOL for every member -- the Half B instrument.",
         "measured": float(inv_worst), "threshold": float(INVARIANCE_TOL), "direction": "upper",
         "control": "worst |R^2(raw) - R^2(transformed)| over every SD-106 consumer cell"},
    ]
    scoped_out: List[Dict[str, Any]] = []
    repro_deltas = [float(p["repro_1023a_consumer_diag_delta"]) for p in per_seed
                    if p.get("repro_1023a_consumer_diag_delta") is not None]
    if dry_run:
        # resolve_p0a_config FORCES epochs=2 under --dry-run (x1023a's one-knob proof), so the
        # step floor is not meaningful for the smoke regime: disposition (a), scoped out. The
        # 1023a reproduction gate likewise cannot apply at dry scale (6 BC episodes, 3 passes).
        scoped_out.append({"name": "sd106_p0a_n_steps_supra_shipped",
                           "applies_note": "dry_run forces epochs=2; floor not meaningful"})
        scoped_out.append({"name": "sd106_consumer_diag_reproduces_1023a",
                           "applies_note": "dry_run data/pass counts cannot reproduce a full-scale cell"})
    else:
        checks.append(
            {"name": "sd106_p0a_n_steps_supra_shipped",
             "description": "P0a optimiser steps on every SD-106 cell must clear P0A_STEP_FLOOR, "
                            "proving the epochs=40 knob reached the trainer (x1023a's gate).",
             "measured": float(min(steps) if steps else 0.0), "threshold": float(P0A_STEP_FLOOR),
             "direction": "lower",
             "control": "V3-EXQ-1041: 1320/1320/1160 at epochs=40; 396/396/348 at epochs=12"})
        checks.append(
            {"name": "sd106_consumer_diag_reproduces_1023a",
             "description": ("The in-run sd106_sense_diag__mlp128 cell (P0a-ONLY warmup) must "
                             "reproduce V3-EXQ-1023a's consumer-rung agreement on every seed to "
                             "within REPRO_TOL: MECH-566's registered 0.05 lift is denominated "
                             "against THAT baseline (0.7273 mean), and a P0a-only baseline that "
                             "drifts from it is a different instrument (red-team F5). Unmet -> "
                             "P0b/P1 DO shape sense-time z_world; re-queue with the full warmup."),
             "measured": float(max(repro_deltas) if repro_deltas else 1.0),
             "threshold": float(REPRO_TOL), "direction": "upper",
             "control": "V3-EXQ-1023a zworld_sd106__mlp128: 0.7542 / 0.7356 / 0.6921 (seeds 42/43/44)"})
    try:
        preconditions = p0_readiness_gate(checks)
        gate_green, gate_reason = True, ""
    except P0NotReady as e:
        preconditions = list(e.preconditions)
        gate_green = False
        gate_reason = "preconditions unmet: " + ", ".join(
            str(p.get("name")) for p in preconditions if not p.get("met"))
    # PER-HALF GATES (red-team RT2-4): each half's direction is voided only by ITS OWN
    # instrument's preconditions. The shared SD-106-warmup checks belong to both.
    HALF_A_ONLY = {"p0a_buffer_captured_exactly"}
    HALF_B_ONLY = {"instrument_rawfield_control_supra_floor",
                   "anchor_pca32_reaches_parity_bar_on_majority",
                   "recondition_transforms_invertible",
                   "sd106_consumer_diag_reproduces_1023a"}
    unmet = {str(p.get("name")) for p in preconditions if not p.get("met")}
    gate_a_green = bool(not (unmet - HALF_B_ONLY))
    gate_b_green = bool(not (unmet - HALF_A_ONLY))

    # ---- non-degeneracy ------------------------------------------------------------------
    # The manipulation must REACH the decoder: at the consumer rung, a whitened member's fitted
    # decoder must not be bit-identical to the diag member's (final CE differs). An identical
    # agreement with an identical CE means the transform never changed the decoder's input --
    # the exact reading a dry-scale run can produce (all arms at the majority-class share).
    reach = []
    for s in seeds:
        ce = {}
        for m in MEMBERS:
            c = _cell(rows, _arm_id(_member_track(TRACK_SD106_SENSE, m), CONSUMER_RUNG), s) or {}
            ce[m] = (c.get("decoder_training") or {}).get("final_ce_loss")
        if all(ce.get(m) is not None for m in MEMBERS):
            reach.append(bool(max(abs(float(ce[m]) - float(ce["diag"])) for m in ("zca", "lda")) > 1e-9))
    manipulation_reaches_decoder = bool(reach and all(reach))
    # P1 must be ABLE to read LOW: the untrained encoder's PC recoverability must sit at or
    # below P1_LOW_CEIL on a seed majority, else a HIGH on the trained code is vacuous.
    unt_low_flags = [p["p1_untrained_reads_low"] for p in per_seed
                     if p.get("p1_untrained_reads_low") is not None]
    p1_control_discriminates = bool(sum(1 for b in unt_low_flags if b) >= majority)
    seps = [p["pc_recoverability_untrained"] for p in per_seed
            if p.get("pc_recoverability_untrained") is not None]
    lift_all = [p["best_lift_sense"] for p in per_seed if p.get("best_lift_sense") is not None]
    overlap_all = [p["p1_score"] for p in per_seed if p.get("p1_score") is not None]
    cond_all = [r.get("transform_condition_number") for r in rows
                if r.get("recondition_member") in ("zca", "lda")
                and r.get("transform_condition_number") is not None]
    transforms_nontrivial = bool(cond_all and max(float(c) for c in cond_all) > 1.0 + 1e-6)
    degeneracy = check_degeneracy({
        "best_lift_sense": lift_all,
        "p1_score": overlap_all,
    })
    # Per-claim non-degeneracy (red-team F2): each half's own controls decide whether ITS
    # criterion is scorable, and both reach the top-level flag the indexer reads.
    p1_non_degenerate = bool(gate_a_green and len(overlap_all) >= majority and p1_control_discriminates)
    m1_non_degenerate = bool(gate_b_green and transforms_nontrivial and manipulation_reaches_decoder
                             and len(lift_all) >= majority)
    if gate_a_green and not p1_control_discriminates and p1 not in ("PARITY_UNMET",):
        p1 = "DEGENERATE"

    # ---- verdict grid: the two halves are INDEPENDENT (red-team F3) ---------------------------
    a_label = {"HIGH": "retained_directions_are_pca32",
               "LOW": "which_directions_confirmed_directly",
               "PARITY_UNMET": "probe_premise_parity_unmet",
               "DEGENERATE": "probe_cannot_read_low"}.get(p1, "recoverability_indeterminate")
    b_label = {"CONFIRM": "mech566_conditioning_confirmed",
               "FALSIFY": "mech566_conditioning_falsified",
               "PARTIAL": "mech566_conditioning_partial",
               "HARMFUL": "mech566_whitening_harmful_or_amplification"}.get(m1, "mech566_indeterminate")
    a_determinate = p1 in ("HIGH", "LOW")
    b_determinate = m1 in ("CONFIRM", "FALSIFY", "PARTIAL")
    if not gate_green:
        label, outcome = "substrate_not_ready_requeue", "FAIL"
    elif not seeds_sufficient:
        label, outcome = "insufficient_seeds_for_majority", "FAIL"
    else:
        label = "%s__%s" % (a_label, b_label)
        outcome = "PASS" if (a_determinate and b_determinate) else "FAIL"

    dir_sd106 = {"LOW": "weakens", "HIGH": "mixed"}.get(p1, "unknown")
    dir_m566 = {"CONFIRM": "supports", "FALSIFY": "weakens", "PARTIAL": "mixed"}.get(m1, "unknown")
    if not gate_a_green or not p1_non_degenerate:
        dir_sd106 = "unknown"
    if not gate_b_green or not m1_non_degenerate:
        dir_m566 = "unknown"

    def _mean(key: str) -> Optional[float]:
        vals = [p[key] for p in per_seed if p.get(key) is not None]
        return float(np.mean(vals)) if vals else None

    flat = {
        "n_seeds": len(seeds), "seed_majority_required": int(majority),
        "gate_green": 1 if gate_green else 0,
        "transforms_nontrivial": 1 if transforms_nontrivial else 0,
        "manipulation_reaches_decoder": 1 if manipulation_reaches_decoder else 0,
        # Half A
        "parity_ratio_mean": _mean("parity_ratio"),
        "parity_high_floor": float(PARITY_HIGH_FLOOR),
        "overlap_sd106_mean": _mean("overlap_sd106"),
        "overlap_off_mean": _mean("overlap_off"),
        "overlap_ceiling_mean": _mean("overlap_ceiling"),
        "overlap_chance_mean": _mean("overlap_chance"),
        "overlap_rel_mean": _mean("overlap_rel"),
        "overlap_rel_off_mean": _mean("overlap_rel_off"),
        "overlap_high_floor": float(OVERLAP_HIGH_FLOOR), "overlap_low_ceil": float(OVERLAP_LOW_CEIL),
        "overlap_rel_untrained_mean": _mean("overlap_rel_untrained"),
        "parity_untrained_mean": _mean("parity_untrained"),
        # THE P1 STATISTIC
        "pc_recoverability_sd106_mean": _mean("pc_recoverability_sd106"),
        "pc_recoverability_tail_sd106_mean": _mean("pc_recoverability_tail_sd106"),
        "pc_recoverability_off_mean": _mean("pc_recoverability_off"),
        "pc_recoverability_untrained_mean": _mean("pc_recoverability_untrained"),
        "pc_recoverability_tail_untrained_mean": _mean("pc_recoverability_tail_untrained"),
        "pc_recoverability_random_projection_mean": _mean("pc_recoverability_random_projection"),
        "pc_recoverability_untrained_max": (max(float(v) for v in seps) if seps else None),
        "n_seeds_untrained_reads_low": sum(1 for b in unt_low_flags if b),
        "p1_score_mean": _mean("p1_score"), "p1_score_off_mean": _mean("p1_score_off"),
        "gate_a_green": 1 if gate_a_green else 0, "gate_b_green": 1 if gate_b_green else 0,
        "p1_high_floor": float(P1_HIGH_FLOOR), "p1_low_ceil": float(P1_LOW_CEIL),
        "n_seeds_p1_high": sum(1 for c in p1_classes if c == "HIGH"),
        "n_seeds_p1_low": sum(1 for c in p1_classes if c == "LOW"),
        "decision_cells_r2_untrained_mean": _mean("decision_cells_r2_untrained_mean"),
        "decision_cells_score_mean": _mean("decision_cells_score"),
        "p1_control_discriminates": 1 if p1_control_discriminates else 0,
        "p1_non_degenerate": 1 if p1_non_degenerate else 0,
        "m1_non_degenerate": 1 if m1_non_degenerate else 0,
        "decision_cells_r2_code_mean": _mean("decision_cells_r2_code_mean"),
        "decision_cells_r2_pca_mean": _mean("decision_cells_r2_pca_mean"),
        "field_block_r2_code_mean": _mean("field_block_r2_code_mean"),
        "field_block_r2_pca_mean": _mean("field_block_r2_pca_mean"),
        # Half B
        "sd106_sense_diag_consumer_mean": _mean_over_dict(per_seed, "consumer_agreement_sense", "diag"),
        "sd106_sense_zca_consumer_mean": _mean_over_dict(per_seed, "consumer_agreement_sense", "zca"),
        "sd106_sense_lda_consumer_mean": _mean_over_dict(per_seed, "consumer_agreement_sense", "lda"),
        "sd106_path_diag_consumer_mean": _mean_over_dict(per_seed, "consumer_agreement_path", "diag"),
        "sd106_path_zca_consumer_mean": _mean_over_dict(per_seed, "consumer_agreement_path", "zca"),
        "sd106_path_lda_consumer_mean": _mean_over_dict(per_seed, "consumer_agreement_path", "lda"),
        "pca_consumer_agreement_mean": _mean("pca_consumer_agreement"),
        "lift_zca_sense_mean": _mean_over_dict(per_seed, "lift_sense", "zca"),
        "lift_lda_sense_mean": _mean_over_dict(per_seed, "lift_sense", "lda"),
        "best_lift_sense_mean": _mean("best_lift_sense"),
        "best_lift_sense_min": (min(lift_all) if lift_all else None),
        "sd106_to_pca_gap_mean": _mean("sd106_to_pca_gap"),
        "gap_fraction_closed_by_best_member_mean": _mean("gap_fraction_closed_by_best_member"),
        "n_seeds_m1_harmful": sum(1 for c in m1_classes if c == "HARMFUL"),
        "n_seeds_m1_partial": sum(1 for c in m1_classes if c == "PARTIAL"),
        "mech566_confirm_lift": float(MECH566_CONFIRM_LIFT),
        "mech566_falsify_lift": float(MECH566_FALSIFY_LIFT),
        "n_seeds_m1_confirm": sum(1 for c in m1_classes if c == "CONFIRM"),
        "n_seeds_m1_falsify": sum(1 for c in m1_classes if c == "FALSIFY"),
        "sense_minus_path_diag_mean": _mean("sense_minus_path_diag"),
        "converged_linear_sd106_sense_mean": _mean("converged_linear_sd106_sense"),
        "converged_linear_pca32_mean": _mean("converged_linear_pca32"),
        "converged_linear_gap_mean": _mean("converged_linear_gap"),
        "repro_1023a_consumer_diag_delta_max": (
            max([p["repro_1023a_consumer_diag_delta"] for p in per_seed
                 if p.get("repro_1023a_consumer_diag_delta") is not None] or [None])
            if any(p.get("repro_1023a_consumer_diag_delta") is not None for p in per_seed) else None),
        "repro_tol": float(REPRO_TOL),
        "invariance_abs_delta_worst": float(inv_worst),
    }
    flat = {k: (int(v) if isinstance(v, bool) else v) for k, v in flat.items()}
    flat = {k: v for k, v in flat.items()
            if v is not None and isinstance(v, (int, float)) and (not isinstance(v, float) or v == v)}

    criteria = [
        {"name": "P1_sd106_pc_recoverability_score_on_seed_majority",
         "load_bearing": True, "claim_id": "SD-106",
         "passed": bool(a_determinate),
         "measured": flat.get("p1_score_mean"),
         "threshold_high": float(P1_HIGH_FLOOR), "threshold_low": float(P1_LOW_CEIL),
         "comparator_high": ">=", "comparator_low": "<=", "direction": "interval",
         "class": p1, "per_seed_class": p1_classes,
         "threshold_note": ("rec = unweighted mean over the 32 train-split PC directions of the "
                            "0-clipped holdout R^2 of each PC score from the SD-106 code; "
                            "ceiling 1.0 exact (PCA-32 coordinates). rec >= threshold_high -> "
                            "HIGH, <= threshold_low -> LOW, between -> indeterminate. 'passed' "
                            "means DETERMINATE (either band on a seed majority), not 'good'. "
                            "Premise: parity_ratio >= %.2f on the same buffer; non-degeneracy: "
                            "the untrained encoder reads <= %.2f on a seed majority."
                            % (PARITY_HIGH_FLOOR, P1_LOW_CEIL))},
        {"name": "M1_mech566_recondition_lift_on_seed_majority",
         "load_bearing": True, "claim_id": "MECH-566",
         "passed": bool(b_determinate),
         "measured": flat.get("best_lift_sense_mean"),
         "threshold": float(MECH566_CONFIRM_LIFT), "threshold_falsify": float(MECH566_FALSIFY_LIFT),
         "class": m1, "per_seed_class": m1_classes,
         "threshold_note": ("ZCA lift over diag at mlp128 (the claim's own unsupervised member; "
                            "LDA is label-supervised and recorded only): >= 0.05 on a seed "
                            "majority -> CONFIRM (supports); in (-0.05, 0.02) -> FALSIFY "
                            "(weakens); <= -0.05 -> HARMFUL (indeterminate for the falsifier); "
                            "else PARTIAL (mixed -- the claim's own grey zone, an acceptable "
                            "terminal reading). 'passed' means DETERMINATE.")},
        {"name": "C_transforms_nontrivial",
         "load_bearing": False,
         "passed": bool(transforms_nontrivial),
         "measured": (max(float(c) for c in cond_all) if cond_all else 0.0), "threshold": 1.0,
         "comparator": ">",
         "threshold_note": "the whitening matrices must not be the identity (condition number > 1)."},
        {"name": "C_p1_can_read_low",
         "load_bearing": False,
         "passed": bool(p1_control_discriminates),
         "measured": (max(float(v) for v in seps) if seps else 1.0),
         "threshold": float(P1_LOW_CEIL), "direction": "upper",
         "threshold_note": ("the untrained encoder's rec must be <= P1_LOW_CEIL on a seed "
                            "majority (worst seed reported): the statistic can read LOW for a "
                            "code that carries nothing. Gates P1's non-degeneracy, not the "
                            "verdict.")},
        {"name": "C_manipulation_reaches_decoder",
         "load_bearing": False,
         "passed": bool(manipulation_reaches_decoder),
         "measured": int(sum(1 for r in reach if r)), "threshold": int(len(seeds)),
         "threshold_note": ("on every seed, a whitened member's consumer-rung decoder fits to a "
                            "different final CE than the diag member's -- the transform reached "
                            "the decoder's input. Gates M1's non-degeneracy, not the verdict.")},
    ]
    return {
        "experiment_type": EXPERIMENT_TYPE,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "claim_ids": list(CLAIM_IDS),
        "sleep_driver_pattern": "none",
        "outcome": outcome,
        "evidence_direction": ("mixed" if (dir_sd106 != "unknown" or dir_m566 != "unknown") else "unknown"),
        "evidence_direction_per_claim": {"SD-106": dir_sd106, "MECH-566": dir_m566},
        "evidence_direction_note": (
            "SD-106: 'weakens' iff the retained directions are NOT the PCA-32 directions (LOW), "
            "'mixed' iff they are (HIGH -- re-opens the which-directions closure). MECH-566: "
            "supports / weakens / mixed per M1. diagnostic-purpose: excluded from confidence "
            "scoring; the directions are the routing signal for /failure-autopsy."),
        "readout": flat,
        "per_seed_results": per_seed,
        "arm_results": rows,
        "interpretation": {
            "label": label,
            "half_a_class": p1, "half_b_class": m1,
            "preconditions": preconditions,
            "scoped_out": scoped_out,
            "criteria": criteria,
            "combination_rule": ("P1 (SD-106) and M1 (MECH-566) are INDEPENDENT load-bearing "
                                 "criteria on separate claims; outcome PASS iff the gate is green "
                                 "AND both are determinate. The label concatenates the two "
                                 "readings. Neither criterion's verdict, direction or "
                                 "non-degeneracy depends on the other's: a Half-A premise miss "
                                 "or degeneracy leaves MECH-566's direction intact, and vice "
                                 "versa (red-team F3)."),
            "criteria_non_degenerate": {
                "P1_sd106_pc_recoverability_score_on_seed_majority": p1_non_degenerate,
                "M1_mech566_recondition_lift_on_seed_majority": m1_non_degenerate,
                "C_transforms_nontrivial": bool(len(cond_all) > 0),
                "C_p1_can_read_low": bool(len(seps) > 0),
                "C_manipulation_reaches_decoder": bool(len(reach) > 0),
            },
            "gate_reason": gate_reason,
            "anchor_reachability": anchor_reachability,
            "pre_registered_reading_table": {
                "HIGH_parity_LOW_score": "which_directions_confirmed_directly -> /failure-autopsy -> "
                                         "SD-106 objective-choice decision (user-held)",
                "HIGH_parity_HIGH_score": "retained_directions_are_pca32 -> RE-OPENS the 1023a "
                                          "elimination; Half B becomes the load-bearing reading "
                                          "(MECH-566 / MECH-567 territory); autopsy the pair",
                "HIGH_parity_between": "recoverability_indeterminate -> recorded, no routing",
                "parity_below_floor": "probe_premise_parity_unmet -> reproduction failure vs V3-EXQ-1041 "
                                      "(Half A only; Half B unaffected)",
                "untrained_reads_above_low": "probe_cannot_read_low -> P1 degenerate, SD-106 "
                                             "unknown; decision-cell readouts carry the question",
                "M1_CONFIRM": "MECH-566 supports -> the fix is a fixed reconditioning stage, cheap",
                "M1_FALSIFY": "MECH-566 weakens -> conditioning exonerated; route to MECH-567 "
                              "(target-carrying head) / content explanation",
                "M1_PARTIAL": "MECH-566 mixed -- the claim's own grey zone; an acceptable terminal "
                              "reading (V3-EXQ-1008 measured 5 of 6 OFF-code lifts in it)",
                "M1_HARMFUL": "whitening hurt the consumer -> amplification or harmful conditioning; "
                              "read consumer_input_column_rms before believing either; MECH-566 unknown",
            },
            "posthoc_disclosure": ("All framing of V3-EXQ-1023a's OUTCOME in this driver is "
                                   "post-hoc; only P1 and M1 are pre-registered here."),
            "null_reading": ("A P1 LOW confirms which-directions directly; a P1 HIGH re-opens it. "
                             "An M1 FALSIFY exonerates conditioning on THIS code; it says nothing "
                             "about a differently-trained code. Route to /failure-autopsy."),
        },
        "non_degenerate": bool(p1_non_degenerate and m1_non_degenerate
                               and degeneracy.get("non_degenerate", True)),
        "non_degenerate_per_claim": {
            "SD-106": bool(p1_non_degenerate and degeneracy.get("non_degenerate", True)),
            "MECH-566": bool(m1_non_degenerate and degeneracy.get("non_degenerate", True)),
        },
        "degeneracy_reason": (None if (p1_non_degenerate and m1_non_degenerate
                                       and degeneracy.get("non_degenerate", True))
                              else ("preconditions unmet" if not gate_green
                                    else "; ".join(x for x in (
                                        ("P1 cannot read LOW (untrained encoder above P1_LOW_CEIL)" if not p1_control_discriminates else ""),
                                        ("whitening transforms are the identity" if not transforms_nontrivial else ""),
                                        ("transform never reached the decoder" if not manipulation_reaches_decoder else ""),
                                        (degeneracy.get("degeneracy_reason") or ""),
                                    ) if x))),
        "degenerate_metrics": degeneracy.get("degenerate_metrics"),
        "prior_evidence_context": {
            "v3_exq_1023a_consumer_diag": REF_1023A_CONSUMER_DIAG,
            "v3_exq_1023a_pca_consumer": REF_1023A_PCA_CONSUMER,
            "v3_exq_1041_parity_ratio_epochs40": REF_1041_PARITY_RATIO_EPOCHS40,
            "v3_exq_1008_off_code_zca_lift": {"42": 0.0396, "43": 0.0296, "44": 0.0483},
            "v3_exq_1008_off_code_lda_lift": {"42": 0.0424, "43": 0.0334, "44": 0.0524},
            "standardisation_already_in_baseline": True,
        },
        "off_arm_budget": {"probe_off_epochs": int(SHIPPED_P0A_EPOCHS), "sd106_epochs": int(P0A_EPOCHS),
                           "budget_matched": False,
                           "note": "the OFF probe cell is the shipped-budget negative control for "
                                   "the overlap floor; nothing routes on an ON-minus-OFF quantity."},
        "ethics_preflight": {
            "involves_negative_valence": False, "involves_suffering_like_state": False,
            "involves_self_model": False, "involves_inescapability_or_helplessness": False,
            "involves_offline_replay_over_harm": False, "involves_social_mind_or_language": False,
            "involves_human_data_or_clinical_context": False, "decision": "allow",
        },
    }


def _mean_over_dict(per_seed: List[Dict[str, Any]], key: str, sub: str) -> Optional[float]:
    vals = [p[key][sub] for p in per_seed
            if isinstance(p.get(key), dict) and p[key].get(sub) is not None]
    return float(np.mean(vals)) if vals else None


def _run_self_test() -> int:
    """Design-time arithmetic: the pre-registered bands are satisfiable and ordered, the
    thresholds are the claim's own, and the transform family is invertible on a random code."""
    assert OVERLAP_LOW_CEIL < OVERLAP_HIGH_FLOOR, "bands must be ordered"
    assert 0.0 < P1_LOW_CEIL < P1_HIGH_FLOOR <= 1.0, "P1 bands must be ordered inside [0, 1]"
    assert 0.0 < MECH566_FALSIFY_LIFT < MECH566_CONFIRM_LIFT <= MECH566_HARMFUL_LIFT, "MECH-566 bands must be ordered"
    # 0-clipped per-column R^2: a column predicted WORSE than the train mean reads 0, never < 0.
    yt = torch.randn(100, 3); yp = yt + 10.0
    r2 = _r2_per_column(yt, yp, yt.mean(dim=0, keepdim=True))
    assert bool((r2 >= 0.0).all()), r2
    # Relative-ridge ZCA: a direction with eigenvalue far below the ridge is DAMPED, not amplified.
    reset_all_rng(1)
    zz = torch.randn(500, 32) * torch.tensor([1.0] * 16 + [1e-5] * 16)
    tz = _ZCAWhitenRel(zz)
    rms = _column_rms_report(tz(zz))
    # no direction is AMPLIFIED above unit scale; the near-null half is damped toward zero
    assert rms["rms_max"] < 1.5 and rms["rms_min"] < 1e-2, rms
    # ...and the invertibility check is BLIND to conditioning (it is an instrument for
    # invertibility only), demonstrated on a deliberately ill-conditioned diagonal map.
    class _Bad:
        def __call__(self, x): return x * torch.tensor([1.0] * 16 + [1e6] * 16)
    chk = _invariance_check(_Bad(), zz[:400], zz[400:], torch.randn(500, 250)[:400], torch.randn(500, 250)[400:])
    assert chk["abs_delta"] is not None and chk["abs_delta"] < 1e-2, chk
    # The premise floor must be REACHABLE by the recorded epochs=40 cell (0.980) and must not
    # be so loose that a collapsed code (V3-EXQ-1041's OFF cells sit far lower) reads as HIGH.
    assert PARITY_HIGH_FLOOR < REF_1041_PARITY_RATIO_EPOCHS40, "premise floor must sit below 1041's measured 0.980"
    assert PARITY_HIGH_FLOOR >= 0.90, "premise floor must not admit a collapsed code as HIGH parity"
    assert int(P0A_CONFIG.epochs) == 40 and float(P0A_CONFIG.preservation_weight) == 200.0
    reset_all_rng(0)
    z = torch.randn(400, 32) @ torch.randn(32, 32) * 0.1
    y = torch.randint(0, 5, (400,))
    ws = torch.randn(400, 250)
    for m in MEMBERS:
        t = _make_transform(m, z[:300], y[:300], 5)
        chk = _invariance_check(t, z[:300], z[300:], ws[:300], ws[300:])
        assert chk["abs_delta"] is not None and chk["abs_delta"] < INVARIANCE_TOL, (m, chk)
    U = torch.randn(250, 32)
    pa = principal_angles(U, U)
    assert abs(pa["mean_squared_cosine_overlap"] - 1.0) < 1e-9
    assert len(ARM_IDS) == 17 and len(set(ARM_IDS)) == 17
    print("self-test OK: bands ordered, P0A_CONFIG epochs=40/pw=200, %d members invertible, "
          "principal_angles identity=1, %d arms" % (len(MEMBERS), len(ARM_IDS)), flush=True)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="*", default=None)
    args = ap.parse_args()

    if args.self_test:
        sys.exit(_run_self_test())

    t0 = time.perf_counter()
    seeds = args.seeds if args.seeds else (DRY_RUN_SEEDS if args.dry_run else list(SEEDS))
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
        "warmup_phases": "p0a_only",
        "steps_per_episode": (DRY_RUN_STEPS if args.dry_run else STEPS_PER_EPISODE),
        "bc_episodes": (DRY_RUN_BC_EPISODES if args.dry_run else BC_EPISODES),
        "bc_random_episodes": (DRY_RUN_BC_RANDOM_EPISODES if args.dry_run else BC_RANDOM_EPISODES),
        "bc_train_frac": x1002.BC_TRAIN_FRAC,
        "adapter_passes": (DRY_RUN_ADAPTER_PASSES if args.dry_run else ADAPTER_PASSES),
        "adapter_batch": ADAPTER_BATCH, "adapter_lr": ADAPTER_LR,
        "sd106_p0a_epochs": int(P0A_EPOCHS), "off_probe_p0a_epochs": int(SHIPPED_P0A_EPOCHS),
        "sd106_p0a_step_floor": int(P0A_STEP_FLOOR),
        "sd106_preservation_weight": float(PRESERVATION_WEIGHT),
        "sd106_use_world_encoder_skip": bool(USE_WORLD_ENCODER_SKIP),
        "sd106_parity_bar": float(SD106_PARITY_BAR),
        "projection_dim": int(PROJECTION_DIM),
        "rungs": list(RUNGS), "members": list(MEMBERS),
        "feature_sources": [TRACK_SD106_SENSE, TRACK_SD106_PATH],
        "parity_high_floor": float(PARITY_HIGH_FLOOR),
        "p1_high_floor": float(P1_HIGH_FLOOR), "p1_low_ceil": float(P1_LOW_CEIL),
        "pc_tail_start": int(PC_TAIL_START),
        "mech566_harmful_lift": float(MECH566_HARMFUL_LIFT),
        "m1_routes_on_member": "zca",
        "overlap_high_floor": float(OVERLAP_HIGH_FLOOR), "overlap_low_ceil": float(OVERLAP_LOW_CEIL),
        "zca_ridge_frac": float(ZCA_RIDGE_FRAC),
        "mech566_confirm_lift": float(MECH566_CONFIRM_LIFT),
        "mech566_falsify_lift": float(MECH566_FALSIFY_LIFT),
        "invariance_tol": float(INVARIANCE_TOL), "repro_tol": float(REPRO_TOL),
        "converged_linear": {"l2": float(CONVERGED_LINEAR_L2),
                             "lbfgs_steps": (DRY_RUN_CONVERGED_LINEAR_LBFGS_STEPS if args.dry_run
                                             else CONVERGED_LINEAR_LBFGS_STEPS),
                             "max_iter": int(CONVERGED_LINEAR_LBFGS_MAX_ITER)},
        "raw_field_control_floor": float(RAW_FIELD_CONTROL_FLOOR),
        "seed_majority": int(SEED_MAJORITY), "arms": list(ARM_IDS),
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
