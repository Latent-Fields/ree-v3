#!/opt/local/bin/python3
"""
V3-EXQ-1084: SD-PP-B9 H-pe-source-structurally-wrong probe -- does an
ALTERNATIVE harm-PE source (ensemble disagreement) separate exposure-matched
world surprise better than the converged residual-head PE and its
innovation-variance form do?

STATUS 2026-09-24: **NOT QUEUED -- DO NOT QUEUE AS-IS. Step 4.5 red-team (fable) BLOCKING.**
Recovered from the stranded design subagent of governance-20260924-workset (session
f6e2a1cf, died on a usage limit before red-team/queue) by orch0924-stranded. The
load-bearing alternative ENS_DISAGREE = across-member variance of h_k(z(t-1), a(t-1))
NEVER reads z(t); the world_rule_shift changes only the law (z, a) -> z(t), so the
only path from the manipulation to this DV is the input marginal (z(t-1), a(t-1)) --
which the propensity strata exist to remove -- and ARM_EXPOSED_CONTROL (eps 0.8
random executed actions) is built to RAISE input novelty. `better` is unreachable
by construction and `not_better` near-guaranteed, so the declared-null label
("alternative tracks harm level") would be misattributed. Verified on the pre-queue
probe (seed 42, probe scale, run BEFORE this review; thresholds untouched since):
sep_matched(ENS) = -0.306 (vs stationary alone -0.105, vs exposed -0.402),
d_base_hi95 = -0.25; ENS medians ~3e-8 (shift) / 4e-8 (stationary) / 1.8e-7
(exposed), five orders below the per-step displacement. Also: CANARY_LABEL is
tautologically 0.5 (y scored against y), so label_canary certifies nothing; the
level canary balances only the 4 propensity covariates (worst 0.047 vs tol 0.05 at
probe scale) and one of them, ||z(t)||, is post-treatment; the oracle can pass on
the control arm's randomised action distribution rather than the law change.
Full report: queue-experiment red-team 2026-09-24 (orch0924-stranded); decision on
how to re-specify the alternative (an ensemble PE that reads z(t), the fast
proximity-field input-level PE the 1077 autopsy also names, or recording the leg as
resolved-by-construction) is a USER/GOVERNANCE decision -- GFLAG raised. V3-EXQ-1084
was never queued and never ran: the id is unburned.

experiment_purpose: diagnostic
SLEEP DRIVER: not_applicable (no sleep loop, SWS, REM or aggregation cluster is enabled)
red-team (Step 4.5, fable): BLOCKING -- see STATUS above and RED_TEAM_VERDICT below.

WHAT THIS IS, AND WHAT IT IS NOT
--------------------------------
The readout-axis leg of the GOV-FANOUT-1 four-leg discrimination on
SD-PP-B9-harm-forward-below-persistence-baseline (REE_assembly
evidence/planning/substrate_queue.json, fanout_recommendation), from
failure_autopsy_V3-EXQ-1062a_2026-09-23 (confirmed; ratified by governance
cycle governance-20260923-0717, rec-20260923-10bca6b5). The registered sketch:

    "score an ensemble-disagreement or innovation-variance PE source against the
     residual-head PE on the same ticks; declared null: the alternative is no
     better separated from the VALENCE_HARM level than the residual head is"

The H-harm-head-undertrained leg ALREADY RAN (V3-EXQ-1077, 2026-09-23,
active_error_removed_AMBIGUOUS): the online head reproduced 1062a's defect
(persistence_dominated 3/3), and a decayed-optimiser head trained to a held-out
plateau on the same data converged on 3/3 seeds and came out cannot_determine
vs persistence (d = -0.0006 / +0.0023 / +0.0048). Under-training is therefore
NOT re-proposed here -- the ensemble members below reuse 1077's validated fitter
verbatim, so every "converged" head in this run is 1077's converged head or an
independently initialised twin of it.

It is NOT a re-run of MECH-055's falsifier. The MECH-055 re-derive brake fired
at N=2 and refuses any V3-EXQ-1062b; the autopsy's refused_requeue_scope exempts
fan-out legs that are a new EXQ number, a different mechanism and a different
DV with no claim tag. claim_ids = [] ; bears_on = ["mech055_harm_pe_source_validity"].

THE QUESTION
------------
The dACC harm PE (dacc._affective_pe) is pe = ||z_harm_a(t) - e2_harm_a(z_harm_a(t-1), a(t-1))||,
times a precision factor borrowed from E3. A PE source is useful to MECH-055
axis 3 only if it carries WORLD SURPRISE -- here, the post-training
world_rule_shift (V3-EXQ-1062a's lever: action -> displacement map re-permuted
every 10 world steps, depth 2, onset at the P1->P2 boundary) -- and not merely
the HARM LEVEL the agent is exposed to. 1062a could not ask this because the
shift arm's harm exposure ran 3.28x the stationary arm's (the SD-PP-B9
unblocks_caveat, harm_exposure_relative_deviation_bounded). This run removes
that confound by EXPOSURE MATCHING and then compares PE sources on the SAME
matched ticks.

DESIGN
------
Seeds 42 / 137 / 2026 (1062a's and 1077's). Per seed, ONE shared collection:
  P0 (30 eps) + P1 (60 eps) x 90 steps: V3-EXQ-1077's _collect, imported, which
      is 1062a's P0/P1 exactly (online e2_harm_a, batch 1, lr 5e-4, eps 0.1)
      plus banking of every P1 transition and the online loss curve.
  ENSEMBLE: K = 5 harm-forward heads (the agent's own E2HarmAForward class),
      trained IDENTICALLY on the banked P1 buffer with 1077's decayed-lr
      best-checkpoint fitter (_fit_to_plateau, imported): same data, same split,
      same minibatch stream; they differ ONLY in initialisation. Member 0 is the
      agent's own pre-P1 init, i.e. EXACTLY 1077's ARM_CONVERGED head; members
      1..4 are re-initialised under a pinned forked RNG. Driver-local: this is
      NOT SD-PP-B11 and builds no production reliability module.
  Three no-grad P2 rollouts from ONE deep-copied post-training snapshot, RNG
      re-seeded identically at each start, P2_STEP_BUDGET env steps each:
        ARM_STATIONARY       world_rule_shift off, eps 0.0 (1062a's P2 condition)
        ARM_EXPOSED_CONTROL  world_rule_shift off, eps 0.8 on the EXECUTED action
                             under the CANONICAL map. The agent is misdirected and
                             walks into hazards, but the world's transition law is
                             unchanged and the head is fed the executed action: harm
                             exposure and (z, a) novelty rise WITHOUT world surprise.
                             This is the DESIGN-LEVEL exposure match (the smoke
                             measured the 1062a-style exposure deviation of the shift
                             arm at +4.3 vs stationary but -0.48 vs this arm).
        ARM_SHIFT            1062a's lever enabled at the P2 boundary (interval 10,
                             depth 2), eps 0.0 -- the world-surprise manipulation.
      The two control arms are pooled as the y = 0 population. Every executed
      transition (z_harm_a(t-1), a(t-1), z_harm_a(t)) is recorded with its harm
      covariates; no PE source feeds back into behaviour, so every source is
      scored on the IDENTICAL transitions. There is NO behavioural DV: under the
      1062a config the harm PE cannot change an action (candidate_effort is
      uniform, agent.py ~7789), so the DV is a readout by construction.

PE SOURCES (all computed on the same ticks, from POST-override executed pairs)
------------------------------------------------------------------------------
  PERSIST        REFERENCE. ||z(t) - z(t-1)|| -- the trivial persistence
                 predictor's residual (raw harm-latent displacement).
  RESID_CONV     BASELINE. ||z(t) - h_conv(z(t-1), a(t-1))|| on 1077's converged
                 head (member 0): dacc._affective_pe's residual on the best-trained
                 residual head. The online head is NEVER a baseline -- 1077 showed
                 its PE is model error.
  INNOV_CONV     BASELINE. Standardised innovation of the same residual,
                 sqrt(mean_d e_d^2 / var_d), var_d a running per-dim innovation
                 variance (EMA rate INNOV_RHO) read BEFORE it absorbs e,
                 initialised from the head's held-out P1 residual variance. Once a
                 head sits at persistence its residual ~= the raw innovation, so
                 this is a NAMED BASELINE, not an independent alternative (1077
                 autopsy design constraint 1).
  ENS_DISAGREE   THE ALTERNATIVE. Mean over dims of the across-member variance of
                 h_k(z(t-1), a(t-1)) -- epistemic disagreement; it never sees z(t).
  recorded only  RESID_ENSMEAN (ensemble-mean residual), RESID_ONLINE,
                 INNOV_ONLINE, RESID_ONLINE_PW (x (1 + min(e3.current_precision /
                 5000, 3)), the precision-weighted dACC form).
Every source is computed by the driver from the transition it recorded, whose
action is the one EXECUTED (post epsilon override). The agent's internal dACC PE
rolls e2_harm_a forward on the PRE-override action and is never read.

EXPOSURE MATCHING (tick level, on top of the design-level control arm)
----------------------------------------------------------------------
Eligible ticks: a transition inside an episode, >= BURN_IN (= harm_history_len,
10) steps after reset (the harm-history refill transient -- the shift and
exposed arms have ~10x more episode starts, so this is not optional), in the
shift arm only under a NON-canonical action map, in the controls only under the
canonical one. Harm covariates per tick: harm_exposure (body[10], 1062a's
exposure variable), the per-tick INCREMENT of VALENCE_HARM_DISCRIMINATIVE (1062a
routed on the increment; the level is a monotone accumulator ramp),
||z_harm_a(t)||, and log1p(steps since reset). A logistic propensity model
P(shift | covariates) is fitted on the pooled eligible ticks; ticks outside
common support are trimmed; N_STRATA propensity strata are cut at shift-arm
quantiles; strata with fewer than MIN_PER_ARM_STRATUM ticks on either side are
dropped. All comparisons are WITHIN strata, weighted by shift-tick count (ATT).
The bootstrap below resamples ticks with the propensity fit and strata held fixed
(conditional on the matching), so its CIs omit propensity-model uncertainty --
slightly anti-conservative, stated rather than hidden; the level canaries are what
certify the matching itself.

THE SEPARATION STATISTIC -- relative to persistence, no absolute R2 anywhere
----------------------------------------------------------------------------
  sep(s)  = stratified AUC(source s: shift ticks vs matched control ticks) - 0.5
            (signed: a surprise signal must be HIGHER under the shift). AUC is
            rank-based and scale-free, so no floor on any source's magnitude.
  D_per   = sep(ENS_DISAGREE) - sep(PERSIST)                  (vs persistence)
  D_base  = sep(ENS_DISAGREE) - max(sep(RESID_CONV), sep(INNOV_CONV))  (vs baselines)
CIs by a moving-block bootstrap within each arm (block BOOT_BLOCK ticks, N_BOOT
replicates, pinned seed), PAIRED across sources (same resample). Per seed:
  better      one-sided 95% LOWER bounds of D_base AND D_per both > 0.
  not_better  one-sided 95% UPPER bound of D_base < NULL_MARGIN (0.05 AUC units):
              a meaningful improvement over the baselines is excluded (TOST-style
              equivalence bound, not a zero-margin bar).
  cannot_determine otherwise.

PRE-REGISTERED NULL -- built so it can win cleanly
--------------------------------------------------
The null: the alternative is no better separated from the VALENCE_HARM /
harm-exposure level than the residual head is, i.e. once harm level is matched it
carries no more world surprise. A source that merely tracks harm level has sep ~
0 within strata -- which the level-only canaries below verify ON THE SAME
STATISTIC -- so under the null D_base ~ 0 and not_better fires with room to spare.

PRECONDITIONS (a failure routes requeue-not-verdict, never a verdict)
---------------------------------------------------------------------
  instrument_canary_persistence_gate   persistence_skill_gate.check_canary (the
                                       shipped verdict path reproduces 1062a's 6
                                       recorded cells persistence_dominated)
  reproduction_control_seeds           the ONLINE head is persistence_dominated on
                                       the stationary rollout on >= 2 seeds (the
                                       residual-head defect reproduces; this is a
                                       regime check, not a comparison baseline)
  label_canary_sep_worst_seed          a synthetic source equal to the arm label
                                       scores sep >= 0.45 (analytic 0.5): the
                                       stratified statistic can DETECT
  exposure_match_level_canary          each harm covariate, scored AS IF it were
                                       a PE source, has |sep| <= 0.05 (analytic 0
                                       under perfect matching): the STATED
                                       EXPOSURE-MATCH TOLERANCE, measured on the
                                       very statistic the criteria route on
  common_support_fraction              >= 0.5 of shift ticks survive matching
  matched_rows_worst_arm               >= MIN_MATCHED matched ticks on each side
  ensemble_ood_disagreement_ratio      median disagreement on OOD-perturbed inputs
                                       / on held-out in-distribution inputs >= 1.5
                                       (positive control: the ensemble CAN signal
                                       novelty) AND all member pairs distinct
  ensemble_cross_tick_spread           IQR / median of ENS_DISAGREE over the eval
                                       ticks >= 0.05: the alternative varies across
                                       the ticks its AUC ranks (a constant source
                                       would hand the null a degenerate win)
  baseline_head_converged              ensemble member 0 (the RESID_CONV / INNOV_CONV
                                       baseline) meets 1077's convergence rule
                                       (schedule exhausted before cap AND best
                                       held-out loss <= persistence). Added at
                                       recovery review 2026-09-24: the baseline is
                                       NAMED "converged" and 1077 gated on it, so
                                       this run verifies it instead of assuming it
  shift_fired / controls_canonical     the lever fired, and never in a control
  event_detectable_oracle              a cross-fitted logistic classifier over the
                                       FULL transition (z(t-1), a(t-1), dz, dz x a)
                                       separates the shift arm from the matched
                                       controls within the SAME strata (one-sided
                                       95% lower bound of sep > 0). If NOTHING
                                       computed from the transition can separate
                                       them, no PE source can, and the PE-source
                                       contrast is undefined -- not negative.

PRE-REGISTERED ROUTING (quorum >= 2 of 3 seeds; per-seed statuses counted only
on ELIGIBLE seeds = all per-seed readiness preconditions met)
-------------------------------------------------------------------------------
  persistence-gate canary fails, or < 2 seeds reproduce the defect, or < 2
    eligible seeds                     -> substrate_not_ready_requeue
                                          (precondition failure: requeue, not a verdict)
  < 2 eligible seeds with a detectable event
                                       -> event_not_detectable_cannot_determine
                                          (no PE source over this stream can
                                          separate this lever's surprise from
                                          matched harm level; consistent with H1 /
                                          the representation ceiling; do NOT
                                          requeue this design as-is)
  ENS_DISAGREE `better` on >= 2 detectable seeds
                                       -> pe_source_leg_supported (PASS)
                                          route: candidate predictive-cost source
                                          for SD-PP-B9 / SD-032b; NEEDS GOVERNANCE
                                          (SD-PP-B11 stays registration-only)
  ENS_DISAGREE `not_better` on >= 2 detectable seeds
                                       -> declared_null_alternative_tracks_harm_level
                                          route: next proposal = the
                                          H-harm-head-representation-ceiling probe
                                          (vary harm_history_len, measure per-step
                                          z_harm_a displacement, each setting fitted
                                          with 1077's converged fitter); explicitly
                                          NOT more training
  otherwise                            -> cannot_determine_no_quorum
                                          (requeue-not-verdict: more P2 budget / seeds)
Only pe_source_leg_supported is outcome PASS. evidence_direction is
non_contributory (claim-free diagnostic).

RECORDED, NON-GATING
--------------------
Every source's matched sep (and vs each control arm separately), unmatched AUCs,
and the fraction of each source's raw separation explained by harm level;
|Spearman| of every source with the VALENCE_HARM increment (1062a's C1 redundancy
statistic), the VALENCE_HARM level, harm_exposure and ||z_harm_a||; covariate
SMDs before/after matching and the 1062a-style harm-exposure relative deviation
unmatched (vs each control) and matched; persistence verdicts for the online head,
the ensemble mean and each member on every arm; the online loss curve and every
member's train / held-out / lr curve (Recording Standard 3b/3c); per-tick source
and covariate arrays for every eligible tick. From the V3-EXQ-1077 autopsy's
design constraints 3 and 4, on the stationary AND shift arms' own eval rows:
1077's in-sample best-fit bound + permuted-delta twin and an EPISODE-SPLIT
held-out fit ('nothing learnable' vs 'learnable but not learned'), and the
action sensitivity of the converged head, the ensemble mean and the online head
(1062a's action-swap instrument, normalised by the predicted delta AND by the
true per-step displacement).

WHY NOT REANALYSIS (Step 2.4, GOV-REUSE-1): reanalysis_query scanned 1073
manifests -- 0 carry an ensemble-disagreement, innovation or harm-PE-source
readout, and neither 1062a nor 1077 recorded per-tick transitions, so no
alternative source can be recomputed post hoc. Needs a new run.

KNOWN LIMITATIONS (Step 2.5c): degrading WARNs as V3-EXQ-1077 (SD-PP-B9 itself
-- the subject -- SD-018, SD-106, SD-091, SD-ZWORLD-SENSE-PATH-PARITY, SD-PP-B4,
sd061-resume-progress-ecology, SD-MECH303-THRESHOLD-SOURCING,
mech005-betagate-decommit-counter-and-commit-ceiling). One CORRUPTING entry's
module is exercised incidentally: contextmemory-write-path-addressing-degeneracy
(e1_deep.py::ContextMemory.write is called in agent.sense). Disposition: the DV
never reads ContextMemory; it changes WHICH transitions occur, identically for
every PE source scored on them, so it cannot favour one source over another.
Recorded, not blocking. The other open corrupting entries (MECH-320 tonic
vigour, blocked_agency, SD-105 entropy floor, SD-PP-B5 use_world_interventional)
are default-OFF paths this config does not enable.
"""

import argparse
import copy
import hashlib
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F

from ree_core.agent import REEAgent

from experiment_protocol import emit_outcome
from experiments.pack_writer import write_flat_manifest
from experiments._lib.arm_fingerprint import arm_cell
from experiments._lib.persistence_skill_gate import (
    MIN_ROWS, check_canary, format_verdict, persistence_verdict)
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator

# 1062a's builders and 1077's collection + fitter are IMPORTED, not copied: the
# training regime is 1062a's and every converged head is 1077's by construction.
from experiments import v3_exq_1062a_mech055_affect_channel_separation_postshift as B
from experiments import v3_exq_1077_sdppb9_harm_head_undertrain_probe as C

DRIVER_1062A = Path(B.__file__).resolve()
DRIVER_1077 = Path(C.__file__).resolve()

EXPERIMENT_TYPE = "v3_exq_1084_sdppb9_harm_pe_source_probe"
EXPERIMENT_PURPOSE = "diagnostic"
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"
QUEUE_ID = "V3-EXQ-1084"
CLAIM_IDS: List[str] = []
BEARS_ON = ["mech055_harm_pe_source_validity"]
VALIDATES_SUBSTRATE = "SD-PP-B9-harm-forward-below-persistence-baseline"
FANOUT_LEG = "H-pe-source-structurally-wrong"
RED_TEAM_VERDICT = ("red-team (fable, 2026-09-24): BLOCKING -- ENS_DISAGREE never reads "
                    "z(t), so the transition-law shift reaches it only via the input "
                    "marginal the matching removes; `better` unreachable by construction. "
                    "NOT QUEUED.")

# Both anchor preconditions are computed by SHIPPED instrument code: the
# persistence-gate canary IS persistence_skill_gate.check_canary, and the
# reproduction control counts persistence_verdict statuses -- the path that
# canary pins to persistence_dominated on 1062a's recorded cells.
ANCHOR_REACHABILITY_EXEMPT = (
    "anchors are the shipped check_canary / persistence_verdict path, not a "
    "re-implementation; the canary pins that path to persistence_dominated on "
    "1062a's recorded control cells, so the reproduction anchor is reachable")

# The DV's achievable range is established BY CONSTRUCTION, per seed, on the
# same statistic: sep lives in [-0.5, 0.5] and the label canary must reach
# >= 0.45 on the SAME matched ticks and strata (analytic 0.5), so headroom for
# both D_base and D_per is certified in-run rather than assumed. The
# ensemble-spread floor is a non-degeneracy check (IQR/median of a non-negative
# variance), not a DV bar; the smoke measured it well above 0.05.
CRITERION_ACHIEVABLE_RANGE_EXEMPT = (
    "sep headroom certified in-run by the label canary on the same matched ticks "
    "(label_canary_sep_worst_seed >= 0.45, analytic 0.5); the ENS spread floor is a "
    "non-degeneracy check on a non-negative variance, not a DV bar")

# ---- schedule: 1062a's training, 1077's collection ---------------------------
SEEDS = [42, 137, 2026]
P0_EPS = B.P0_EPS                      # 30
P1_EPS = B.P1_EPS                      # 60
TOTAL_TRAINING_EPS = P0_EPS + P1_EPS   # the [train] ep N/M denominator (90)
STEPS_PER_EPISODE = B.STEPS_PER_EPISODE  # 90
P2_STEP_BUDGET = 2700                  # per arm (1.5x 1062a's 1800: sample size for a clean null)
EPSILON_EVAL = 0.0                     # 1062a's P2 condition (stationary + shift arms)
EXPOSED_EPSILON = 0.8                  # exposed-control arm: ~4/5 of 1062a's lever's misdirection rate
SHIFT_INTERVAL = 10                    # 1062a's ARM_2_HIGH_SHIFT dose, unchanged
ROLLOUT_SEED_OFFSET = 100003           # 1077's: identical RNG stream at each rollout start
BURN_IN = 10                           # == harm_history_len: history refill transient

# ---- ensemble ----------------------------------------------------------------
ENSEMBLE_K = 5
ENSEMBLE_INIT_OFFSET = 7727            # forked-RNG init seed for members 1..K-1
OOD_SCALE = 3.0                        # OOD probe: z + 3 * per-dim std * N(0,1)
OOD_SEED_OFFSET = 31337
OOD_RATIO_MIN = 1.5                    # positive control: disagreement rises off-distribution
ENS_SPREAD_MIN = 0.05                  # IQR / median of ENS_DISAGREE over the eligible eval ticks:
                                       # a tick-constant source would score AUC 0.5 and let the
                                       # null win for a degenerate reason

# ---- innovation-variance source -----------------------------------------------
INNOV_RHO = 0.01                       # EMA rate of the running per-dim innovation variance
VAR_FLOOR = 1e-12

# ---- matching + statistic ----------------------------------------------------
N_STRATA = 10
MIN_PER_ARM_STRATUM = 15
PROPENSITY_RIDGE = 1e-3
MIN_MATCHED = 200                      # matched ticks per arm, per seed
SUPPORT_FRACTION_MIN = 0.5
LEVEL_CANARY_TOL = 0.05                # |sep| of a pure harm covariate: the exposure-match tolerance
LABEL_CANARY_MIN = 0.45                # sep of the arm label itself (analytic 0.5)
N_BOOT = 1000
BOOT_BLOCK = 20
BOOT_SEED = 20260924
ALPHA = 0.05
N_ALTERNATIVES = 1                     # ENS_DISAGREE only (coordinator constraint 1, 1077 autopsy)
NULL_MARGIN = 0.05                     # AUC units; excluded improvement for `not_better`
ORACLE_FOLD_BLOCK = 100
ORACLE_L2 = 1e-2
ORACLE_MAX_ITER = 200

SEEDS_REQUIRED = 2

ARM_STAT = "ARM_STATIONARY"
ARM_EXPO = "ARM_EXPOSED_CONTROL"
ARM_SHIFT = "ARM_SHIFT"
# (arm_id, world_rule_shift on?, P2 epsilon). The two CONTROL arms are pooled as
# y = 0: ARM_EXPOSED_CONTROL raises harm exposure by misdirecting the agent with
# epsilon-random EXECUTED actions under the CANONICAL map -- the head sees the
# executed action, so the world's transition law is unchanged (no world surprise)
# while exposure and (z, a) novelty rise. It is the design-level exposure match;
# the propensity strata are the tick-level one.
ARMS = [(ARM_STAT, False, EPSILON_EVAL), (ARM_EXPO, False, EXPOSED_EPSILON),
        (ARM_SHIFT, True, EPSILON_EVAL)]
# ROLES (fixed by the V3-EXQ-1077 autopsy's design constraints):
#   ALTERNATIVE  ENS_DISAGREE -- the one discriminating alternative.
#   BASELINES    RESID_CONV (1077's converged head = ensemble member 0) and
#                INNOV_CONV (innovation-variance on that same residual). Once a
#                head sits at persistence its residual ~= the raw innovation, so
#                innovation-variance is NOT an independent alternative: it is a
#                named baseline the alternative must beat.
#   REFERENCE    PERSIST (D_per).
#   RECORDED     RESID_ONLINE / RESID_ONLINE_PW / INNOV_ONLINE (the online head is
#                model error per 1077 and is NEVER a comparison baseline) and
#                RESID_ENSMEAN (the ensemble-mean residual).
ALTERNATIVES = ["ENS_DISAGREE"]
BASELINES = ["RESID_CONV", "INNOV_CONV"]
SOURCES = ["PERSIST", "RESID_CONV", "INNOV_CONV", "ENS_DISAGREE",
           "RESID_ENSMEAN", "RESID_ONLINE", "INNOV_ONLINE", "RESID_ONLINE_PW"]
COVARIATES = ["harm_exposure", "valence_harm_increment", "z_harm_a_norm",
              "log1p_steps_since_reset"]

_ZG = ZGoalStreamAccumulator()


def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _config_slice() -> Dict[str, Any]:
    return {
        "base_config": "v3_exq_1062a make_config/_make_env + v3_exq_1077 _collect/_fit_to_plateau (imported)",
        "base_driver_1062a_sha256": _sha256(DRIVER_1062A),
        "base_driver_1077_sha256": _sha256(DRIVER_1077),
        "env": dict(B.ENV_KWARGS),
        "schedule": {"p0": P0_EPS, "p1": P1_EPS, "steps": STEPS_PER_EPISODE,
                     "epsilon_train": B.EPSILON_TRAIN, "epsilon_eval": EPSILON_EVAL,
                     "arms": [[a, sh, e] for a, sh, e in ARMS],
                     "p2_step_budget_per_arm": P2_STEP_BUDGET,
                     "rollout_seed_offset": ROLLOUT_SEED_OFFSET, "burn_in": BURN_IN},
        "shift_arm": {"interval": SHIFT_INTERVAL, "depth": B.WORLD_RULE_SHIFT_DEPTH,
                      "onset": "p2_boundary_post_training"},
        "ensemble": {"k": ENSEMBLE_K, "init_offset": ENSEMBLE_INIT_OFFSET,
                     "member0": "agent pre-P1 init (== V3-EXQ-1077 ARM_CONVERGED); the RESID_CONV / INNOV_CONV baseline head",
                     "fitter": "V3-EXQ-1077 _fit_to_plateau",
                     "batch": C.CONV_BATCH, "lr": C.CONV_LR,
                     "heldout_fraction": C.HELDOUT_FRACTION,
                     "plateau_window": C.PLATEAU_WINDOW, "min_lr": C.MIN_LR,
                     "cap": C.CONV_CAP, "split_seed_offset": C.SPLIT_SEED_OFFSET,
                     "ood_scale": OOD_SCALE, "ood_seed_offset": OOD_SEED_OFFSET,
                     "ood_ratio_min": OOD_RATIO_MIN, "ens_spread_min": ENS_SPREAD_MIN},
        "innovation": {"rho": INNOV_RHO, "var_floor": VAR_FLOOR,
                       "init": "per-dim mean squared residual on the P1 held-out rows"},
        "matching": {"covariates": COVARIATES, "n_strata": N_STRATA,
                     "min_per_arm_stratum": MIN_PER_ARM_STRATUM,
                     "propensity_ridge": PROPENSITY_RIDGE, "weights": "ATT (shift-tick count)"},
        "statistic": {"n_boot": N_BOOT, "boot_block": BOOT_BLOCK, "boot_seed": BOOT_SEED,
                      "alpha": ALPHA, "n_alternatives": N_ALTERNATIVES,
                      "null_margin": NULL_MARGIN, "alternatives": ALTERNATIVES,
                      "baselines": BASELINES, "reference": "PERSIST",
                      "sources": SOURCES},
        "thresholds": {"min_matched": MIN_MATCHED, "support_fraction_min": SUPPORT_FRACTION_MIN,
                       "level_canary_tol": LEVEL_CANARY_TOL,
                       "label_canary_min": LABEL_CANARY_MIN,
                       "seeds_required": SEEDS_REQUIRED},
        "oracle": {"fold_block": ORACLE_FOLD_BLOCK, "l2": ORACLE_L2,
                   "max_iter": ORACLE_MAX_ITER,
                   "features": "z(t-1), a(t-1) one-hot, dz, dz x a"},
    }


def _scale(dry: Optional[str]) -> Dict[str, Any]:
    """Pre-registered scale for a real run; reduced scales exist ONLY for the
    local --dry-run smoke ('smoke') and the pre-queue satisfiability probe
    ('probe'). A real run never scales a threshold."""
    if dry is None:
        return dict(p0=P0_EPS, p1=P1_EPS, spe=STEPS_PER_EPISODE, budget=P2_STEP_BUDGET,
                    cap=C.CONV_CAP, window=C.PLATEAU_WINDOW, every=C.EVAL_EVERY,
                    n_boot=N_BOOT, n_strata=N_STRATA, min_stratum=MIN_PER_ARM_STRATUM,
                    min_matched=MIN_MATCHED, seeds=list(SEEDS), quorum=SEEDS_REQUIRED)
    if dry == "probe":
        return dict(p0=10, p1=20, spe=STEPS_PER_EPISODE, budget=900,
                    cap=C.CONV_CAP, window=C.PLATEAU_WINDOW, every=C.EVAL_EVERY,
                    n_boot=400, n_strata=N_STRATA, min_stratum=MIN_PER_ARM_STRATUM,
                    min_matched=int(MIN_MATCHED * 900 / P2_STEP_BUDGET),
                    seeds=SEEDS[:1], quorum=1)
    return dict(p0=2, p1=3, spe=30, budget=240, cap=600, window=100, every=20,
                n_boot=100, n_strata=4, min_stratum=3, min_matched=15,
                seeds=SEEDS[:1], quorum=1)


# --------------------------------------------------------------------------- #
# Ensemble                                                                     #
# --------------------------------------------------------------------------- #

def _make_member(template: torch.nn.Module, init_state: Dict[str, torch.Tensor],
                 k: int, seed: int) -> torch.nn.Module:
    head = copy.deepcopy(template)
    head.load_state_dict(init_state)
    if k > 0:
        # Forked so the global torch stream (and therefore every later draw in
        # this cell) is untouched by how many members are initialised.
        with torch.random.fork_rng():
            torch.manual_seed(seed * 1009 + ENSEMBLE_INIT_OFFSET + k)
            for m in head.modules():
                if m is not head and hasattr(m, "reset_parameters"):
                    m.reset_parameters()
    return head


def _heldout_idx(n: int, seed: int) -> torch.Tensor:
    """The fitter's held-out rows, reproduced exactly (1077's split)."""
    g = torch.Generator().manual_seed(seed + C.SPLIT_SEED_OFFSET)
    perm = torch.randperm(n, generator=g)
    return perm[:int(round(C.HELDOUT_FRACTION * n))]


def _train_ensemble(template, init_state, col, seed, sc) -> Tuple[List[torch.nn.Module], List[Dict]]:
    members, fits = [], []
    for k in range(ENSEMBLE_K):
        head = _make_member(template, init_state, k, seed)
        if k > 0:
            same = all(torch.equal(head.state_dict()[n], init_state[n]) for n in init_state)
            assert not same, "ensemble member %d init is identical to member 0" % k
        gen = torch.Generator().manual_seed(seed + C.SPLIT_SEED_OFFSET)
        fit = C._fit_to_plateau(head, col["prev"], col["act"], col["next"], gen,
                                C.HELDOUT_FRACTION, sc["cap"], sc["window"], sc["every"])
        head.eval()
        print("  [ensemble] seed=%d member=%d converged=%s stop=%d decays=%d "
              "heldout/persistence=%.4f"
              % (seed, k, fit["converged"], fit["stop_step"], fit["n_lr_decays"],
                 fit["eval_loss_over_persistence"]), flush=True)
        members.append(head)
        fits.append(fit)
    return members, fits


def _ens_predict(members, prev, act) -> torch.Tensor:
    with torch.no_grad():
        return torch.stack([m(prev, act) for m in members])   # [K, n, D]


# --------------------------------------------------------------------------- #
# P2 rollout from the shared post-training snapshot                            #
# --------------------------------------------------------------------------- #

def _rollout(agent0, env0, seed: int, budget: int, shift: bool,
             epsilon: float) -> Dict[str, Any]:
    # hippocampal._rng defaults to the `random` MODULE itself; the copies must
    # SHARE it (1077's memo), and every rollout re-seeds the global streams.
    memo = {id(random): random, id(np.random): np.random}
    agent = copy.deepcopy(agent0, memo)
    env = copy.deepcopy(env0, dict(memo))
    canonical = dict(env.ACTIONS)
    map_canonical_at_onset = dict(env._action_map) == canonical
    if shift:
        B._enable_post_training_shift(env, SHIFT_INTERVAL)
    C._seed_all(seed + ROLLOUT_SEED_OFFSET)
    vh_idx = B.VALENCE_HARM_DISCRIMINATIVE
    rec: Dict[str, List[Any]] = {k: [] for k in (
        "prev", "act", "next", "he", "vh", "dvh", "k", "t", "ep", "shifts",
        "permuted", "prec")}
    total, n_eps = 0, 0
    with torch.no_grad():
        while total < budget:
            agent.reset()
            _obs, od = env.reset()
            n_eps += 1
            prev_zha: Optional[torch.Tensor] = None
            prev_action: Optional[torch.Tensor] = None
            prev_vh: Optional[float] = None
            k = 0
            while total < budget:
                body, world, harm, harm_a, hh = B._obs_tensors(od)
                latent = agent.sense(obs_body=body, obs_world=world, obs_harm=harm,
                                     obs_harm_a=harm_a, obs_harm_history=hh)
                ticks = agent.clock.advance()
                e1_prior = (agent._e1_tick(latent) if ticks.get("e1_tick")
                            else torch.zeros(1, B.WORLD_DIM, device=agent.device))
                candidates = agent.generate_trajectories(latent, e1_prior, ticks)
                action = agent.select_action(candidates, ticks)
                if epsilon > 0.0 and random.random() < epsilon:
                    ai = random.randint(0, env.action_dim - 1)
                    action = torch.zeros(1, env.action_dim, device=agent.device)
                    action[0, ai] = 1.0
                vh = float(agent.residue_field.evaluate_valence(
                    latent.z_world).reshape(-1)[vh_idx].item())
                he = float(body[0, B.IDX_HARM_EXPOSURE])
                # A transition (z(t-1), a(t-1)) -> z(t), recorded with the
                # executed action (not this tick's): the pairing the agent's own
                # dACC PE uses (1062a P1 note).
                if (prev_zha is not None and prev_action is not None
                        and latent.z_harm_a is not None and prev_vh is not None):
                    rec["prev"].append(prev_zha.cpu())
                    rec["act"].append(prev_action.cpu())
                    rec["next"].append(latent.z_harm_a.detach().cpu().clone())
                    rec["he"].append(he)
                    rec["vh"].append(vh)
                    rec["dvh"].append(vh - prev_vh)
                    rec["k"].append(k)
                    rec["t"].append(total)
                    rec["ep"].append(n_eps)
                    rec["shifts"].append(int(env._world_rule_shift_count))
                    rec["permuted"].append(bool(dict(env._action_map) != canonical))
                    rec["prec"].append(float(getattr(agent.e3, "current_precision",
                                                     float("nan"))))
                B._drive_valence_write_paths(agent, body)
                _obs, harm_signal, done, _info, od = env.step(
                    int(action.argmax(dim=-1).item()))
                agent.update_residue(float(harm_signal) if harm_signal is not None else 0.0)
                prev_zha = (latent.z_harm_a.detach().clone()
                            if latent.z_harm_a is not None else None)
                prev_action = action.detach().clone()
                prev_vh = vh
                total += 1
                k += 1
                if done:
                    break
    _ZG.observe(agent)
    cat = (lambda xs, d: torch.cat(xs) if xs else torch.zeros(0, d))
    return {
        "prev": cat(rec["prev"], B.HARM_A_DIM), "act": cat(rec["act"], env.action_dim),
        "next": cat(rec["next"], B.HARM_A_DIM),
        "he": np.asarray(rec["he"], dtype=float), "vh": np.asarray(rec["vh"], dtype=float),
        "dvh": np.asarray(rec["dvh"], dtype=float), "k": np.asarray(rec["k"], dtype=float),
        "t": np.asarray(rec["t"], dtype=float), "ep": np.asarray(rec["ep"], dtype=float),
        "shifts": np.asarray(rec["shifts"], dtype=float),
        "permuted": np.asarray(rec["permuted"], dtype=bool),
        "prec": np.asarray(rec["prec"], dtype=float),
        "n_steps": total, "n_episodes": n_eps,
        "n_world_rule_shifts": int(env._world_rule_shift_count),
        "action_map_canonical_at_onset": bool(map_canonical_at_onset),
        "n_action_map_entries_permuted_at_end": int(sum(
            1 for a, v in env._action_map.items() if canonical.get(a) != v)),
    }


# --------------------------------------------------------------------------- #
# PE sources                                                                   #
# --------------------------------------------------------------------------- #

def _innovation(resid: torch.Tensor, var0: torch.Tensor) -> np.ndarray:
    """Standardised innovation, read BEFORE the running variance absorbs e."""
    var = var0.clone()
    out = np.empty(int(resid.shape[0]), dtype=float)
    for i in range(int(resid.shape[0])):
        e2 = resid[i] ** 2
        out[i] = float(torch.sqrt((e2 / (var + VAR_FLOOR)).mean()).item())
        var = (1.0 - INNOV_RHO) * var + INNOV_RHO * e2
    return out


def _sources(ro: Dict[str, Any], online, members, var0_on, var0_cv) -> Dict[str, np.ndarray]:
    """Every source from the POST-override EXECUTED transition (z(t-1), a(t-1),
    z(t)) the driver recorded -- never the agent's internal dACC PE, whose
    e2_harm_a roll-forward uses the PRE-override action (1077 autopsy constraint
    6). The studied arms (stationary, shift) run at eps 0.0 anyway."""
    prev, act, nxt = ro["prev"], ro["act"], ro["next"]
    with torch.no_grad():
        p_on = online(prev, act) if prev.shape[0] else prev
        stack = (_ens_predict(members, prev, act) if prev.shape[0]
                 else prev.unsqueeze(0).expand(len(members), -1, -1))
        p_cv = stack[0]            # member 0 == V3-EXQ-1077's converged head
        p_em = stack.mean(0)
        r_on = nxt - p_on
        r_cv = nxt - p_cv
        prec_norm = np.minimum(ro["prec"] / B.DACC_PRECISION_SCALE, 3.0)
        prec_norm = np.where(np.isfinite(prec_norm), prec_norm, 0.0)
        return {
            "PERSIST": (nxt - prev).norm(dim=-1).numpy().astype(float),
            "RESID_ONLINE": r_on.norm(dim=-1).numpy().astype(float),
            "RESID_CONV": r_cv.norm(dim=-1).numpy().astype(float),
            "RESID_ENSMEAN": (nxt - p_em).norm(dim=-1).numpy().astype(float),
            "ENS_DISAGREE": stack.var(0, unbiased=False).mean(-1).numpy().astype(float),
            "INNOV_ONLINE": _innovation(r_on, var0_on),
            "INNOV_CONV": _innovation(r_cv, var0_cv),
            "RESID_ONLINE_PW": (r_on.norm(dim=-1).numpy().astype(float)
                                * (1.0 + prec_norm)),
            "_pred_online": p_on, "_pred_conv": p_cv, "_pred_ensmean": p_em,
            "_stack": stack,
        }


# --------------------------------------------------------------------------- #
# Recorded, NON-GATING: learnability + action sensitivity (1077 autopsy 3, 4)   #
# --------------------------------------------------------------------------- #

def _learnability(ro: Dict[str, Any], el: np.ndarray, template, init_state,
                  seed: int, sc: Dict[str, Any]) -> Dict[str, Any]:
    """1077's in-sample best-fit bound + permuted-delta twin, plus an
    EPISODE-SPLIT held-out fit, on this arm's own eval rows. Distinguishes
    'nothing learnable' (real ~ permuted, held-out cannot_determine) from
    'learnable but not learned' (real separates from permuted and the held-out
    split is ready). Every fit is 1077's _fit_to_plateau from the agent's pre-P1
    init. NEVER gates anything."""
    elt = torch.as_tensor(el)
    prev, act, nxt = ro["prev"][elt], ro["act"][elt], ro["next"][elt]
    ep = ro["ep"][el]
    n = int(prev.shape[0])
    if n < MIN_ROWS:
        return {"skipped": "fewer than %d eval rows" % MIN_ROWS}
    out: Dict[str, Any] = {"n_rows": n}
    pg = torch.Generator().manual_seed(seed + C.PERMUTE_SEED_OFFSET)
    pidx = torch.randperm(n, generator=pg)
    t_perm = prev + (nxt - prev)[pidx]
    for bname, tgt in (("in_sample_real", nxt), ("in_sample_permuted_delta", t_perm)):
        h = copy.deepcopy(template)
        h.load_state_dict(init_state)
        og = torch.Generator().manual_seed(seed + C.SPLIT_SEED_OFFSET + 1)
        fit = C._fit_to_plateau(h, prev, act, tgt, og, 0.0, sc["cap"], sc["window"], sc["every"])
        with torch.no_grad():
            op = h(prev, act)
        out[bname] = {"schedule_exhausted": fit["schedule_exhausted"],
                      "stop_step": fit["stop_step"],
                      "loss_over_persistence": fit["eval_loss_over_persistence"],
                      "verdict": C._verdict_dict(persistence_verdict(op, tgt, prev))}
    rv = out["in_sample_real"]["verdict"]
    pv = out["in_sample_permuted_delta"]["verdict"]
    out["real_separates_from_permuted"] = bool(
        rv.get("ci_low") is not None and pv.get("ci_high") is not None
        and rv["ci_low"] > pv["ci_high"])
    # Episode-split held-out fit: train on even-ranked episodes, score odd ones.
    uniq = sorted(set(int(x) for x in ep))
    tr_eps = set(uniq[0::2])
    tr = torch.as_tensor(np.array([int(x) in tr_eps for x in ep], dtype=bool))
    te = ~tr
    if int(tr.sum()) >= MIN_ROWS and int(te.sum()) >= MIN_ROWS:
        h = copy.deepcopy(template)
        h.load_state_dict(init_state)
        og = torch.Generator().manual_seed(seed + C.SPLIT_SEED_OFFSET + 2)
        fit = C._fit_to_plateau(h, prev[tr], act[tr], nxt[tr], og, 0.0,
                                sc["cap"], sc["window"], sc["every"])
        with torch.no_grad():
            op = h(prev[te], act[te])
        out["episode_split_heldout"] = {
            "n_train_rows": int(tr.sum()), "n_heldout_rows": int(te.sum()),
            "n_train_episodes": len(tr_eps), "n_heldout_episodes": len(uniq) - len(tr_eps),
            "schedule_exhausted": fit["schedule_exhausted"], "stop_step": fit["stop_step"],
            "verdict": C._verdict_dict(persistence_verdict(op, nxt[te], prev[te]))}
    else:
        out["episode_split_heldout"] = {"skipped": "too few rows on one side of the split",
                                        "n_train_rows": int(tr.sum()),
                                        "n_heldout_rows": int(te.sum())}
    return out


def _action_sensitivity(f, prev: torch.Tensor, act: torch.Tensor, nxt: torch.Tensor,
                        map_keys: List[int]) -> Dict[str, Any]:
    """1062a's H3 instrument on a given head: hold z(t-1), swap ONLY the action for
    a deterministic different map key. Normalised two ways: by the head's own
    predicted delta (1062a's form) and by the TRUE per-step displacement."""
    n = int(prev.shape[0])
    if n == 0:
        return {"n_rows": 0}
    taken = act.argmax(dim=-1).tolist()
    cf = [map_keys[(map_keys.index(t) + 1) % len(map_keys)] if t in map_keys else map_keys[0]
          for t in taken]
    a_cf = torch.zeros_like(act)
    a_cf[torch.arange(n), torch.as_tensor(cf)] = 1.0
    with torch.no_grad():
        p = f(prev, act)
        pcf = f(prev, a_cf)
    num = float((p - pcf).norm(dim=-1).mean())
    den_pred = float((p - prev).norm(dim=-1).mean())
    den_true = float((nxt - prev).norm(dim=-1).mean())
    return {"n_rows": n, "mean_action_swap_shift": num,
            "mean_predicted_delta": den_pred, "mean_true_displacement": den_true,
            "ratio_to_predicted_delta_1062a_form": (num / den_pred if den_pred > 0 else None),
            "ratio_to_true_displacement": (num / den_true if den_true > 0 else None)}


# --------------------------------------------------------------------------- #
# Statistics                                                                   #
# --------------------------------------------------------------------------- #

def _auc(pos: np.ndarray, neg: np.ndarray) -> float:
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    sn = np.sort(neg)
    lo = np.searchsorted(sn, pos, side="left")
    hi = np.searchsorted(sn, pos, side="right")
    return float((lo.sum() + 0.5 * (hi - lo).sum()) / (pos.size * neg.size))


def _strat_sep(v: np.ndarray, y: np.ndarray, s: np.ndarray, n_strata: int) -> float:
    num, den = 0.0, 0.0
    for j in range(n_strata):
        m = s == j
        pos = v[m & y]
        neg = v[m & ~y]
        if pos.size == 0 or neg.size == 0:
            continue
        num += pos.size * _auc(pos, neg)
        den += pos.size
    return num / den - 0.5 if den > 0 else float("nan")


def _rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(x.size, dtype=float)
    xs = x[order]
    i = 0
    while i < x.size:
        j = i
        while j + 1 < x.size and xs[j + 1] == xs[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 10 or a.size != b.size:
        return float("nan")
    ra, rb = _rankdata(a), _rankdata(b)
    if ra.std() < 1e-12 or rb.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def _propensity(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, List[float]]:
    mu = X.mean(0)
    sd = X.std(0) + 1e-12
    Z = np.column_stack([np.ones(X.shape[0]), (X - mu) / sd])
    beta = np.zeros(Z.shape[1])
    yf = y.astype(float)
    for _ in range(100):
        p = 1.0 / (1.0 + np.exp(-np.clip(Z @ beta, -30, 30)))
        W = p * (1.0 - p)
        H = Z.T @ (Z * W[:, None]) + PROPENSITY_RIDGE * np.eye(Z.shape[1])
        g = Z.T @ (yf - p) - PROPENSITY_RIDGE * beta
        step = np.linalg.solve(H, g)
        beta = beta + step
        if float(np.abs(step).max()) < 1e-9:
            break
    p = 1.0 / (1.0 + np.exp(-np.clip(Z @ beta, -30, 30)))
    return p, [float(b) for b in beta]


def _match(p: np.ndarray, y: np.ndarray, n_strata: int, min_stratum: int
           ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Common-support trim + propensity strata cut at shift-arm quantiles.
    Returns (matched_mask, stratum_id, info)."""
    lo = max(float(p[y].min()), float(p[~y].min()))
    hi = min(float(p[y].max()), float(p[~y].max()))
    keep = (p >= lo) & (p <= hi)
    s = np.full(p.size, -1, dtype=int)
    if (keep & y).sum() == 0 or (keep & ~y).sum() == 0:
        return np.zeros(p.size, dtype=bool), s, {"support_lo": lo, "support_hi": hi,
                                                   "strata": [], "n_dropped_strata": n_strata}
    edges = np.quantile(p[keep & y], np.linspace(0.0, 1.0, n_strata + 1))
    s_all = np.clip(np.searchsorted(edges[1:-1], p, side="right"), 0, n_strata - 1)
    s[keep] = s_all[keep]
    matched = keep.copy()
    strata_info = []
    dropped = 0
    for j in range(n_strata):
        nj_s = int(((s == j) & y).sum())
        nj_c = int(((s == j) & ~y).sum())
        ok = nj_s >= min_stratum and nj_c >= min_stratum
        if not ok:
            matched &= ~(s == j)
            dropped += 1
        strata_info.append({"stratum": j, "n_shift": nj_s, "n_stationary": nj_c,
                            "kept": ok})
    s[~matched] = -1
    return matched, s, {"support_lo": lo, "support_hi": hi, "strata": strata_info,
                        "n_dropped_strata": dropped, "edges": [float(e) for e in edges]}


def _block_resample(n: int, rng: np.random.Generator, block: int) -> np.ndarray:
    if n <= block:
        return rng.integers(0, n, size=n)
    nb = int(math.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=nb)
    return (starts[:, None] + np.arange(block)[None, :]).reshape(-1)[:n]


def _smd(c: np.ndarray, y: np.ndarray, s: Optional[np.ndarray], n_strata: int) -> float:
    sd = math.sqrt((float(c[y].var()) + float(c[~y].var())) / 2.0) + 1e-12
    if s is None:
        return float((c[y].mean() - c[~y].mean()) / sd)
    ms, mc, ws = 0.0, 0.0, 0.0
    for j in range(n_strata):
        a = c[(s == j) & y]
        b = c[(s == j) & ~y]
        if a.size == 0 or b.size == 0:
            continue
        ms += a.size * a.mean()
        mc += a.size * b.mean()
        ws += a.size
    return float((ms - mc) / ws / sd) if ws > 0 else float("nan")


def _matched_rel_dev(c: np.ndarray, y: np.ndarray, s: np.ndarray, n_strata: int) -> float:
    """1062a-style (shift - stationary) / stationary on harm exposure, ATT-weighted."""
    ms, mc, ws = 0.0, 0.0, 0.0
    for j in range(n_strata):
        a = c[(s == j) & y]
        b = c[(s == j) & ~y]
        if a.size == 0 or b.size == 0:
            continue
        ms += a.size * a.mean()
        mc += a.size * b.mean()
        ws += a.size
    if ws <= 0 or abs(mc) < 1e-12:
        return float("nan")
    return float((ms - mc) / abs(mc))


def _oracle_scores(feat: np.ndarray, y: np.ndarray, fold: np.ndarray) -> np.ndarray:
    """Cross-fitted L2 logistic classifier of the arm label from the transition."""
    out = np.full(y.size, np.nan)
    for f in (0, 1):
        tr = fold != f
        te = fold == f
        if te.sum() == 0 or tr.sum() < 20 or y[tr].all() or (~y[tr]).all():
            continue
        mu = feat[tr].mean(0)
        sd = feat[tr].std(0) + 1e-8
        Xtr = torch.tensor((feat[tr] - mu) / sd, dtype=torch.float64)
        ytr = torch.tensor(y[tr].astype(float), dtype=torch.float64)
        w = torch.zeros(Xtr.shape[1], dtype=torch.float64, requires_grad=True)
        b = torch.zeros(1, dtype=torch.float64, requires_grad=True)
        opt = torch.optim.LBFGS([w, b], max_iter=ORACLE_MAX_ITER,
                                line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            loss = (F.binary_cross_entropy_with_logits(Xtr @ w + b, ytr)
                    + ORACLE_L2 * (w * w).sum())
            loss.backward()
            return loss
        opt.step(closure)
        Xte = torch.tensor((feat[te] - mu) / sd, dtype=torch.float64)
        with torch.no_grad():
            out[te] = (Xte @ w + b).numpy()
    return out


def _q(a: np.ndarray, q: float) -> float:
    a = a[np.isfinite(a)]
    return float(np.quantile(a, q)) if a.size else float("nan")


# --------------------------------------------------------------------------- #
# One seed                                                                     #
# --------------------------------------------------------------------------- #

def _eligible(arm: str, ro: Dict[str, Any]) -> np.ndarray:
    """Inside an episode, past the harm-history refill transient; the shift arm
    only under a NON-canonical map (surprise ticks), the controls only under the
    canonical one (asserted by construction: their lever is never enabled)."""
    base = ro["k"] >= BURN_IN
    return (base & ro["permuted"]) if arm == ARM_SHIFT else (base & ~ro["permuted"])


def _cov(ro: Dict[str, Any]) -> np.ndarray:
    return np.column_stack([ro["he"], ro["dvh"], ro["next"].norm(dim=-1).numpy(),
                            np.log1p(ro["k"])])


def _feats(ro: Dict[str, Any], el: np.ndarray) -> np.ndarray:
    elt = torch.as_tensor(el)
    prev = ro["prev"][elt].numpy()
    act = ro["act"][elt].numpy()
    dz = (ro["next"][elt] - ro["prev"][elt]).numpy()
    dza = (dz[:, :, None] * act[:, None, :]).reshape(dz.shape[0], -1)
    return np.column_stack([prev, act, dz, dza])


ORDER = [ARM_SHIFT, ARM_STAT, ARM_EXPO]   # arm_code = index; y = (arm == ARM_SHIFT)


def _analyse(ros: Dict[str, Dict[str, Any]], srcs: Dict[str, Dict[str, np.ndarray]],
             seed: int, sc: Dict[str, Any]) -> Dict[str, Any]:
    n_strata = sc["n_strata"]
    els = {a: _eligible(a, ros[a]) for a in ORDER}
    Xp, yp, cp, pp = [], [], [], []
    for i, a in enumerate(ORDER):
        Xa = _cov(ros[a])[els[a]]
        Xp.append(Xa)
        yp.append(np.full(Xa.shape[0], a == ARM_SHIFT))
        cp.append(np.full(Xa.shape[0], i))
        pp.append(np.arange(Xa.shape[0]))
    X = np.vstack(Xp)
    y = np.concatenate(yp).astype(bool)
    arm_code = np.concatenate(cp)
    pos_in_arm = np.concatenate(pp)
    vals: Dict[str, np.ndarray] = {}
    for name in SOURCES:
        vals[name] = np.concatenate([srcs[a][name][els[a]] for a in ORDER])
    for i, cname in enumerate(COVARIATES):
        vals["CANARY_LEVEL_" + cname] = X[:, i].copy()
    vals["CANARY_LABEL"] = y.astype(float)
    info: Dict[str, Any] = {
        "n_eligible_by_arm": {a: int(els[a].sum()) for a in ORDER},
        "n_eligible_shift": int(y.sum()), "n_eligible_control": int((~y).sum()),
        "n_control_ticks_under_permuted_map": int(sum(
            int((ros[a]["permuted"]).sum()) for a in (ARM_STAT, ARM_EXPO))),
    }
    if y.sum() < 10 or (~y).sum() < 10:
        info["analysis_skipped"] = "fewer than 10 eligible ticks in the shift arm or controls"
        return info

    # Oracle: everything computable from the transition itself, cross-fitted.
    feat = np.vstack([_feats(ros[a], els[a]) for a in ORDER])
    fold = (pos_in_arm // ORACLE_FOLD_BLOCK) % 2
    vals["ORACLE"] = _oracle_scores(feat, y, fold)

    p, beta = _propensity(X, y)
    matched, s, minfo = _match(p, y, n_strata, sc["min_stratum"])
    info.update({"propensity_beta": beta, "match": minfo,
                 "n_matched_shift": int((matched & y).sum()),
                 "n_matched_control": int((matched & ~y).sum()),
                 "n_matched_by_control_arm": {a: int((matched & (arm_code == i)).sum())
                                              for i, a in enumerate(ORDER) if a != ARM_SHIFT},
                 "support_fraction": float((matched & y).sum() / max(1, int(y.sum())))})

    point: Dict[str, float] = {}
    unmatched: Dict[str, float] = {}
    per_control: Dict[str, Dict[str, float]] = {}
    for name, v in vals.items():
        ok = np.isfinite(v)
        unmatched[name] = _auc(v[y & ok], v[~y & ok]) - 0.5
        mm = matched & ok
        point[name] = _strat_sep(v[mm], y[mm], s[mm], n_strata)
        per_control[name] = {}
        for i, a in enumerate(ORDER):
            if a == ARM_SHIFT:
                continue
            mc = mm & (y | (arm_code == i))
            per_control[name][a] = _strat_sep(v[mc], y[mc], s[mc], n_strata)
    info["sep_unmatched"] = unmatched
    info["sep_matched"] = point
    info["sep_matched_vs_each_control_recorded"] = per_control
    info["smd_unmatched"] = {c: _smd(X[:, i], y, None, n_strata) for i, c in enumerate(COVARIATES)}
    info["smd_matched"] = {c: _smd(X[matched, i], y[matched], s[matched], n_strata)
                           for i, c in enumerate(COVARIATES)}
    he = X[:, 0]
    info["harm_exposure_rel_dev_unmatched_1062a_style"] = {
        a: (float((he[y].mean() - he[arm_code == i].mean()) / abs(he[arm_code == i].mean()))
            if abs(he[arm_code == i].mean()) > 1e-12 else None)
        for i, a in enumerate(ORDER) if a != ARM_SHIFT}
    info["harm_exposure_rel_dev_matched_1062a_style"] = _matched_rel_dev(
        he[matched], y[matched], s[matched], n_strata)

    # moving-block bootstrap within each arm, PAIRED across sources
    rng = np.random.default_rng(BOOT_SEED + seed)
    arm_idx = []
    for i in range(len(ORDER)):
        ix = np.where(matched & (arm_code == i))[0]
        arm_idx.append(ix[np.argsort(pos_in_arm[ix])])
    names = list(vals.keys())
    boots = {n: np.full(sc["n_boot"], np.nan) for n in names}
    if (matched & y).any() and (matched & ~y).any():
        for b in range(sc["n_boot"]):
            ii = np.concatenate([ix[_block_resample(ix.size, rng, BOOT_BLOCK)]
                                 for ix in arm_idx if ix.size > 0])
            yb, sb = y[ii], s[ii]
            for n in names:
                vb = vals[n][ii]
                ok = np.isfinite(vb)
                boots[n][b] = _strat_sep(vb[ok], yb[ok], sb[ok], n_strata)

    q_better = ALPHA / N_ALTERNATIVES
    best_res_b = np.maximum(boots[BASELINES[0]], boots[BASELINES[1]])
    best_res_pt = max(point[BASELINES[0]], point[BASELINES[1]])
    alt: Dict[str, Any] = {}
    for a in ALTERNATIVES:
        d_res_b = boots[a] - best_res_b
        d_per_b = boots[a] - boots["PERSIST"]
        res_lo, per_lo = _q(d_res_b, q_better), _q(d_per_b, q_better)
        res_hi = _q(d_res_b, 1.0 - ALPHA)
        if math.isfinite(res_lo) and math.isfinite(per_lo) and res_lo > 0 and per_lo > 0:
            status = "better"
        elif math.isfinite(res_hi) and res_hi < NULL_MARGIN:
            status = "not_better"
        else:
            status = "cannot_determine"
        alt[a] = {"sep": point[a], "d_base": point[a] - best_res_pt,
                  "d_per": point[a] - point["PERSIST"],
                  "d_base_lo95": res_lo, "d_per_lo95": per_lo,
                  "d_base_hi95": res_hi, "status": status}
    info["alternatives"] = alt
    info["baseline_d_per"] = {r: point[r] - point["PERSIST"] for r in BASELINES}
    info["recorded_d_per"] = {r: point[r] - point["PERSIST"]
                              for r in ("RESID_ONLINE", "RESID_ENSMEAN", "INNOV_ONLINE")}
    info["sep_ci95"] = {n: [_q(boots[n], 0.025), _q(boots[n], 0.975)] for n in names}
    info["oracle_sep"] = point["ORACLE"]
    info["oracle_sep_lo95_one_sided"] = _q(boots["ORACLE"], ALPHA)
    level = [abs(point["CANARY_LEVEL_" + c]) for c in COVARIATES]
    info["level_canary_abs_sep"] = dict(zip(COVARIATES, level))
    info["level_canary_worst_abs_sep"] = (max(level) if all(math.isfinite(x) for x in level)
                                          else float("nan"))
    info["label_canary_sep"] = point["CANARY_LABEL"]
    ev = vals["ENS_DISAGREE"][np.isfinite(vals["ENS_DISAGREE"])]
    med = float(np.median(ev)) if ev.size else float("nan")
    info["ens_cross_tick_iqr_over_median"] = (
        float((np.quantile(ev, 0.75) - np.quantile(ev, 0.25)) / med)
        if ev.size and med > 0 else float("nan"))

    # redundancy with the harm level, per source, per arm (recorded)
    red: Dict[str, Any] = {}
    for name in SOURCES:
        red[name] = {}
        for a in ORDER:
            ro, el = ros[a], els[a]
            v = srcs[a][name][el]
            red[name][a] = {
                "abs_spearman_valence_harm_increment": abs(_spearman(v, ro["dvh"][el])),
                "abs_spearman_harm_exposure": abs(_spearman(v, ro["he"][el])),
                "abs_spearman_z_harm_a_norm": abs(_spearman(
                    v, ro["next"][torch.as_tensor(el)].norm(dim=-1).numpy())),
                "abs_spearman_valence_harm_level": abs(_spearman(v, ro["vh"][el])),
                "mean": float(np.mean(v)) if v.size else float("nan"),
            }
        um = unmatched[name]
        red[name]["harm_explained_fraction"] = (
            float(1.0 - point[name] / um) if abs(um) > 0.02 and math.isfinite(point[name])
            else None)
    info["redundancy_with_harm_level"] = red

    r5 = (lambda a: [float("%.5g" % x) if math.isfinite(x) else None for x in a])
    info["per_tick"] = {
        "arm_order": ORDER, "arm_code": [int(x) for x in arm_code],
        "stratum": [int(x) for x in s], "pos_in_arm": [int(x) for x in pos_in_arm],
        "propensity": r5(p),
        **{"cov_" + c: r5(X[:, i]) for i, c in enumerate(COVARIATES)},
        **{"src_" + n: r5(vals[n]) for n in SOURCES + ["ORACLE"]},
    }
    return info


def _run_seed(seed: int, sc: Dict[str, Any], dry: Optional[str]) -> Tuple[List[Dict], Dict]:
    print("Seed %d Condition %s" % (seed, "STATIONARY+EXPOSED+SHIFT"), flush=True)
    rows: List[Dict[str, Any]] = []
    with arm_cell(
        seed, config_slice=_config_slice(), script_path=Path(__file__),
        config_slice_declared=True, include_driver_script_in_hash=False,
        extra_substrate_paths=[DRIVER_1062A, DRIVER_1077],
        extra_ineligible_reasons=(["all_arms_share_one_collection_cell"]
                                  + (["dry_run"] if dry else [])),
    ) as cell:
        random.seed(seed)   # as 1062a / 1077, after arm_cell's full reset
        env = B._make_env(seed)
        agent = REEAgent(B.make_config(env))
        init_state = copy.deepcopy(agent.e2_harm_a.state_dict())
        col = C._collect(agent, env, seed, sc["p0"], sc["p1"], sc["spe"])
        online = copy.deepcopy(agent.e2_harm_a)
        online.eval()
        C._detach_nonleaf(agent)
        C._detach_nonleaf(env)
        n_buf = int(col["prev"].shape[0])
        print("  [collect] seed=%d banked=%d online_updates=%d"
              % (seed, n_buf, len(col["online_losses"])), flush=True)

        members, fits = _train_ensemble(agent.e2_harm_a, init_state, col, seed, sc)

        # calibration rows = the fitter's held-out rows (all rows if too few)
        ho = _heldout_idx(n_buf, seed)
        if int(ho.shape[0]) < 10:
            ho = torch.arange(n_buf)
        hp, ha, hn = col["prev"][ho], col["act"][ho], col["next"][ho]
        with torch.no_grad():
            var0_on = ((hn - online(hp, ha)) ** 2).mean(0)
            st_in = _ens_predict(members, hp, ha)
            var0_cv = ((hn - st_in[0]) ** 2).mean(0)   # member 0 = the converged baseline head
            dis_in = st_in.var(0, unbiased=False).mean(-1)
            zsd = col["prev"].std(0) + 1e-8
            g = torch.Generator().manual_seed(seed + OOD_SEED_OFFSET)
            hp_ood = hp + OOD_SCALE * zsd * torch.randn(hp.shape, generator=g)
            dis_ood = _ens_predict(members, hp_ood, ha).var(0, unbiased=False).mean(-1)
            pair_min = min(
                float((st_in[i] - st_in[j]).abs().mean().item())
                for i in range(ENSEMBLE_K) for j in range(i + 1, ENSEMBLE_K))
        med_in = float(dis_in.median().item())
        ood_ratio = float(dis_ood.median().item()) / med_in if med_in > 0 else float("inf")
        ens_ok = bool(math.isfinite(ood_ratio) and ood_ratio >= OOD_RATIO_MIN
                      and pair_min > 1e-9)

        ros: Dict[str, Dict[str, Any]] = {}
        srcs: Dict[str, Dict[str, Any]] = {}
        for arm, shift, eps in ARMS:
            ros[arm] = _rollout(agent, env, seed, sc["budget"], shift=shift, epsilon=eps)
            srcs[arm] = _sources(ros[arm], online, members, var0_on, var0_cv)
            print("  [rollout] %s seed=%d steps=%d episodes=%d transitions=%d shifts=%d"
                  % (arm, seed, ros[arm]["n_steps"], ros[arm]["n_episodes"],
                     int(ros[arm]["prev"].shape[0]), ros[arm]["n_world_rule_shifts"]),
                  flush=True)

        # persistence verdicts (the reproduction control + recorded context)
        verdicts: Dict[str, Any] = {}
        for arm, _sh, _e in ARMS:
            ro, src = ros[arm], srcs[arm]
            elt = torch.as_tensor(_eligible(arm, ro))
            tg, pv = ro["next"][elt], ro["prev"][elt]
            v_on = persistence_verdict(src["_pred_online"][elt], tg, pv)
            verdicts[arm] = {
                "online": C._verdict_dict(v_on),
                "ensemble_mean": C._verdict_dict(
                    persistence_verdict(src["_pred_ensmean"][elt], tg, pv)),
                "members": [C._verdict_dict(persistence_verdict(src["_stack"][k][elt], tg, pv))
                            for k in range(ENSEMBLE_K)],
            }
            print("  [persistence] %s seed=%d online status=%s d=%s"
                  % (arm, seed, v_on.status, verdicts[arm]["online"].get("relative_skill")),
                  flush=True)

        # Recorded, NON-GATING (1077 autopsy constraints 3 and 4).
        map_keys = sorted(env._action_map.keys())
        learn: Dict[str, Any] = {}
        act_sens: Dict[str, Any] = {}
        for arm in (ARM_STAT, ARM_SHIFT):
            el = _eligible(arm, ros[arm])
            learn[arm] = _learnability(ros[arm], el, agent.e2_harm_a, init_state, seed, sc)
            elt = torch.as_tensor(el)
            pv_, ac_, nx_ = ros[arm]["prev"][elt], ros[arm]["act"][elt], ros[arm]["next"][elt]
            act_sens[arm] = {
                "converged_member0": _action_sensitivity(members[0], pv_, ac_, nx_, map_keys),
                "ensemble_mean": _action_sensitivity(
                    lambda z, a: _ens_predict(members, z, a).mean(0), pv_, ac_, nx_, map_keys),
                "online": _action_sensitivity(online, pv_, ac_, nx_, map_keys),
            }
            print("  [learnability] %s seed=%d in_sample_real=%s permuted=%s separates=%s "
                  "episode_split=%s conv_action_sens=%s"
                  % (arm, seed,
                     (learn[arm].get("in_sample_real") or {}).get("verdict", {}).get("status"),
                     (learn[arm].get("in_sample_permuted_delta") or {}).get("verdict", {}).get("status"),
                     learn[arm].get("real_separates_from_permuted"),
                     (learn[arm].get("episode_split_heldout") or {}).get("verdict", {}).get("status"),
                     act_sens[arm]["converged_member0"].get("ratio_to_true_displacement")),
                  flush=True)

        an = _analyse(ros, srcs, seed, sc)
        reproduced = verdicts[ARM_STAT]["online"]["status"] == "persistence_dominated"
        shift_fired = bool(ros[ARM_SHIFT]["n_world_rule_shifts"] >= 1
                           and an.get("n_eligible_shift", 0) > 0)
        n_ms = an.get("n_matched_shift", 0)
        n_mc = an.get("n_matched_control", 0)
        lvl = an.get("level_canary_worst_abs_sep", float("nan"))
        lab = an.get("label_canary_sep", float("nan"))
        checks = {
            "reproduced": bool(reproduced),
            "shift_fired": shift_fired,
            "controls_canonical": an.get("n_control_ticks_under_permuted_map", 1) == 0,
            "label_canary_ok": bool(math.isfinite(lab) and lab >= LABEL_CANARY_MIN),
            "exposure_matched": bool(math.isfinite(lvl) and lvl <= LEVEL_CANARY_TOL),
            "support_ok": bool(an.get("support_fraction", 0.0) >= SUPPORT_FRACTION_MIN),
            "rows_ok": bool(min(n_ms, n_mc) >= sc["min_matched"]),
            "ensemble_ok": ens_ok,
            # RESID_CONV / INNOV_CONV are named "the converged residual head"
            # (member 0 == V3-EXQ-1077's ARM_CONVERGED head). 1077 gated its own
            # converged arm on fit["converged"]; this run must verify the same
            # thing rather than assume it, or the baseline could be an
            # unconverged head sitting ABOVE persistence (as at probe scale).
            "baseline_head_converged": bool(fits[0]["converged"]),
            "ensemble_varies_across_eval_ticks": bool(
                math.isfinite(an.get("ens_cross_tick_iqr_over_median", float("nan")))
                and an["ens_cross_tick_iqr_over_median"] >= ENS_SPREAD_MIN),
        }
        eligible = all(checks.values())
        olo = an.get("oracle_sep_lo95_one_sided", float("nan"))
        detectable = bool(math.isfinite(olo) and olo > 0)
        summary = {
            "seed": seed, "checks": checks, "eligible": eligible, "detectable": detectable,
            "alt_status": ({a: an["alternatives"][a]["status"] for a in ALTERNATIVES}
                           if "alternatives" in an else {}),
            "ood_ratio": ood_ratio, "member_pair_min_abs_diff": pair_min,
            "n_members_converged": int(sum(1 for f in fits if f["converged"])),
            "baseline_heldout_over_persistence": float(fits[0]["eval_loss_over_persistence"]),
            "ens_cross_tick_iqr_over_median": an.get("ens_cross_tick_iqr_over_median",
                                                     float("nan")),
            "n_matched_shift": n_ms, "n_matched_control": n_mc,
            "level_canary_worst_abs_sep": lvl, "label_canary_sep": lab,
            "support_fraction": an.get("support_fraction", float("nan")),
            "oracle_sep": an.get("oracle_sep", float("nan")), "oracle_sep_lo95": olo,
            "online_stationary_d": verdicts[ARM_STAT]["online"].get("relative_skill"),
        }
        print("  [analysis] seed=%d eligible=%s detectable=%s checks=%s alt=%s"
              % (seed, eligible, detectable,
                 ",".join("%s:%d" % (k, int(v)) for k, v in checks.items()),
                 ",".join("%s:%s" % (k, v) for k, v in summary["alt_status"].items())),
              flush=True)

        for arm, shift, eps in ARMS:
            ro, src = ros[arm], srcs[arm]
            el = _eligible(arm, ro)
            elt = torch.as_tensor(el)
            row: Dict[str, Any] = {
                "arm_id": arm, "seed": seed, "cell_id": "%s/seed%d" % (arm, seed),
                "world_rule_shift": shift, "p2_epsilon": eps,
                "n_p2_env_steps": ro["n_steps"], "n_p2_episodes": ro["n_episodes"],
                "n_transitions": int(ro["prev"].shape[0]), "n_eligible": int(el.sum()),
                "n_world_rule_shifts": ro["n_world_rule_shifts"],
                "action_map_canonical_at_onset": ro["action_map_canonical_at_onset"],
                "n_action_map_entries_permuted_at_end": ro["n_action_map_entries_permuted_at_end"],
                "mean_harm_exposure_eligible": float(ro["he"][el].mean()) if el.any() else None,
                "mean_valence_harm_increment_eligible": (float(ro["dvh"][el].mean())
                                                         if el.any() else None),
                "final_valence_harm_level": float(ro["vh"][-1]) if ro["vh"].size else None,
                "mean_z_harm_a_norm_eligible": (float(ro["next"][elt].norm(dim=-1).mean())
                                                if el.any() else None),
                "mean_steps_since_reset_eligible": float(ro["k"][el].mean()) if el.any() else None,
                "source_means_eligible": {n: float(np.mean(src[n][el])) if el.any() else None
                                          for n in SOURCES},
                "persistence_verdicts": verdicts[arm],
            }
            if arm == ARM_STAT:
                row["training"] = {
                    "online": {"mode": "online_batch1", "n_updates": len(col["online_losses"]),
                               "loss_curve_per_update": col["online_losses"],
                               "loss_curve_p1_episode_mean": col["online_ep_mean"]},
                    "ensemble_members": [dict(f, member=k) for k, f in enumerate(fits)],
                    "n_banked_transitions": n_buf,
                    "innovation_var0_online": [float(x) for x in var0_on],
                    "innovation_var0_conv": [float(x) for x in var0_cv],
                    "ensemble_ood": {"median_disagreement_in": med_in,
                                     "median_disagreement_ood": float(dis_ood.median()),
                                     "ratio": ood_ratio, "member_pair_min_abs_diff": pair_min},
                }
                row["analysis"] = an
                row["seed_summary"] = summary
                row["learnability_recorded_nongating"] = learn
                row["action_sensitivity_recorded_nongating"] = act_sens
            cell.stamp(row)
            rows.append(row)
        ok = eligible and detectable and any(
            v == "better" for v in summary["alt_status"].values())
        print("verdict: %s" % ("PASS" if ok else "FAIL"), flush=True)
    return rows, summary


# --------------------------------------------------------------------------- #
# Routing                                                                      #
# --------------------------------------------------------------------------- #

ROUTES = {
    "pe_source_leg_supported": (
        "PASS: ensemble disagreement separates exposure-matched world surprise better "
        "than the converged residual head, its innovation-variance form AND persistence "
        "on >= 2 seeds. Route: "
        "candidate predictive-cost source for SD-PP-B9 / SD-032b (the dACC harm PE); "
        "NEEDS GOVERNANCE before any build -- this run licenses nothing, and "
        "SD-PP-B11 stays registration-only."),
    "declared_null_alternative_tracks_harm_level": (
        "FAIL, declared null won: ensemble disagreement carries no more exposure-matched "
        "world surprise than the converged residual head / its innovation-variance form "
        "(a >= NULL_MARGIN improvement excluded on >= 2 seeds). Route the NEXT proposal "
        "to the H-harm-head-representation-ceiling probe (vary harm_history_len, measure "
        "per-step z_harm_a displacement, each setting fitted with V3-EXQ-1077's "
        "converged fitter, DV skill-vs-persistence). Explicitly NOT more training: "
        "V3-EXQ-1077 already converged the head."),
    "event_not_detectable_cannot_determine": (
        "FAIL, cannot_determine: a cross-fitted classifier over the full transition "
        "cannot separate the shift arm from the matched stationary arm on >= 2 "
        "eligible seeds, so NO PE source over this stream can -- the PE-source "
        "contrast is undefined, not negative. Consistent with H1 (exposure-error "
        "coupling) / the representation ceiling; do NOT requeue this design as-is -- "
        "route to the representation-ceiling probe or a different decoupling lever."),
    "cannot_determine_no_quorum": (
        "FAIL, cannot_determine: no status reached quorum among eligible, detectable "
        "seeds. Requeue-not-verdict: a larger P2 budget or more seeds, same design."),
    "substrate_not_ready_requeue": (
        "FAIL, precondition failure: requeue-not-verdict. Read the failed "
        "precondition; nothing about the PE source was measured."),
}


def _route(summaries: List[Dict[str, Any]], canary_ok: bool, quorum: int) -> Dict[str, Any]:
    n_rep = sum(1 for s in summaries if s["checks"]["reproduced"])
    elig = [s for s in summaries if s["eligible"]]
    det = [s for s in elig if s["detectable"]]
    n_better = {a: sum(1 for s in det if s["alt_status"].get(a) == "better")
                for a in ALTERNATIVES}
    n_null = sum(1 for s in det
                 if all(s["alt_status"].get(a) == "not_better" for a in ALTERNATIVES))
    best_alt = max(ALTERNATIVES, key=lambda a: n_better[a])
    if not canary_ok:
        label, why = "substrate_not_ready_requeue", "persistence_skill_gate canary failed"
    elif n_rep < quorum:
        label = "substrate_not_ready_requeue"
        why = ("reproduction control failed: online head persistence_dominated on the "
               "stationary rollout on %d seeds, quorum %d" % (n_rep, quorum))
    elif len(elig) < quorum:
        label = "substrate_not_ready_requeue"
        why = ("fewer than %d eligible seeds (%d): a readiness precondition (label "
               "canary / exposure-match tolerance / support / rows / ensemble / shift) "
               "failed" % (quorum, len(elig)))
    elif len(det) < quorum:
        label = "event_not_detectable_cannot_determine"
        why = ("oracle detectability on %d of %d eligible seeds, quorum %d"
               % (len(det), len(elig), quorum))
    elif n_better[best_alt] >= quorum:
        label = "pe_source_leg_supported"
        why = "%s better than residual and persistence on %d detectable seeds" % (
            best_alt, n_better[best_alt])
    elif n_null >= quorum:
        label = "declared_null_alternative_tracks_harm_level"
        why = ("ENS_DISAGREE not_better (upper bound of D_base < %.2f) on %d "
               "detectable seeds" % (NULL_MARGIN, n_null))
    else:
        label = "cannot_determine_no_quorum"
        why = "no quorum: better=%s null_seeds=%d" % (n_better, n_null)
    return {"label": label, "why": why, "route": ROUTES[label],
            "n_reproduced": n_rep, "n_eligible": len(elig), "n_detectable": len(det),
            "n_better_by_alternative": n_better, "best_alternative": best_alt,
            "n_null_seeds": n_null, "quorum": quorum}


def _worst(summaries, key, mode):
    vals = [(s[key], s["seed"]) for s in summaries
            if s.get(key) is not None and math.isfinite(float(s[key]))]
    if not vals:
        return float("nan"), None
    v = min(vals) if mode == "min" else max(vals)
    return float(v[0]), v[1]


def run_experiment(dry: Optional[str]) -> Tuple[Dict[str, Any], float]:
    t0 = time.perf_counter()
    sc = _scale(dry)
    canary = check_canary()
    canary_ok = bool(canary.get("ok"))
    rows: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []
    for sd in sc["seeds"]:
        r, s = _run_seed(sd, sc, dry)
        rows.extend(r)
        summaries.append(s)
    routed = _route(summaries, canary_ok, sc["quorum"])
    label = routed["label"]
    outcome = "PASS" if label == "pe_source_leg_supported" else "FAIL"
    q = float(sc["quorum"])

    lvl, lvl_seed = _worst(summaries, "level_canary_worst_abs_sep", "max")
    lab, lab_seed = _worst(summaries, "label_canary_sep", "min")
    sup, sup_seed = _worst(summaries, "support_fraction", "min")
    ood, ood_seed = _worst(summaries, "ood_ratio", "min")
    ens_sp, ens_sp_seed = _worst(summaries, "ens_cross_tick_iqr_over_median", "min")
    nrow = [min(s["n_matched_shift"], s["n_matched_control"]) for s in summaries]
    preconditions = [
        {"name": "instrument_canary_persistence_gate",
         "description": "persistence_skill_gate.check_canary reproduces 1062a's 6 recorded "
                        "cells persistence_dominated",
         "measured": 1.0 if canary_ok else 0.0, "threshold": 1.0,
         "control": "pinned 1062a manifest cells (known persistence_dominated)",
         "direction": "lower", "met": canary_ok},
        {"name": "reproduction_control_persistence_dominated_seeds",
         "description": "online head persistence_dominated on the STATIONARY rollout "
                        "(eps 0.0): the residual-head defect under question reproduces",
         "measured": float(routed["n_reproduced"]), "threshold": q,
         "control": "known-negative: 1062a 3/3 stationary and 1077 3/3 at eps 0.0",
         "direction": "lower", "met": routed["n_reproduced"] >= q},
        {"name": "label_canary_sep_worst_seed",
         "description": "a synthetic source EQUAL to the arm label scored by the same "
                        "stratified statistic (analytic 0.5): the instrument can detect",
         "measured": lab, "threshold": LABEL_CANARY_MIN, "offending_cell": lab_seed,
         "control": "known-positive synthetic source", "direction": "lower",
         "met": bool(math.isfinite(lab) and lab >= LABEL_CANARY_MIN)},
        {"name": "exposure_match_level_canary_worst_abs_sep",
         "description": "worst |sep| over the four harm covariates (harm_exposure, "
                        "VALENCE_HARM increment, ||z_harm_a||, log1p steps) each scored "
                        "AS a PE source by the same stratified statistic (analytic 0 "
                        "under perfect matching): the stated exposure-match tolerance",
         "measured": lvl, "threshold": LEVEL_CANARY_TOL, "offending_cell": lvl_seed,
         "control": "known-null synthetic sources (pure harm level)",
         "direction": "upper", "met": bool(math.isfinite(lvl) and lvl <= LEVEL_CANARY_TOL)},
        {"name": "common_support_fraction_worst_seed",
         "description": "fraction of shift-arm surprise ticks retained after trimming and "
                        "thin-stratum removal",
         "measured": sup, "threshold": SUPPORT_FRACTION_MIN, "offending_cell": sup_seed,
         "direction": "lower", "met": bool(math.isfinite(sup) and sup >= SUPPORT_FRACTION_MIN)},
        {"name": "matched_rows_worst_arm_worst_seed",
         "description": "fewest matched ticks in either arm on any seed",
         "measured": float(min(nrow)) if nrow else 0.0, "threshold": float(sc["min_matched"]),
         "direction": "lower", "met": bool(nrow and min(nrow) >= sc["min_matched"])},
        {"name": "ensemble_ood_disagreement_ratio_worst_seed",
         "description": "median ensemble disagreement on OOD-perturbed inputs / on "
                        "held-out in-distribution inputs (members must also be pairwise "
                        "distinct): the alternative source is non-degenerate and CAN move",
         "measured": ood, "threshold": OOD_RATIO_MIN, "offending_cell": ood_seed,
         "control": "known-positive: inputs pushed 3 sd off the training manifold",
         "direction": "lower", "met": bool(math.isfinite(ood) and ood >= OOD_RATIO_MIN)},
        {"name": "baseline_head_converged_seeds",
         "description": "seeds where ensemble member 0 (the RESID_CONV / INNOV_CONV "
                        "baseline head, == V3-EXQ-1077 ARM_CONVERGED) meets 1077's own "
                        "convergence rule: lr schedule exhausted before the cap AND best "
                        "held-out loss <= persistence on the same rows",
         "measured": float(sum(1 for s in summaries if s["checks"]["baseline_head_converged"])),
         "threshold": q, "control": "V3-EXQ-1077 ARM_CONVERGED converged 3/3 on this collection",
         "direction": "lower",
         "met": sum(1 for s in summaries if s["checks"]["baseline_head_converged"]) >= q},
        {"name": "ensemble_cross_tick_spread_worst_seed",
         "description": "IQR / median of ENS_DISAGREE over the eligible eval ticks of all "
                        "arms: the alternative VARIES across the very ticks its AUC ranks "
                        "(a tick-constant source scores sep 0 and would hand the null a "
                        "degenerate win)",
         "measured": ens_sp, "threshold": ENS_SPREAD_MIN, "offending_cell": ens_sp_seed,
         "direction": "lower", "met": bool(math.isfinite(ens_sp) and ens_sp >= ENS_SPREAD_MIN)},
        {"name": "shift_fired_seeds",
         "description": "seeds where the world_rule_shift fired in P2 and produced "
                        "eligible surprise ticks",
         "measured": float(sum(1 for s in summaries if s["checks"]["shift_fired"])),
         "threshold": q, "direction": "lower",
         "met": sum(1 for s in summaries if s["checks"]["shift_fired"]) >= q},
        {"name": "eligible_seeds",
         "description": "seeds meeting EVERY per-seed readiness check (the count the "
                        "routing gates on)",
         "measured": float(routed["n_eligible"]), "threshold": q, "direction": "lower",
         "met": routed["n_eligible"] >= q},
        {"name": "event_detectable_oracle_seeds",
         "description": "eligible seeds where a cross-fitted classifier over the full "
                        "transition separates the arms within the same strata (one-sided "
                        "95% lower bound of sep > 0): the surprise is in the stream at all",
         "measured": float(routed["n_detectable"]), "threshold": q, "direction": "lower",
         "met": routed["n_detectable"] >= q},
    ]
    criteria = [
        {"name": "C1_alternative_better_than_residual_and_persistence",
         "load_bearing": True,
         "measured": float(routed["n_better_by_alternative"][routed["best_alternative"]]),
         "threshold": q,
         "description": "detectable seeds where %s has one-sided 95%% lower bounds of "
                        "D_base (vs max of RESID_CONV, INNOV_CONV) and D_per (vs PERSIST) "
                        "both > 0" % routed["best_alternative"],
         "passed": label == "pe_source_leg_supported"},
        {"name": "N1_declared_null_all_alternatives_not_better", "load_bearing": False,
         "measured": float(routed["n_null_seeds"]), "threshold": q,
         "description": "detectable seeds where ENS_DISAGREE's D_base one-sided 95%% "
                        "upper bound < %.2f (the declared null)" % NULL_MARGIN,
         "passed": label == "declared_null_alternative_tracks_harm_level"},
    ]
    combination_rule = (
        "Preconditions first (canary; reproduction >= quorum; eligible seeds >= quorum) -> "
        "else substrate_not_ready_requeue. Then oracle-detectable eligible seeds >= quorum "
        "-> else event_not_detectable_cannot_determine. Then, among detectable seeds, "
        "ENS_DISAGREE `better` on >= quorum -> pe_source_leg_supported (PASS); "
        "ENS_DISAGREE `not_better` on >= quorum seeds -> "
        "declared_null_alternative_tracks_harm_level; else cannot_determine_no_quorum. "
        "better = one-sided 95% lower bounds of BOTH "
        "D_base = sep(ENS) - max(sep(RESID_CONV), sep(INNOV_CONV)) and "
        "D_per = sep(ENS) - sep(PERSIST) > 0; not_better = one-sided 95% upper bound "
        "of D_base < NULL_MARGIN. sep = propensity-stratified (ATT) AUC - 0.5 of shift "
        "vs pooled matched controls. Only C1 is load-bearing for PASS.")
    all_pre_met = all(p["met"] for p in preconditions)
    seps = []
    for r in rows:
        an = r.get("analysis") or {}
        for n in SOURCES:
            v = (an.get("sep_matched") or {}).get(n)
            if v is not None and math.isfinite(v):
                seps.append(round(float(v), 9))
    non_degenerate = bool(all_pre_met and len(set(seps)) > 1)

    flat: Dict[str, float] = {
        "n_seeds": float(len(summaries)), "n_reproduced": float(routed["n_reproduced"]),
        "n_eligible": float(routed["n_eligible"]), "n_detectable": float(routed["n_detectable"]),
        "n_null_seeds": float(routed["n_null_seeds"]),
        "canary_ok": 1.0 if canary_ok else 0.0,
        "all_preconditions_met": 1.0 if all_pre_met else 0.0,
        "level_canary_worst_abs_sep": lvl, "label_canary_worst_sep": lab,
        "support_fraction_worst": sup, "ood_ratio_worst": ood,
    }
    for a in ALTERNATIVES:
        flat["n_better_%s" % a] = float(routed["n_better_by_alternative"][a])
    for r in rows:
        an = r.get("analysis")
        if not an or "sep_matched" not in an:
            continue
        sd = r["seed"]
        for n in SOURCES + ["ORACLE"]:
            flat["s%d_sep_%s" % (sd, n)] = an["sep_matched"].get(n)
            flat["s%d_sep_unmatched_%s" % (sd, n)] = an["sep_unmatched"].get(n)
        for a in ALTERNATIVES:
            al = an["alternatives"][a]
            for k in ("d_base", "d_per", "d_base_lo95", "d_per_lo95", "d_base_hi95"):
                flat["s%d_%s_%s" % (sd, a, k)] = al[k]
            flat["s%d_%s_better" % (sd, a)] = 1.0 if al["status"] == "better" else 0.0
            flat["s%d_%s_not_better" % (sd, a)] = 1.0 if al["status"] == "not_better" else 0.0
        flat["s%d_oracle_sep_lo95" % sd] = an.get("oracle_sep_lo95_one_sided")
        for arm, lr in (r.get("learnability_recorded_nongating") or {}).items():
            for key in ("in_sample_real", "in_sample_permuted_delta", "episode_split_heldout"):
                d = ((lr.get(key) or {}).get("verdict") or {}).get("relative_skill")
                flat["s%d_%s_%s_d" % (sd, arm, key)] = d
            flat["s%d_%s_real_separates_from_permuted" % (sd, arm)] = (
                1.0 if lr.get("real_separates_from_permuted") else 0.0)
        for arm, asd in (r.get("action_sensitivity_recorded_nongating") or {}).items():
            flat["s%d_%s_conv_action_sens_true_disp" % (sd, arm)] = (
                asd["converged_member0"].get("ratio_to_true_displacement"))
        flat["s%d_support_fraction" % sd] = an.get("support_fraction")
        for ca, rd in (an.get("harm_exposure_rel_dev_unmatched_1062a_style") or {}).items():
            flat["s%d_harm_exposure_rel_dev_unmatched_vs_%s" % (sd, ca)] = rd
        flat["s%d_harm_exposure_rel_dev_matched" % sd] = an.get(
            "harm_exposure_rel_dev_matched_1062a_style")
    flat = {k: float(v) for k, v in flat.items()
            if v is not None and isinstance(v, (int, float)) and math.isfinite(float(v))}

    manifest: Dict[str, Any] = {
        "run_id": "%s_%s_v3" % (EXPERIMENT_TYPE, time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())),
        "queue_id": QUEUE_ID,
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": ARCHITECTURE_EPOCH,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "claim_ids": CLAIM_IDS,
        "bears_on": BEARS_ON,
        "validates_substrate": VALIDATES_SUBSTRATE,
        "fanout_leg": FANOUT_LEG,
        "predecessor": "V3-EXQ-1077",
        "evidence_direction": "non_contributory",
        "outcome": outcome,
        "timestamp_utc": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
        "sleep_driver_pattern": "not_applicable",
        "red_team": RED_TEAM_VERDICT,
        "dry_run_scale": dry,
        "base_drivers": {"1062a": {"path": DRIVER_1062A.name, "sha256": _sha256(DRIVER_1062A)},
                         "1077": {"path": DRIVER_1077.name, "sha256": _sha256(DRIVER_1077)}},
        "non_degenerate": non_degenerate,
        "degeneracy_reason": (None if non_degenerate else
                              "a precondition is unmet or every source separation is identical"),
        "readout": flat,
        "arm_results": rows,
        "seed_summaries": summaries,
        "routing": routed,
        "canary": canary,
        "interpretation": {
            "label": label,
            "why": routed["why"],
            "route": routed["route"],
            "routing_table": ROUTES,
            "combination_rule": combination_rule,
            "preconditions": preconditions,
            "criteria": criteria,
            "criteria_non_degenerate": {c["name"]: non_degenerate for c in criteria},
        },
        "ethics_preflight": {
            "involves_negative_valence": False, "involves_suffering_like_state": False,
            "involves_self_model": False,
            "involves_inescapability_or_helplessness": False,
            "involves_offline_replay_over_harm": False,
            "involves_social_mind_or_language": False,
            "involves_human_data_or_clinical_context": False,
            "decision": "allow"},
    }
    return manifest, t0


if __name__ == "__main__":
    _ap = argparse.ArgumentParser()
    _ap.add_argument("--dry-run", action="store_true")
    _ap.add_argument("--dry-scale", choices=["smoke", "probe"], default="smoke",
                     help="with --dry-run only: 'smoke' (seconds) or 'probe' (the "
                          "pre-queue satisfiability probe, ~20-40 min, 1 seed)")
    _args = _ap.parse_args()
    _dry = _args.dry_scale if _args.dry_run else None

    _manifest, _t0 = run_experiment(_dry)
    _out_path = write_flat_manifest(
        _manifest, dry_run=_args.dry_run, config=_config_slice(),
        seeds=_scale(_dry)["seeds"], script_path=Path(__file__),
        started_at=_t0, z_goal_stream_stats=_ZG.stats())
    print("[%s] outcome=%s label=%s" % (EXPERIMENT_TYPE, _manifest["outcome"],
                                         _manifest["interpretation"]["label"]), flush=True)
    _o = str(_manifest["outcome"]).upper()
    emit_outcome(outcome=_o if _o in ("PASS", "FAIL") else "FAIL",
                 manifest_path=_out_path, dry_run=_args.dry_run)
