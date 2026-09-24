"""V3-EXQ-1092 -- SD-PP-B5 inverted-action-map readout vs its ACTION-BLIND NULL at alpha 0.3/0.9.

!! DO NOT QUEUE -- STILL NOT QUEUED, INERT ON MAIN (2026-09-24, SECOND red-team BLOCKING) !!
The option-B re-point described below was implemented, validated (validate_experiments
--strict: 1 OK / 0 warnings; validate_recording --strict: complete) and smoked clean (rc=0;
C1 discriminating -- FAIL at alpha 0.3 on the skill clause, PASS at 0.9). A SECOND Step 4.5
red-team (fable, one foreground pass) then returned BLOCKING on the re-pointed design too.
Every finding was re-verified by this session against source and the dry-run manifest:

  F2 (deepest) THE ACTION-BLIND NULL IS 1 IN POPULATION, SO THE RE-POINT BARELY MOVED THE BAR.
     The inverted map is a BIJECTION (0<->1, 2<->3, 4 fixed) applied to i.i.d.-uniform action
     INDICES, so the executed MOVE sequence is i.i.d. uniform under BOTH maps and the (z0, z1)
     law is IDENTICAL across the two batteries -- only the action LABEL differs. Hence
     blind_null -> 1 in population, and "cross_ratio - blind_null > 0" reduces to "the head
     predicts worse at the OPPOSITE-direction label than at the true one, on rows of the same
     law". MEASURED: blind_null = 0.930 / 0.949 / 0.940 / 0.958, identity-MSE ratios
     1.075 / 1.053 / 1.064 / 1.044 -- all scattered about 1, i.e. realization noise.
     CONSEQUENCE FOR THE WHOLE BAR QUESTION: the 1.09-2.41 blind nulls that motivated
     GFLAG-0470 were an artefact of V3-EXQ-1079's POST-DEATH batteries, where the law is NOT
     preserved. On a LIVE battery options A and B very nearly coincide, so the first stop's
     premise -- that the bar choice is materially load-bearing -- does not hold on live rows.
     It also makes C1 a COMPONENT of d_act rather than a new property: d_act averages over all
     four alternative labels, C1 uses the single opposite-direction one, and measured
     (cross_ratio - blind_null)/(S/T - 1) = 1.34 / 1.21 / 1.68 / 1.40 -- monotone tracking in
     every cell, including the two alpha-0.3 cells where skill is NEGATIVE.
  F2c A FALSE PREMISE THIS DOCSTRING ITSELF ASSERTED -- corrected in place below.
  F1 THE QUEUED RUN'S OUTCOME IS ALREADY DETERMINED. Every cell is a pure function of
     (seed, alpha) and --dry-run executes seeds 42 and 123 at FULL budget against
     SEEDS_REQUIRED = 2, so C1 and C2 are both satisfied before seed 456 runs; only the
     non-load-bearing C2b split is open. (This is the lineage's dry-run convention, inherited
     from V3-EXQ-1082 -- not specific to this driver, but it binds here.)
  F3 POST_RESET_SKIP = 3 IS AN ARM-ASYMMETRIC INSTRUMENT CONSTANT. The driver never calls
     REEAgent.reset(), so z_world's EMA carries across env resets; after 3 skipped steps the
     residual weight on the PREVIOUS episode is (1-alpha)^3 = 0.343 at alpha 0.3 versus 0.001
     at alpha 0.9. With 28-33 resets per 512-row battery (~16-18 rows/episode) a large share
     of the 0.3 arm's rows carry cross-episode transients the 0.9 arm does not. The constant
     was inherited from V3-EXQ-1082, which ran ONLY at 0.9. C2's alpha contrast and C2b's
     attribution are both confounded by it. THIS ONE IS INDEPENDENT OF THE BAR QUESTION and
     would bind any future alpha-contrast design on this collector.
  F4 (OWED, deliberately NOT applied) C1's per-row inputs (e_head_c, e_id_o, e_id_c) are not
     persisted, so the new load-bearing CI is not re-derivable post hoc. Not patched because
     this driver is blocked and shipping unsmoked code to it would be worse than recording it.
  F5 (text only, FIXED) several manifest/docstring strings still described the option-C design.

Changing what gets measured is the USER's call, so this driver was NOT redesigned a third time.
Full verification tables: REE_assembly evidence/planning/
sdppb5_inverted_map_ratio_criterion_aliasing_staged_20260924.md. Governance: GFLAG-0470,
GFLAG-0475, GFLAG-0482. THE RECOVERABLE HALF IS SETTLED AND LANDED:
reanalysis_sdppb5_off_head_action_read_alpha09_live_battery_20260924T171654Z (OFF head reads
its action at alpha 0.9 on a live battery, 3/3 seeds, zero compute).
Does e2.world_forward's prediction degrade MORE under an inverted action map than an
action-blind predictor does on the same battery pair -- and does that differ between
alpha_world 0.3 (damped) and 0.9 (SD-008's stable floor)?

PURPOSE: diagnostic (validates_substrate SD-PP-B5-z-world-per-step-displacement-range).
Non-contributory to governance confidence by design; claim_ids EMPTY, bears_on names the claims.

SLEEP DRIVER: not applicable (no sleep machinery; P0 world_forward training only).

WHY THIS RUN EXISTS, AND WHAT IT IS *NOT* ASKING
------------------------------------------------
Routed by confirmed failure_autopsy_V3-EXQ-1082_2026-09-24 targets[0].fanout_recommendation,
suggested_probes[0] (hypothesis H-operating-point). It composes V3-EXQ-1079's inverted-map env
with V3-EXQ-1082's live reset-on-done collector.

It does NOT ask "does the OFF head read its action at 0.9". THAT IS ALREADY ANSWERED, at zero
compute, by reanalysis
  reanalysis_sdppb5_off_head_action_read_alpha09_live_battery_20260924T171654Z
(REE_assembly evidence/reanalysis/), derived in closed form from V3-EXQ-1082's landed ARM_OFF
per-row errors: READY on 3/3 seeds (S/T 2.0371/1.5512/1.5858 > 1.0; skill +0.3486/+0.2194/
+0.2149 > 0.0; d_act CI lower > 0 on 3/3). Asking it again here would re-derive landed data --
which is exactly what the first draft of this driver did, and why it was refused.

UNCERTAINTY THIS RUN REMOVES: whether the head's world-model is INVERSION-SENSITIVE beyond the
battery's own difficulty -- i.e. whether an inverted action map is a genuine CONTRADICTION for
this head, or merely a harder set of rows. Reading its action (established) and being
contradicted by an inverted rule are different properties. V3-EXQ-1073 measured cross-battery
ratios of 0.760/0.901/0.881 -- BELOW 1 -- and self-routed
`confidently_wrong_condition_unposeable_on_this_head`.
CORRECTION (2026-09-24, red-team F2c): an earlier draft of this docstring said those ratios
were measured "on a head that did read its action". That is FALSE, and it was this driver's
own assertion. The 1073 autopsy records the opposite -- "B5 (head ignores the action;
copy-the-input)" and "the frame (an action-blind head) is wrong" -- and 1073's manifest
carries NO d_act or shuffle readout at all. 1073 is therefore NOT evidence that
reading-the-action and being-contradicted-by-an-inversion dissociate, and H-difficulty-only
below has no supporting case on record.

HYPOTHESES DISCRIMINATED (>= 2, per GOV-FANOUT-1):
  H-inversion-sensitive   the head encodes the action->displacement RULE, so inverting the map
                          costs it more than it costs an action-blind predictor
                          (cross_ratio - blind_null > 0). An inverted-map contradiction is
                          posable, and the V3-EXQ-1073 MECH-572 design can be re-posed at 0.9.
  H-difficulty-only       the head reads its action (established) but does not encode the rule
                          in a way an inversion contradicts; the whole cross-battery ratio is
                          the two row sets' difficulty difference (cross_ratio ~ blind_null).
                          1073's sub-1 ratios are this shape. Do NOT build the contradiction.
  H-operating-point       (crossed with the above) whichever holds, is it alpha-dependent?
                          C2 plus its positive-control split separates "the HEAD improved" from
                          "the BATTERY became more action-explainable for any reader".
STOP RESULT: if C1 fails at 0.9 -- the ratio sits within its action-blind null -- the
InfoNCE and encoder-displacement legs stay unbuilt and the MECH-572 contradiction re-pose stays
refused, on a measured basis rather than an assumed one. That is a real, informative outcome.

NATIVE CONSUMER THE RATIO REACHES: `experiments/_lib/action_sensitivity_gate.readiness_verdict`
is the shipped pre-flight gate a consolidation/contradiction experiment is required to pass
before it may pose an inverted-action-map condition (that module's own docstring: "this gate
exists to be run BEFORE that kind of experiment"). This run supplies the counterfactual-battery
form of that gate on a live battery -- the form 1073 needed and did not have.

THE BAR -- TWO USER DECISIONS, RECORDED BECAUSE THE SECOND REVERSED THE FIRST
-----------------------------------------------------------------------------
The autopsy's sketch says "inverted-map battery MSE / original MSE with a bar excluding the
degenerate ~1.0 ... reuse readiness_verdict". `readiness_verdict` has TWO forms with DIFFERENT
nulls, and the sketch pairs one form's manipulation with the other's bar:
  * action_shuffle_ratio (counterfactual_battery=None): SAME rows, actions permuted. Null IS
    exactly 1.0 -- difficulty cancels by construction.
  * battery_pair_ratio (counterfactual_battery=cf): TWO batteries, one from an inverted-map
    env. Null is NOT 1.0; it is the ACTION-BLIND ratio over those two row sets.
Stop 1 (chip-20260924-sdppb5-invmap-ratio-bar-decision) -> USER chose option C: load-bear the
same-rows form at 1.0. Implemented; the Step 4.5 red-team then showed that criterion is
`d_act > 0` restated ((1+d_act)/(1-d_act) == S/T identically) and already answered by 1082,
whose ARM_OFF cells this driver's 0.9 arm reproduces BIT-FOR-BIT. Stop 2
(chip-20260924-sdppb5-invmap-loadbearing-recoverable) -> USER chose "(2) then (1)"
(rec-20260924-5812c302): emit the reanalysis above, then RE-POINT the load-bearing criterion at
option B. That is what this driver now does. GFLAG-0470 + GFLAG-0475; analysis in REE_assembly
evidence/planning/sdppb5_inverted_map_ratio_criterion_aliasing_staged_20260924.md.

GOV-REUSE-1 RE-RUN AGAINST THE *NEW* LOAD-BEARING STATISTIC (2026-09-24, the GFLAG-0475 lesson
in force): recoverability is a property of the CRITERION, not of the experiment, so moving the
criterion requires re-running Step 2.4. Scanned 1076 manifests in evidence/experiments (521
carrying arm_results -- denominator printed because a silent zero here is indistinguishable
from a broken search). EXACTLY ONE carries a cross-battery ratio together with an action-blind
null: V3-EXQ-1079, and its collector never read `done`, so every one of its rows is POST-DEATH
(that is also why its blind nulls read 1.09-2.41 while live ones measure ~0.95). The
live-battery form of this statistic exists NOWHERE in the corpus. NOT recoverable -> run.

DESIGN
------
2 arms x 3 seeds = 6 cells. The ONLY between-arm difference is alpha_world.
  ARM_ALPHA_0p3  alpha_world 0.3 (from_dims default; SD-008's damped point)
  ARM_ALPHA_0p9  alpha_world 0.9 (SD-008's stable floor; 1082's operating point)
alpha_world is set EXPLICITLY at from_dims and asserted threaded (MECH-307 guard). Both arms
are OFF -- no interventional margin, asserted. Env, dims, budget (3600 P0 steps), optimiser,
batch sampling and seeds are V3-EXQ-1082's ARM_OFF.

DONE-HANDLING: 1082's `_collect_live_battery` (ported from V3-EXQ-1073 :1054-1056) in the P0
rollout and in BOTH battery collectors: unpack `done`, on done -> env.reset() and prev=None, so
no transition spanning a death is recorded; the collector also skips POST_RESET_SKIP
transitions after each reset and records per-row health. `battery_rows_all_live` and
`battery_cf_rows_all_live` both require ZERO post-death rows -- the inverted map changes the
trajectory, so the second is not implied by the first.

THE READOUTS, per cell (head = the cell's OWN trained e2.world_forward)
  LOAD-BEARING  cross_ratio vs blind_null. readiness_verdict(counterfactual_battery=inverted,
                ratio_floor=blind_null, skill_floor=0.0) == "ready" AND a TWO-SAMPLE bootstrap
                CI on (cross_ratio - blind_null) excluding 0. Two-sample, not paired: the
                original and inverted batteries are different row sets, so they are resampled
                independently.
  RECORDED      the raw >1.0 verdict on the same cross-battery ratio (the autopsy's literal
                declared null, kept evaluable); the SAME-ROWS action_shuffle verdict and its
                16-draw spread (DEMOTED from load-bearing -- see above); d_act + paired CI;
                skill_vs_identity; the ridge positive control; the untrained head; model_r2
                and the persistence verdict.

VERDICT GRID -- the declared null is a DISJUNCTION, so PASS rejects BOTH clauses:
  N1 (re-pointed) "at 0.9 the inverted-map ratio does not exceed its action-blind null"  -> C1
  N2              "the 0.3 and 0.9 arms do not differ"                                   -> C2
  PASS  inverted_map_exceeds_blind_null_and_head_gains_beyond_ceiling -- C1+C2, and the
        difference-of-differences says the HEAD gained beyond the ridge ceiling.
  PASS  inverted_map_exceeds_blind_null_alpha_gain_attributable_to_battery -- C1+C2, but the
        ceiling gained MORE: the alpha gain is the battery's, not the head's.
  PASS  inverted_map_exceeds_blind_null_alpha_attribution_undetermined -- C1+C2, split unclear.
  FAIL  inverted_map_exceeds_blind_null_alpha_contrast_undetermined -- C1 only.
  FAIL  inverted_map_ratio_within_action_blind_null_at_operating_point -- C1 fails on every
        valid seed. NOTE this does NOT say the head is action-blind: the reanalysis above
        established it reads its action at 0.9 on 3/3. It says the INVERTED-MAP readout does
        not clear the nuisance bar, i.e. H-difficulty-only.
  FAIL  inverted_map_readout_undetermined -- neither resolves.
  FAIL  substrate_not_ready_requeue -- any precondition unmet.
(Labels are written to be TRUE in the state they fire; the 2026-09-24 red-team found the
previous draft's "read absent ... despite d_act" label was false in its only reachable state.)

C2'S POSITIVE-CONTROL SPLIT (V3-EXQ-1079 :84-91, :755-758, carried into the GROWS direction):
1079 split its SHRINKS label on whether the budget-free ridge control moved too. The measured
hazard here is the GROWS analogue -- the head's contrast can be positive while the ceiling's is
LARGER, in which case more linearly readable action information reached z_world and the head
did not use it. So the attribution statistic is a difference of differences,
[d_head(0.9)-d_head(0.3)] - [d_pc(0.9)-d_pc(0.3)], with 1079's literal pc contrast recorded
alongside. It SPLITS the label and never changes the counts.

DV-SYMMETRY (Step 3.5), per arm -- BOTH arms, same statement:
The load-bearing DV is a DIFFERENCE of two MSE ratios over the same two batteries. Any uniform
positive rescaling of z_world cancels in each ratio (numerator and denominator both scale by
c^2) and therefore in their difference, and any consistent relabeling of action indices leaves
it unchanged. The manipulation is NOT invariant under that group -- and, importantly, the
BLIND-NULL SUBTRACTION is what makes this true rather than an assumption: whatever alpha does
to the displacement SCALE enters cross_ratio and blind_null alike and cancels, so what survives
is specifically the head's rule-sensitivity over and above an action-blind reader on the same
rows. (The previous draft argued the DV moves with alpha "therefore alpha is not a rescaling";
that inference was invalid for a head RETRAINED per arm -- identity MSE differs ~9.5x and rms
|dz| ~3.1x across arms and skill at 0.3 is NEGATIVE, so d_act at 0.3 is diluted by a fixed
LR/step budget. Recorded here rather than dropped: `rms_dz_per_dim_battery`,
`transition_l2_mean` and both identity MSEs are emitted per cell so a reader can see the scale
change, and C2's positive-control split is what keeps a scale artefact from being read as a
head result.)

MULTI-ARM GATE (Step 3.5 / V3-EXQ-785): preconditions are whole-run worst-cell, which is
correct rather than the 785 defect -- no precondition is structurally unsatisfiable for either
arm (both are ordinary OFF cells differing only in a blend coefficient). C1 is read at
ARM_ALPHA_0p9 by design; `rows_aligned` is scoped by `applies_to` to C2 alone, so an unpairable
contrast cannot vacate C1.

RE-DERIVE BRAKE (Step 2.5b): does not hold. claim_ids [], a new EXQ NUMBER, purpose diagnostic,
and the confirmed 1082 autopsy is the producer half routing this probe. GOV-DIAG-1 counts the
full SD-PP-B5 token at 2 (1079 + 1082), below N=3; the autopsy notes the chain "is converging
..., not circling".

RED-TEAM (Step 4.5, fable): first pass on the option-C draft returned BLOCKING (3 findings, all
verified, all addressed by the option-B re-point above); the verdict of record for the design
AS QUEUED is in the queue entry note.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import random
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiment_protocol import emit_outcome
from experiments._lib.action_sensitivity_gate import (
    MIN_DISTINCT_ACTIONS as GATE_MIN_DISTINCT,
    RATIO_FLOOR, SKILL_FLOOR, battery_pair_ratio, check_canary, format_verdict,
    identity_predictor_mse, readiness_verdict)
from experiments._lib.arm_fingerprint import arm_cell
from experiments._lib.persistence_skill_gate import (
    BOOTSTRAP_SEED, CI_LEVEL, N_BOOTSTRAP, per_row_squared_error, persistence_verdict)
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator
from experiments.pack_writer import write_flat_manifest
from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig

EXPERIMENT_PURPOSE = "diagnostic"
EXPERIMENT_TYPE = "v3_exq_1092_sdppb5_inverted_action_map_alpha_operating_point"
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"
QUEUE_ID = "V3-EXQ-1092"

CLAIM_IDS: List[str] = []
# SD-PP-B5 + MECH-573 from the chip; SD-PP-B10 + SD-008 because the 0.3-vs-0.9 alpha contrast
# is that pair's subject and the 1082 autopsy's own bears_on lists both.
BEARS_ON = ["SD-PP-B5-z-world-per-step-displacement-range",
            "SD-PP-B10-zworld-encoder-action-displacement", "MECH-573", "SD-008"]
VALIDATES_SUBSTRATE = "SD-PP-B5-z-world-per-step-displacement-range"
AUTOPSY = "failure_autopsy_V3-EXQ-1082_2026-09-24"

# Counting anchors (rows, distinct actions, live rows, alpha read-back, acts alignment) are
# reachable by construction under fixed-length uniform-random collection with reset-on-done;
# the canary anchor is the gate's own pinned-value check; and the positive-control anchor runs
# the SHIPPED `_shuffle_verdict` predicate -- the very function the load-bearing criterion
# routes on -- against a ridge head that is action-aware BY CONSTRUCTION, so the bar it must
# clear is reachable by that control by construction rather than by a hand-written narrower
# predicate (the V3-EXQ-778d failure shape).
ANCHOR_REACHABILITY_EXEMPT = (
    "counting anchors reachable by construction (random-action fixed-length live batteries "
    "with reset-on-done); canary anchor is the gate's own pinned-value check; "
    "positive-control anchor runs the SHIPPED _ratio_vs_blind_ci predicate -- the very "
    "function C1 routes on -- against an action-aware-by-construction ridge head, so the "
    "bar is reachable by that control by construction rather than by a narrower hand-written "
    "predicate")
# Load-bearing bars are seed COUNTS over three-valued verdicts, and BOTH directions of
# starvation already self-report rather than masquerading as a FAIL: C1's statistic is
# certified to have room by the positive-control precondition (same predicate, same battery);
# C2's per-seed contrast returns `cannot_determine` -- not "differ" and not "does not differ"
# -- on a degenerate or zero-width bootstrap CI, and `non_degenerate` additionally requires
# every realised CI width > 0. So a starved C2 lands as undetermined + non_degenerate:false,
# which is the distinction this check exists to protect. Note also that "the arms do not
# differ" is one of the two disjuncts of the autopsy's OWN declared null, i.e. a legitimate
# informative outcome here, not only a failure-to-measure.
CRITERION_ACHIEVABLE_RANGE_EXEMPT = (
    "load-bearing bars are seed counts over three-valued verdicts; C1's range is certified "
    "per seed by positive_control_cross_exceeds_blind_null_seeds, which runs C1's OWN "
    "statistic (_ratio_vs_blind_ci) on an action-aware-by-construction ridge head and "
    "self-routes substrate_not_ready_requeue when that bar is out of reach; a starved C2 "
    "self-reports as cannot_determine + non_degenerate:false rather than as a FAIL")

SEEDS = [42, 123, 456]
SELF_DIM = 16
WORLD_DIM = 16
GRID_SIZE = 5
N_HAZARDS = 1
N_RESOURCES = 1
P0_STEPS = 3600
STEPS_PER_EPISODE = 90
EPISODES_PER_RUN = P0_STEPS // STEPS_PER_EPISODE      # 40 -- the [train] denominator
BATCH_K = 8
MIN_BUF_BEFORE_TRAIN = 16
BUF_CAP = 4096
LR = 1e-3
MAX_GRAD_NORM = 1.0

# The two arms. alpha_world EXPLICIT in both; 0.3 is from_dims' default but is still passed
# explicitly so the MECH-307 assert covers it and the manifest records it per cell.
ARMS: List[Tuple[str, float]] = [("ARM_ALPHA_0p3", 0.3), ("ARM_ALPHA_0p9", 0.9)]
OPERATING_POINT_ARM = "ARM_ALPHA_0p9"       # C1 is read here (SD-008's stable floor)

LIVE_BATTERY_N = 512        # 1082's LIVE_BATTERY_N, per the lineage
POST_RESET_SKIP = 3         # transitions skipped after each reset (EMA residual 1e-3 at 0.9)
BATTERY_RNG_XOR = 0xB477E2  # 1082's, so the action draw sequence matches the lineage
SHUFFLE_GEN_SEED = 20260924 # the shuffle permutation is stochastic -> pin it
N_SHUFFLE_DRAWS = 16        # recorded draw-variance of the shuffle ratio (never gates)
MIN_BATTERY_ROWS = 32       # persistence_skill_gate.MIN_ROWS
MIN_DISTINCT_ACTIONS = GATE_MIN_DISTINCT
RIDGE_REL_LAMBDA = 1e-3
SEEDS_REQUIRED = 2          # of 3, as 1082

_ZG = ZGoalStreamAccumulator()


def _to_b(x: Any, device: Any) -> torch.Tensor:
    t = x if isinstance(x, torch.Tensor) else torch.as_tensor(x, dtype=torch.float32)
    return (t.unsqueeze(0) if t.ndim == 1 else t).to(device)


def _make_env(seed: int, invert_action_map: bool = False) -> CausalGridWorldV2:
    """V3-EXQ-1079 :258-267 verbatim. The inversion swaps the two axis pairs (0<->1, 2<->3)
    and preserves every index > 3 (index 4 is the stay action at action_dim 5)."""
    env = CausalGridWorldV2(seed=seed, size=GRID_SIZE, num_hazards=N_HAZARDS,
                            num_resources=N_RESOURCES, use_proxy_fields=True)
    if invert_action_map:
        am = env._action_map
        env._action_map = {0: am[1], 1: am[0], 2: am[3], 3: am[2],
                           **{k: v for k, v in am.items() if k > 3}}
    return env


def _config_slice(alpha_world: float) -> Dict[str, Any]:
    """Declares ONLY what the cell's computation reads."""
    return {
        "env": {"size": GRID_SIZE, "num_hazards": N_HAZARDS,
                "num_resources": N_RESOURCES, "use_proxy_fields": True},
        "dims": {"self_dim": SELF_DIM, "world_dim": WORLD_DIM},
        "latent": {"alpha_world": float(alpha_world)},
        "schedule": {"p0_steps": P0_STEPS, "batch_k": BATCH_K, "lr": LR,
                     "min_buf": MIN_BUF_BEFORE_TRAIN, "max_grad_norm": MAX_GRAD_NORM,
                     "buf_cap": BUF_CAP, "reset_on_done": True,
                     "steps_per_episode": STEPS_PER_EPISODE},
        "battery": {"n_live": LIVE_BATTERY_N, "post_reset_skip": POST_RESET_SKIP,
                    "rng_xor": BATTERY_RNG_XOR, "invert_action_map_pair": True},
        # The shuffle permutation is STOCHASTIC, so its seed and draw count change the
        # recorded shuffle_ratio / shuffle_draw_spread. They are declared here because this
        # driver emits CROSS-DRIVER-reusable fingerprints
        # (include_driver_script_in_hash=False): under-declaring a readout-affecting
        # constant is a false-cache-HIT, which corrupts a conclusion rather than merely
        # wasting compute (arm_reuse_fingerprint_plan.md 7b; V3-EXQ-798's SSL_BIN_EDGES).
        "shuffle": {"gen_seed": SHUFFLE_GEN_SEED, "n_draws": N_SHUFFLE_DRAWS,
                    "ratio_floor": RATIO_FLOOR, "skill_floor": SKILL_FLOOR},
        "positive_control": {"ridge_rel_lambda": RIDGE_REL_LAMBDA},
        "margin": {"use_world_interventional": False},
    }


def _build(seed: int, alpha_world: float) -> Tuple[CausalGridWorldV2, REEAgent]:
    torch.manual_seed(seed)
    env = _make_env(seed)
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim, world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim, self_dim=SELF_DIM, world_dim=WORLD_DIM,
        alpha_world=float(alpha_world))
    # MECH-307 guard: from_dims silently swallows unknown kwargs. Assert it landed.
    if float(getattr(cfg.latent, "alpha_world", -1.0)) != float(alpha_world):
        raise RuntimeError("from_dims did not thread alpha_world")
    if bool(getattr(cfg.e2, "use_world_interventional", False)):
        raise RuntimeError("OFF arm must not carry the interventional margin")
    return env, REEAgent(cfg)


def _sense_zworld(agent: REEAgent, obs: Dict[str, Any]) -> torch.Tensor:
    obs_harm = obs.get("harm_obs", None)
    return agent.sense(_to_b(obs["body_state"], agent.device),
                       _to_b(obs["world_state"], agent.device),
                       obs_harm=(_to_b(obs_harm, agent.device)
                                 if obs_harm is not None else None)
                       ).z_world.detach().reshape(-1).clone()


def _reset(env: CausalGridWorldV2) -> Dict[str, Any]:
    obs = env.reset()
    return obs[-1] if isinstance(obs, tuple) else obs


def _onehot(idx: int, a_dim: int) -> torch.Tensor:
    a = torch.zeros(a_dim, dtype=torch.float32)
    a[idx] = 1.0
    return a


def _collect_live_battery(agent: REEAgent, env: CausalGridWorldV2, rng: random.Random,
                          n: int) -> Dict[str, Any]:
    """LIVE battery, V3-EXQ-1082 :292-333 verbatim. done-handling ported from V3-EXQ-1073
    (:1054-1056): unpack done, on done -> env.reset() and prev=None, so the dying transition
    is never recorded. Also skips the first POST_RESET_SKIP transitions of every episode."""
    obs = _reset(env)
    rows: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    health: List[Tuple[float, float]] = []
    ws_norm: List[float] = []
    prev: Optional[Tuple[torch.Tensor, torch.Tensor, float]] = None
    since_reset = 0
    n_resets = 0
    causes: Dict[str, int] = {}
    guard = 0
    while len(rows) < n and guard < n * 20:
        guard += 1
        z = _sense_zworld(agent, obs)
        h_now = float(env.agent_health)
        if (prev is not None and since_reset > POST_RESET_SKIP
                and bool(torch.isfinite(z).all())):
            rows.append((prev[0], prev[1], z))
            health.append((prev[2], h_now))
            ws_norm.append(float(torch.as_tensor(obs["world_state"]).float().norm()))
        idx = rng.randrange(env.action_dim)
        a = _onehot(idx, env.action_dim)
        _, _, done, info, obs = env.step(a.unsqueeze(0).to(agent.device))
        prev = (z, a, h_now)
        since_reset += 1
        if done:
            c = str((info or {}).get("done_cause", ""))
            causes[c] = causes.get(c, 0) + 1
            obs = _reset(env)
            prev = None
            since_reset = 0
            n_resets += 1
    if not rows:
        raise RuntimeError("battery collected zero rows")
    z0 = torch.stack([r[0] for r in rows])
    acts = torch.stack([r[1] for r in rows])
    z1 = torch.stack([r[2] for r in rows])
    n_dead = sum(1 for h0, h1 in health if h0 <= 0.0 or h1 <= 0.0)
    return {"z0": z0, "acts": acts, "z1": z1, "n_resets": n_resets, "causes": causes,
            "n_postdeath_rows": n_dead, "min_health": min(min(h) for h in health),
            "max_world_state_norm": max(ws_norm),
            "mean_world_state_norm": sum(ws_norm) / len(ws_norm)}


def _battery_mse(head: Any, z0: torch.Tensor, acts: torch.Tensor,
                 z1: torch.Tensor) -> float:
    with torch.no_grad():
        return float(((head(z0, acts) - z1) ** 2).mean().item())


def _swap_errors(head: Any, z0: torch.Tensor, acts: torch.Tensor, z1: torch.Tensor
                 ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-row squared error at the TRUE action, and the per-row MEAN squared error over
    every OTHER action (V3-EXQ-1079/1082 verbatim). Same rows, same targets."""
    a_dim = int(acts.shape[-1])
    idx = acts.argmax(-1)
    with torch.no_grad():
        e_true = per_row_squared_error(head(z0, acts), z1)
        alts = [per_row_squared_error(
                    head(z0, F.one_hot((idx + k) % a_dim, a_dim).float()), z1)
                for k in range(1, a_dim)]
        e_swap = torch.stack(alts).mean(dim=0)
    return e_true.double(), e_swap.double()


def _d_from(e_true: torch.Tensor, e_swap: torch.Tensor) -> Optional[float]:
    den = float(e_swap.sum() + e_true.sum())
    return (float(e_swap.sum() - e_true.sum()) / den) if den > 0 else None


def _pct(sorted_vals: List[float], q: float) -> float:
    n = len(sorted_vals)
    pos = q * (n - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return float(sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac)


def _classify(lo: Optional[float], hi: Optional[float], pos: str, neg: str) -> str:
    if lo is None or hi is None:
        return "cannot_determine"
    if lo > 0.0:
        return pos
    if hi < 0.0:
        return neg
    return "cannot_determine"


def _d_act_ci(e_true: torch.Tensor, e_swap: torch.Tensor, seed_offset: int
              ) -> Dict[str, Any]:
    """Paired row bootstrap (V3-EXQ-1079/1082 verbatim). Analytic null 0."""
    n = int(e_true.shape[0])
    d = _d_from(e_true, e_swap)
    out: Dict[str, Any] = {"d_act": d, "ci_low": None, "ci_high": None,
                           "n_rows": n, "n_bootstrap": 0}
    if d is None or n < MIN_BATTERY_ROWS:
        out["status"] = "cannot_determine"
        return out
    gen = torch.Generator(device="cpu")
    gen.manual_seed(BOOTSTRAP_SEED + seed_offset)
    idx = torch.randint(0, n, (N_BOOTSTRAP, n), generator=gen)
    bt, bs = e_true[idx].sum(1), e_swap[idx].sum(1)
    den = bs + bt
    ok = den > 0
    vals = sorted(((bs - bt)[ok] / den[ok]).tolist())
    if len(vals) < N_BOOTSTRAP // 2:
        out["status"] = "cannot_determine"
        return out
    a = (1.0 - CI_LEVEL) / 2.0
    out.update(ci_low=_pct(vals, a), ci_high=_pct(vals, 1.0 - a), n_bootstrap=len(vals))
    out["status"] = _classify(out["ci_low"], out["ci_high"], "reads_action", "swap_better")
    return out


def _contrast_ci(et_hi: torch.Tensor, es_hi: torch.Tensor, et_lo: torch.Tensor,
                 es_lo: torch.Tensor, seed_offset: int) -> Dict[str, Any]:
    """CI on d_act(0.9) - d_act(0.3). Row INDICES are resampled JOINTLY (V3-EXQ-1079
    construction), which is valid only because both arms' batteries are driven by the same
    rng from the same env seed, so row i is the same (state, action) under both encoders --
    asserted by the acts_hash alignment precondition."""
    n = int(et_hi.shape[0])
    d_hi, d_lo = _d_from(et_hi, es_hi), _d_from(et_lo, es_lo)
    out: Dict[str, Any] = {"delta": None if (d_hi is None or d_lo is None)
                           else d_hi - d_lo, "ci_low": None, "ci_high": None,
                           "n_rows": n, "n_bootstrap": 0}
    if out["delta"] is None or n < MIN_BATTERY_ROWS or int(et_lo.shape[0]) != n:
        out["status"] = "cannot_determine"
        return out
    gen = torch.Generator(device="cpu")
    gen.manual_seed(BOOTSTRAP_SEED + seed_offset)
    idx = torch.randint(0, n, (N_BOOTSTRAP, n), generator=gen)
    th, sh = et_hi[idx].sum(1), es_hi[idx].sum(1)
    tl, sl = et_lo[idx].sum(1), es_lo[idx].sum(1)
    dh, dl = sh + th, sl + tl
    ok = (dh > 0) & (dl > 0)
    vals = sorted((((sh - th) / dh) - ((sl - tl) / dl))[ok].tolist())
    if len(vals) < N_BOOTSTRAP // 2:
        out["status"] = "cannot_determine"
        return out
    a = (1.0 - CI_LEVEL) / 2.0
    out.update(ci_low=_pct(vals, a), ci_high=_pct(vals, 1.0 - a), n_bootstrap=len(vals))
    out["status"] = _classify(out["ci_low"], out["ci_high"], "grows", "shrinks")
    return out


def _ratio_vs_blind_ci(e_head_o: torch.Tensor, e_head_c: torch.Tensor,
                       e_id_o: torch.Tensor, e_id_c: torch.Tensor,
                       seed_offset: int) -> Dict[str, Any]:
    """THE LOAD-BEARING statistic (user option B, 2026-09-24): a CI on

        cross_ratio - blind_null
        = MSE_cf(head)/MSE_orig(head)  -  MSE_cf(identity)/MSE_orig(identity)

    The two batteries are DIFFERENT row sets collected in different envs, so this is a
    TWO-SAMPLE bootstrap: original rows and inverted-map rows are resampled INDEPENDENTLY
    (unlike `_contrast_ci`, where the rows are shared and the resample is joint). Null is 0:
    an action-blind predictor moves the ratio by exactly the battery's own difficulty
    difference, which is what `blind_null` measures on the same pair.

    Row-SUM errors are used throughout (per_row_squared_error's convention); the constant
    n*dim factor cancels in every ratio, so these ratios equal the MSE ratios.
    """
    no, nc = int(e_head_o.shape[0]), int(e_head_c.shape[0])
    out: Dict[str, Any] = {"delta": None, "ci_low": None, "ci_high": None,
                           "cross_ratio": None, "blind_null": None,
                           "n_rows_orig": no, "n_rows_cf": nc, "n_bootstrap": 0}
    if no < MIN_BATTERY_ROWS or nc < MIN_BATTERY_ROWS:
        out["status"] = "cannot_determine"
        return out
    so, sc = float(e_head_o.sum()), float(e_head_c.sum())
    io, ic = float(e_id_o.sum()), float(e_id_c.sum())
    if so <= 0 or io <= 0:
        out["status"] = "cannot_determine"
        return out
    out["cross_ratio"] = sc / so
    out["blind_null"] = ic / io
    out["delta"] = out["cross_ratio"] - out["blind_null"]
    gen = torch.Generator(device="cpu")
    gen.manual_seed(BOOTSTRAP_SEED + seed_offset)
    io_idx = torch.randint(0, no, (N_BOOTSTRAP, no), generator=gen)
    ic_idx = torch.randint(0, nc, (N_BOOTSTRAP, nc), generator=gen)
    ho, hc = e_head_o[io_idx].sum(1), e_head_c[ic_idx].sum(1)
    do, dc = e_id_o[io_idx].sum(1), e_id_c[ic_idx].sum(1)
    ok = (ho > 0) & (do > 0)
    vals = sorted(((hc / ho) - (dc / do))[ok].tolist())
    if len(vals) < N_BOOTSTRAP // 2:
        out["status"] = "cannot_determine"
        return out
    a = (1.0 - CI_LEVEL) / 2.0
    out.update(ci_low=_pct(vals, a), ci_high=_pct(vals, 1.0 - a), n_bootstrap=len(vals))
    out["status"] = _classify(out["ci_low"], out["ci_high"],
                              "exceeds_blind_null", "below_blind_null")
    return out


def _dd_ci(eth_hi: torch.Tensor, esh_hi: torch.Tensor, eth_lo: torch.Tensor,
           esh_lo: torch.Tensor, etp_hi: torch.Tensor, esp_hi: torch.Tensor,
           etp_lo: torch.Tensor, esp_lo: torch.Tensor, seed_offset: int) -> Dict[str, Any]:
    """C2's ATTRIBUTION statistic -- V3-EXQ-1079's positive-control contrast (1079 :84-91,
    :755-758) carried into the GROWS direction as a difference of differences:

        [d_act_head(0.9) - d_act_head(0.3)] - [d_act_pc(0.9) - d_act_pc(0.3)]

    1079 split its SHRINKS label on whether the PC contrast moved too. The measured failure
    this run was refused for (2026-09-24 red-team F2) is the GROWS analogue: the head's
    contrast can be POSITIVE while the closed-form ridge ceiling's contrast is LARGER, so
    the alpha gain belongs to the battery -- more linearly readable action information
    reached z_world -- rather than to the head reading better. Null is 0. Rows are shared
    across all four error vectors within a seed (both arms are row-aligned, asserted), so
    the resample is JOINT.
    """
    n = int(eth_hi.shape[0])
    dh = _d_from(eth_hi, esh_hi)
    dl = _d_from(eth_lo, esh_lo)
    ph = _d_from(etp_hi, esp_hi)
    pl = _d_from(etp_lo, esp_lo)
    out: Dict[str, Any] = {"dd": None, "ci_low": None, "ci_high": None,
                           "head_delta": None, "pc_delta": None, "n_bootstrap": 0}
    if None in (dh, dl, ph, pl) or n < MIN_BATTERY_ROWS:
        out["status"] = "cannot_determine"
        return out
    out["head_delta"] = dh - dl
    out["pc_delta"] = ph - pl
    out["dd"] = out["head_delta"] - out["pc_delta"]
    gen = torch.Generator(device="cpu")
    gen.manual_seed(BOOTSTRAP_SEED + seed_offset)
    idx = torch.randint(0, n, (N_BOOTSTRAP, n), generator=gen)

    def _d(t: torch.Tensor, sw: torch.Tensor) -> torch.Tensor:
        a, b = t[idx].sum(1), sw[idx].sum(1)
        return (b - a) / (b + a)

    den_ok = True
    for t, sw in ((eth_hi, esh_hi), (eth_lo, esh_lo), (etp_hi, esp_hi), (etp_lo, esp_lo)):
        if float((t[idx].sum(1) + sw[idx].sum(1)).min()) <= 0:
            den_ok = False
    if not den_ok:
        out["status"] = "cannot_determine"
        return out
    vals = sorted(((_d(eth_hi, esh_hi) - _d(eth_lo, esh_lo))
                   - (_d(etp_hi, esp_hi) - _d(etp_lo, esp_lo))).tolist())
    a = (1.0 - CI_LEVEL) / 2.0
    out.update(ci_low=_pct(vals, a), ci_high=_pct(vals, 1.0 - a), n_bootstrap=len(vals))
    out["status"] = _classify(out["ci_low"], out["ci_high"],
                              "head_gains_beyond_ceiling", "ceiling_gains_more")
    return out


def _pc_contrast_ci(etp_hi: torch.Tensor, esp_hi: torch.Tensor, etp_lo: torch.Tensor,
                    esp_lo: torch.Tensor, seed_offset: int) -> Dict[str, Any]:
    """V3-EXQ-1079's LITERAL positive-control contrast, pc_d_act(0.9) - pc_d_act(0.3),
    recorded verbatim so the port is checkable against 1079 rather than only inferred
    from the difference-of-differences above."""
    return _contrast_ci(etp_hi, esp_hi, etp_lo, esp_lo, seed_offset=seed_offset)


def _ridge_head(buf: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> Any:
    """POSITIVE CONTROL (V3-EXQ-1079/1082 verbatim): z1_hat = z0 + [1, z0, a, z0 (x) a] @ W,
    ridge-fitted on the cell's own P0 buffer. Action-aware BY CONSTRUCTION, so it is the
    right positive control for an action-sensitivity statistic."""
    z0 = torch.stack([b[0] for b in buf]).double()
    a = torch.stack([b[1] for b in buf]).double()
    z1 = torch.stack([b[2] for b in buf]).double()

    def phi(z: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        cross = (z.unsqueeze(-1) * act.unsqueeze(-2)).reshape(z.shape[0], -1)
        return torch.cat([torch.ones(z.shape[0], 1, dtype=z.dtype), z, act, cross], -1)

    P = phi(z0, a)
    G = P.T @ P
    lam = RIDGE_REL_LAMBDA * float(torch.diagonal(G).mean())
    W = torch.linalg.solve(G + lam * torch.eye(G.shape[0], dtype=G.dtype), P.T @ (z1 - z0))

    def head(z: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        zd, ad = z.double(), act.double()
        return (zd + phi(zd, ad) @ W).float()
    return head


def _shuffle_verdict(head: Any, z0: torch.Tensor, acts: torch.Tensor, z1: torch.Tensor,
                     seed_offset: int) -> Any:
    """RECORDED, NEVER GATES since the 2026-09-24 option-B re-point (it was option C's
    load-bearing readout). Same-rows action_shuffle form, where RATIO_FLOOR=1.0 IS the true
    null. Gate defaults on both floors; the permutation is stochastic so the generator is
    pinned for reproducibility."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(SHUFFLE_GEN_SEED + seed_offset)
    return readiness_verdict(head, z0, acts, z1, counterfactual_battery=None,
                             min_rows=MIN_BATTERY_ROWS,
                             min_distinct_actions=MIN_DISTINCT_ACTIONS,
                             ratio_floor=RATIO_FLOOR, skill_floor=SKILL_FLOOR,
                             generator=gen)


def _shuffle_draw_spread(head: Any, z0: torch.Tensor, acts: torch.Tensor,
                         z1: torch.Tensor, seed_offset: int) -> Dict[str, Any]:
    """RECORDED, never gates: the shuffle permutation is random, so report the spread over
    N_SHUFFLE_DRAWS independent draws. A load-bearing verdict taken from ONE draw whose
    spread straddles the floor is a verdict the reader must be able to see is fragile."""
    vals: List[float] = []
    for k in range(N_SHUFFLE_DRAWS):
        v = _shuffle_verdict(head, z0, acts, z1, seed_offset + 1000 * (k + 1))
        if v.ratio is not None and math.isfinite(v.ratio):
            vals.append(float(v.ratio))
    if not vals:
        return {"n_draws": 0, "min": None, "max": None, "mean": None,
                "frac_above_floor": None}
    return {"n_draws": len(vals), "min": min(vals), "max": max(vals),
            "mean": sum(vals) / len(vals),
            "frac_above_floor": sum(1 for v in vals if v > RATIO_FLOOR) / len(vals)}


def _action_sep(head: Any, z0: torch.Tensor, acts: torch.Tensor) -> float:
    """Mean over rows and alternative actions of ||f(z,a) - f(z,a_alt)||_2."""
    a_dim = int(acts.shape[-1])
    idx = acts.argmax(-1)
    with torch.no_grad():
        p = head(z0, acts)
        d = [(p - head(z0, F.one_hot((idx + k) % a_dim, a_dim).float())).norm(dim=-1)
             for k in range(1, a_dim)]
    return float(torch.stack(d).mean().item())


def _encoder_equal(a: REEAgent, b: REEAgent) -> bool:
    sa, sb = a.latent_stack.state_dict(), b.latent_stack.state_dict()
    return sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)


def _hash(*ts: torch.Tensor) -> str:
    h = hashlib.sha256()
    for t in ts:
        h.update(t.detach().double().cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def _run_cell(arm: str, alpha_world: float, seed: int) -> Dict[str, Any]:
    print(f"Seed {seed} Condition {arm}", flush=True)
    with arm_cell(seed, config_slice=_config_slice(alpha_world),
                  script_path=Path(__file__), config_slice_declared=True,
                  include_driver_script_in_hash=False) as cell:
        env, agent = _build(seed, alpha_world)
        rng = random.Random(seed)

        # --- P0 training, reset-on-done (1073/1082 pattern), OFF objective only ---------
        opt = torch.optim.Adam(agent.e2.parameters(), lr=LR)
        buf: Deque[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = deque(maxlen=BUF_CAP)
        obs = _reset(env)
        prev: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        n_train_resets = 0
        train_causes: Dict[str, int] = {}
        for step in range(1, P0_STEPS + 1):
            if step % STEPS_PER_EPISODE == 0:
                ep = step // STEPS_PER_EPISODE
                print(f"  [train] {arm} seed={seed} ep {ep}/{EPISODES_PER_RUN} "
                      f"phase=P0", flush=True)
            z = _sense_zworld(agent, obs)
            if prev is not None and bool(torch.isfinite(z).all()):
                buf.append((prev[0], prev[1], z))
            a = _onehot(rng.randrange(env.action_dim), env.action_dim)
            _, _, done, info, obs = env.step(a.unsqueeze(0).to(agent.device))
            prev = (z, a)
            if done:
                c = str((info or {}).get("done_cause", ""))
                train_causes[c] = train_causes.get(c, 0) + 1
                obs = _reset(env)
                prev = None
                n_train_resets += 1
            if len(buf) < MIN_BUF_BEFORE_TRAIN:
                continue
            pool = list(buf)
            batch = pool if len(pool) <= BATCH_K else rng.sample(pool, BATCH_K)
            b0 = torch.stack([t[0] for t in batch]).to(agent.device)
            ba = torch.stack([t[1] for t in batch]).to(agent.device)
            b1 = torch.stack([t[2] for t in batch]).to(agent.device)
            opt.zero_grad(set_to_none=True)
            loss = F.mse_loss(agent.e2.world_forward(b0, ba), b1)
            if not math.isfinite(float(loss.detach().item())):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.e2.parameters(), max_norm=MAX_GRAD_NORM)
            opt.step()

        # --- LIVE batteries from FRESH agents at THIS arm's alpha -----------------------
        # alpha changes the encoder, so unlike 1082 the battery agent is per-arm. Same seed
        # -> the frozen encoder is identical to the trained cell's initial encoder.
        _, ag_b = _build(seed, alpha_world)
        enc_eq = _encoder_equal(agent, ag_b)
        bat = _collect_live_battery(ag_b, _make_env(seed),
                                    random.Random(seed ^ BATTERY_RNG_XOR), LIVE_BATTERY_N)
        _, ag_cf = _build(seed, alpha_world)
        bat_cf = _collect_live_battery(ag_cf, _make_env(seed, invert_action_map=True),
                                       random.Random(seed ^ BATTERY_RNG_XOR),
                                       LIVE_BATTERY_N)
        bz0, bacts, bz1 = bat["z0"], bat["acts"], bat["z1"]
        cf = (bat_cf["z0"], bat_cf["acts"], bat_cf["z1"])
        head = agent.e2.world_forward
        untrained = ag_b.e2.world_forward

        # --- LOAD-BEARING: cross-battery inverted-map ratio vs its ACTION-BLIND NULL ----
        # (USER option B, 2026-09-24. The bar is the blind null, not 1.0: the two batteries
        # are different row sets, so an action-blind predictor already moves the ratio by
        # their difficulty difference. `readiness_verdict` is handed ratio_floor=blind_null
        # so the gate's MECH-573 skill clause is retained and only the bar is re-denominated.)
        cross_ratio, cross_mse_o, cross_mse_c = battery_pair_ratio(
            head, (bz0, bacts, bz1), cf)
        id_orig = identity_predictor_mse(bz0, bz1)
        id_cf = identity_predictor_mse(cf[0], cf[2])
        blind_null = (id_cf / id_orig) if id_orig > 0 else None
        with torch.no_grad():
            e_head_o = per_row_squared_error(head(bz0, bacts), bz1).double()
            e_head_c = per_row_squared_error(head(cf[0], cf[1]), cf[2]).double()
            e_id_o = per_row_squared_error(bz0, bz1).double()
            e_id_c = per_row_squared_error(cf[0], cf[2]).double()
        vs_blind = _ratio_vs_blind_ci(e_head_o, e_head_c, e_id_o, e_id_c, seed_offset=seed)
        v_blindbar = readiness_verdict(
            head, bz0, bacts, bz1, counterfactual_battery=cf,
            min_rows=MIN_BATTERY_ROWS, min_distinct_actions=MIN_DISTINCT_ACTIONS,
            ratio_floor=(blind_null if blind_null is not None else RATIO_FLOOR),
            skill_floor=SKILL_FLOOR)
        print(format_verdict(v_blindbar,
                             f"{arm} seed={seed} [LOAD-BEARING cross-battery vs blind null "
                             f"{blind_null}]"), flush=True)
        # LOAD-BEARING per-cell predicate: the gate says ready AND the CI on
        # (cross_ratio - blind_null) excludes 0, so a point estimate cannot carry it alone.
        reads_invmap = bool(v_blindbar.status == "ready"
                            and vs_blind.get("ci_low") is not None
                            and vs_blind["ci_low"] > 0.0)

        # --- RECORDED, never gating: the autopsy's literal raw >1.0 bar ----------------
        v_cross = readiness_verdict(head, bz0, bacts, bz1, counterfactual_battery=cf,
                                    min_rows=MIN_BATTERY_ROWS,
                                    min_distinct_actions=MIN_DISTINCT_ACTIONS,
                                    ratio_floor=RATIO_FLOOR, skill_floor=SKILL_FLOOR)
        cross_reads_raw = bool(cross_ratio is not None and cross_ratio > RATIO_FLOOR)
        cross_reads_blind = bool(cross_ratio is not None and blind_null is not None
                                 and cross_ratio > blind_null)

        # --- RECORDED, never gating: the same-rows shuffle verdict ---------------------
        # Demoted from load-bearing on 2026-09-24: its ratio clause is d_act > 0 restated
        # ((1+d_act)/(1-d_act) == S/T identically), and reanalysis
        # reanalysis_sdppb5_off_head_action_read_alpha09_live_battery_20260924T171654Z
        # already settled it from V3-EXQ-1082's landed per-row errors (ready 3/3 at 0.9).
        # Kept because it is free and is what makes this artifact comparable to 1082.
        v_shuf = _shuffle_verdict(head, bz0, bacts, bz1, seed_offset=seed)
        shuf_spread = _shuffle_draw_spread(head, bz0, bacts, bz1, seed_offset=seed)

        # --- RECORDED: d_act (1082 verbatim), positive control, untrained head ----------
        et, es = _swap_errors(head, bz0, bacts, bz1)
        d_head = _d_act_ci(et, es, seed_offset=seed)
        pc = _ridge_head(list(buf))
        pt, ps = _swap_errors(pc, bz0, bacts, bz1)
        d_pc = _d_act_ci(pt, ps, seed_offset=seed + 7)
        # READINESS positive control on the SAME statistic the load-bearing criterion routes
        # on (Step 3.5 same-statistic rule). C1 is now the cross-battery ratio vs its
        # action-blind null, so the control must clear THAT bar, not the shuffle bar: an
        # action-aware-by-construction ridge head fitted on the ORIGINAL env's transitions
        # must degrade more than an action-blind predictor when the action map is inverted.
        # If it cannot, this battery pair cannot test the question and a below-bar trained
        # head is not evidence -> substrate_not_ready_requeue.
        with torch.no_grad():
            e_pc_o = per_row_squared_error(pc(bz0, bacts), bz1).double()
            e_pc_c = per_row_squared_error(pc(cf[0], cf[1]), cf[2]).double()
        pc_vs_blind = _ratio_vs_blind_ci(e_pc_o, e_pc_c, e_id_o, e_id_c,
                                         seed_offset=seed + 41)
        pc_cross_exceeds_blind = bool(pc_vs_blind.get("ci_low") is not None
                                      and pc_vs_blind["ci_low"] > 0.0)
        # The shuffle-form control is RECORDED alongside (it was the option-C readiness gate).
        v_pc_shuf = _shuffle_verdict(pc, bz0, bacts, bz1, seed_offset=seed + 7)
        ut, us = _swap_errors(untrained, bz0, bacts, bz1)
        mse_init = _battery_mse(untrained, bz0, bacts, bz1)
        mse_final = _battery_mse(head, bz0, bacts, bz1)
        with torch.no_grad():
            pv = persistence_verdict(head(bz0, bacts), bz1, bz0)

        print(f"  [readout] {arm} seed={seed} alpha={alpha_world} | LOAD-BEARING "
              f"cross_ratio={cross_ratio} blind_null={blind_null} "
              f"delta={vs_blind['delta']} CI=[{vs_blind['ci_low']}, {vs_blind['ci_high']}] "
              f"{vs_blind['status']} reads_invmap={reads_invmap} | RECORDED raw>1="
              f"{cross_reads_raw} shuffle_ratio={v_shuf.ratio}/{v_shuf.status} | "
              f"d_act={d_head['d_act']} CI=[{d_head['ci_low']}, {d_head['ci_high']}] | "
              f"pc_shuf={v_pc_shuf.status} | postdeath={bat['n_postdeath_rows']}/"
              f"{bat_cf['n_postdeath_rows']}", flush=True)

        row: Dict[str, Any] = {
            "arm": arm, "alpha_world_requested": float(alpha_world),
            "alpha_world": float(agent.config.latent.alpha_world), "seed": seed,
            # ---- LOAD-BEARING: cross-battery inverted-map ratio vs its blind null ----
            "reads_inverted_map": 1 if reads_invmap else 0,
            "blindbar_status": v_blindbar.status,
            "blindbar_reason": v_blindbar.reason,
            "blindbar_ratio_floor_used": blind_null,
            "blindbar_verdict_full": v_blindbar.to_dict(),
            "cross_minus_blind": vs_blind["delta"],
            "cross_minus_blind_ci_low": vs_blind["ci_low"],
            "cross_minus_blind_ci_high": vs_blind["ci_high"],
            "cross_minus_blind_status": vs_blind["status"],
            "cross_minus_blind_n_bootstrap": vs_blind["n_bootstrap"],
            # ---- RECORDED, never gates: same-rows shuffle form ----
            "shuffle_status": v_shuf.status,
            "shuffle_ratio": v_shuf.ratio,
            "shuffle_ratio_floor": float(RATIO_FLOOR),
            "shuffle_skill": v_shuf.skill,
            "shuffle_skill_floor": float(SKILL_FLOOR),
            "shuffle_reason": v_shuf.reason,
            "shuffle_verdict_full": v_shuf.to_dict(),
            "shuffle_draw_spread": shuf_spread,
            # ---- RECORDED: cross-battery inverted-map ratio (autopsy's literal readout) --
            "cross_ratio": cross_ratio,
            "cross_mse_original": cross_mse_o,
            "cross_mse_inverted": cross_mse_c,
            "cross_blind_null": blind_null,
            "cross_ratio_minus_blind_null": (None if (cross_ratio is None
                                                      or blind_null is None)
                                             else cross_ratio - blind_null),
            "cross_reads_raw_bar_1p0": 1 if cross_reads_raw else 0,
            "cross_reads_vs_blind_null": 1 if cross_reads_blind else 0,
            "cross_status_raw_bar": v_cross.status,
            "cross_verdict_full": v_cross.to_dict(),
            "identity_predictor_mse_original": id_orig,
            "identity_predictor_mse_inverted": id_cf,
            # ---- RECORDED: d_act and friends ----
            "d_act": d_head["d_act"], "d_act_ci_low": d_head["ci_low"],
            "d_act_ci_high": d_head["ci_high"], "d_act_status": d_head["status"],
            "pc_d_act": d_pc["d_act"], "pc_d_act_ci_low": d_pc["ci_low"],
            "pc_d_act_ci_high": d_pc["ci_high"], "pc_d_act_status": d_pc["status"],
            "pc_live": bool(d_pc["ci_low"] is not None and d_pc["ci_low"] > 0.0),
            "pc_cross_ratio": pc_vs_blind["cross_ratio"],
            "pc_cross_minus_blind": pc_vs_blind["delta"],
            "pc_cross_minus_blind_ci_low": pc_vs_blind["ci_low"],
            "pc_cross_minus_blind_ci_high": pc_vs_blind["ci_high"],
            "pc_cross_minus_blind_status": pc_vs_blind["status"],
            "pc_cross_exceeds_blind": 1 if pc_cross_exceeds_blind else 0,
            "pc_shuffle_status": v_pc_shuf.status,
            "pc_shuffle_ratio": v_pc_shuf.ratio,
            "pc_shuffle_reads": 1 if v_pc_shuf.status == "ready" else 0,
            "d_act_untrained_head": _d_from(ut, us),
            # ---- reconstruction / readability context (never gates) ----
            "battery_mse_init": mse_init, "battery_mse_final": mse_final,
            "conv_rel_drop": (1.0 - mse_final / mse_init) if mse_init > 0 else None,
            "identity_predictor_mse": id_orig,
            "skill_vs_identity": (1.0 - mse_final / id_orig) if id_orig > 0 else None,
            "model_r2": pv.model_r2, "persistence_r2": pv.persistence_r2,
            "persistence_relative_skill": pv.relative_skill,
            "persistence_status": pv.status,
            "persistence_ci_low": pv.ci_low, "persistence_ci_high": pv.ci_high,
            # ---- live-battery hygiene, BOTH batteries ----
            "encoder_equal_to_battery_agent": enc_eq,
            "battery_hash": _hash(bz0, bacts, bz1),
            "battery_acts_hash": _hash(bacts),
            "battery_cf_acts_hash": _hash(cf[1]),
            "n_rows_battery": int(bz0.shape[0]),
            "n_rows_battery_cf": int(cf[0].shape[0]),
            "n_distinct_actions_battery": int(torch.unique(bacts, dim=0).shape[0]),
            "n_distinct_actions_battery_cf": int(torch.unique(cf[1], dim=0).shape[0]),
            "battery_n_resets": bat["n_resets"], "battery_done_causes": bat["causes"],
            "battery_n_postdeath_rows": bat["n_postdeath_rows"],
            "battery_cf_n_resets": bat_cf["n_resets"],
            "battery_cf_done_causes": bat_cf["causes"],
            "battery_cf_n_postdeath_rows": bat_cf["n_postdeath_rows"],
            "battery_min_health": bat["min_health"],
            "battery_cf_min_health": bat_cf["min_health"],
            "battery_max_world_state_norm": bat["max_world_state_norm"],
            "battery_mean_world_state_norm": bat["mean_world_state_norm"],
            # ---- displacement scale: the DV-symmetry disclosure (see docstring) ----
            "rms_dz_per_dim_battery": float(((bz1 - bz0) ** 2).mean().sqrt().item()),
            "rms_dz_per_dim_battery_cf": float(((cf[2] - cf[0]) ** 2).mean().sqrt().item()),
            "transition_l2_mean": float((bz1 - bz0).norm(dim=-1).mean().item()),
            "transition_l2_mean_cf": float((cf[2] - cf[0]).norm(dim=-1).mean().item()),
            "head_action_separation_l2_mean": _action_sep(head, bz0, bacts),
            "train_n_resets": n_train_resets, "train_done_causes": train_causes,
            "p0_buffer_rows": len(buf),
            # ---- per-row errors so every contrast is re-derivable post hoc ----
            "per_row_se_true": [float(x) for x in et.tolist()],
            "per_row_se_swap_mean": [float(x) for x in es.tolist()],
            "per_row_se_true_pc": [float(x) for x in pt.tolist()],
            "per_row_se_swap_mean_pc": [float(x) for x in ps.tolist()],
        }
        cell.stamp(row)
        _ZG.observe(agent)
    print(f"verdict: {'PASS' if reads_invmap else 'FAIL'}", flush=True)
    return row


def _t(r: Dict[str, Any], k: str) -> torch.Tensor:
    return torch.tensor(r[k], dtype=torch.float64)


def _seed_analysis(rows: List[Dict[str, Any]], seed: int) -> Dict[str, Any]:
    by = {r["arm"]: r for r in rows if r["seed"] == seed}
    res: Dict[str, Any] = {"seed": seed}
    if len(by) != len(ARMS):
        res.update(valid=False, reason="cell missing")
        return res
    hi = by[OPERATING_POINT_ARM]
    lo = by["ARM_ALPHA_0p3"]
    # Pairing validity: both arms' batteries are driven by the same rng from the same env
    # seed, so the ACTION sequence must be bit-identical for the joint bootstrap to pair
    # row i with row i. Encoders differ by construction (that IS the arm), so z differs.
    aligned = bool(hi["battery_acts_hash"] == lo["battery_acts_hash"]
                   and hi["n_rows_battery"] == lo["n_rows_battery"])
    res.update(rows_aligned=aligned,
               encoder_equal=bool(hi["encoder_equal_to_battery_agent"]
                                  and lo["encoder_equal_to_battery_agent"]),
               pc_shuffle_reads=bool(hi["pc_shuffle_reads"] and lo["pc_shuffle_reads"]),
               pc_cross_ok=bool(hi["pc_cross_exceeds_blind"]
                                and lo["pc_cross_exceeds_blind"]),
               pc_live=bool(hi["pc_live"] and lo["pc_live"]))
    # Validity is keyed on the control for the LOAD-BEARING statistic (pc_cross_ok); the
    # shuffle-form control is recorded but no longer gates, since it is no longer C1's form.
    res["valid"] = bool(res["encoder_equal"] and res["pc_cross_ok"])

    # ---- C1 (LOAD-BEARING): the inverted-map ratio vs its action-blind null at 0.9 ----
    res["reads_inverted_map_at_operating_point"] = bool(hi["reads_inverted_map"])
    for tag, cell in (("hi", hi), ("lo", lo)):
        res[f"cross_ratio_{tag}"] = cell["cross_ratio"]
        res[f"cross_blind_null_{tag}"] = cell["cross_blind_null"]
        res[f"cross_minus_blind_{tag}"] = cell["cross_minus_blind"]
        res[f"cross_minus_blind_ci_low_{tag}"] = cell["cross_minus_blind_ci_low"]
        res[f"cross_minus_blind_ci_high_{tag}"] = cell["cross_minus_blind_ci_high"]
        res[f"cross_minus_blind_status_{tag}"] = cell["cross_minus_blind_status"]
        res[f"blindbar_status_{tag}"] = cell["blindbar_status"]

    # ---- C2: do the arms differ on the head's action read? (joint paired bootstrap) ----
    if aligned:
        c = _contrast_ci(_t(hi, "per_row_se_true"), _t(hi, "per_row_se_swap_mean"),
                         _t(lo, "per_row_se_true"), _t(lo, "per_row_se_swap_mean"),
                         seed_offset=seed + 57)
        # V3-EXQ-1079's LITERAL positive-control contrast, recorded verbatim.
        pc = _pc_contrast_ci(_t(hi, "per_row_se_true_pc"), _t(hi, "per_row_se_swap_mean_pc"),
                             _t(lo, "per_row_se_true_pc"), _t(lo, "per_row_se_swap_mean_pc"),
                             seed_offset=seed + 91)
        # ATTRIBUTION (1079's split, GROWS direction): did the HEAD gain beyond the ceiling?
        dd = _dd_ci(_t(hi, "per_row_se_true"), _t(hi, "per_row_se_swap_mean"),
                    _t(lo, "per_row_se_true"), _t(lo, "per_row_se_swap_mean"),
                    _t(hi, "per_row_se_true_pc"), _t(hi, "per_row_se_swap_mean_pc"),
                    _t(lo, "per_row_se_true_pc"), _t(lo, "per_row_se_swap_mean_pc"),
                    seed_offset=seed + 113)
    else:
        blank = {"delta": None, "ci_low": None, "ci_high": None,
                 "status": "cannot_determine", "n_rows": 0, "n_bootstrap": 0}
        c = dict(blank)
        pc = dict(blank)
        dd = {"dd": None, "ci_low": None, "ci_high": None, "head_delta": None,
              "pc_delta": None, "status": "cannot_determine", "n_bootstrap": 0}
    res["alpha_contrast"] = c
    res["alpha_contrast_pc"] = pc
    res["alpha_contrast_attribution"] = dd
    res["arms_differ"] = bool(c["status"] in ("grows", "shrinks"))
    res["head_gains_beyond_ceiling"] = bool(dd["status"] == "head_gains_beyond_ceiling")
    res["ceiling_gains_more"] = bool(dd["status"] == "ceiling_gains_more")

    # ---- RECORDED, never gates: the autopsy's literal raw bar, and bar agreement ----
    res["cross_reads_raw_hi"] = bool(hi["cross_reads_raw_bar_1p0"])
    res["cross_reads_vs_blind_hi"] = bool(hi["cross_reads_vs_blind_null"])
    res["bars_agree_hi"] = bool(res["cross_reads_raw_hi"] == res["cross_reads_vs_blind_hi"])
    res["shuffle_status_hi"] = hi["shuffle_status"]
    res["shuffle_status_lo"] = lo["shuffle_status"]
    res["shuffle_ratio_hi"] = hi["shuffle_ratio"]
    res["shuffle_ratio_lo"] = lo["shuffle_ratio"]
    return res


def _worst(rows: List[Dict[str, Any]], key: str, lowest: bool = True) -> Tuple[float, str]:
    w = (min if lowest else max)(rows, key=lambda r: r[key])
    return float(w[key]), f"{w['arm']}/seed={w['seed']}"


def run_experiment(dry_run: bool = False) -> Tuple[Dict[str, Any], float]:
    t0 = time.perf_counter()
    canary = check_canary()
    print(f"[gate-canary] ok={canary.get('ok')} n_seeds={canary.get('n_seeds')}", flush=True)
    # Dry run: 2 seeds x BOTH arms at the FULL budget (SEEDS_REQUIRED stays reachable).
    seeds = SEEDS[:2] if dry_run else SEEDS
    rows: List[Dict[str, Any]] = []
    for seed in seeds:
        for arm, alpha in ARMS:
            rows.append(_run_cell(arm, alpha, seed))

    per_seed = [_seed_analysis(rows, s) for s in seeds]
    valid = [s for s in per_seed if s["valid"]]
    for s in per_seed:
        att = s.get("alpha_contrast_attribution") or {}
        print(f"  [seed] {s['seed']} valid={s['valid']} "
              f"C1_invmap>blind@0.9={s.get('reads_inverted_map_at_operating_point')} "
              f"(cross={s.get('cross_ratio_hi')} blind={s.get('cross_blind_null_hi')} "
              f"delta={s.get('cross_minus_blind_hi')} "
              f"CI=[{s.get('cross_minus_blind_ci_low_hi')}, "
              f"{s.get('cross_minus_blind_ci_high_hi')}]) | "
              f"C2 alpha_contrast={s.get('alpha_contrast', {}).get('status')} "
              f"head_delta={att.get('head_delta')} pc_delta={att.get('pc_delta')} "
              f"dd={att.get('dd')} attribution={att.get('status')} | RECORDED raw>1="
              f"{s.get('cross_reads_raw_hi')} bars_agree={s.get('bars_agree_hi')} "
              f"shuf_hi={s.get('shuffle_ratio_hi')}", flush=True)

    n_valid = len(valid)
    n_reads = sum(1 for s in valid if s["reads_inverted_map_at_operating_point"])
    n_differ = sum(1 for s in valid if s["arms_differ"])
    n_head_gains = sum(1 for s in valid if s["head_gains_beyond_ceiling"])
    n_ceiling_gains = sum(1 for s in valid if s["ceiling_gains_more"])
    n_aligned = sum(1 for s in per_seed if s.get("rows_aligned"))
    n_enc = sum(1 for s in per_seed if s.get("encoder_equal"))
    n_pc_shuf = sum(1 for s in per_seed if s.get("pc_shuffle_reads"))
    n_pc_cross = sum(1 for s in per_seed if s.get("pc_cross_ok"))
    n_pc_dact = sum(1 for s in per_seed if s.get("pc_live"))

    worst_rows, worst_rows_cell = _worst(rows, "n_rows_battery")
    worst_rows_cf, worst_rows_cf_cell = _worst(rows, "n_rows_battery_cf")
    worst_distinct, worst_distinct_cell = _worst(rows, "n_distinct_actions_battery")
    worst_distinct_cf, worst_distinct_cf_cell = _worst(
        rows, "n_distinct_actions_battery_cf")
    worst_dead, worst_dead_cell = _worst(rows, "battery_n_postdeath_rows", lowest=False)
    worst_dead_cf, worst_dead_cf_cell = _worst(
        rows, "battery_cf_n_postdeath_rows", lowest=False)
    # Worst |read-back alpha - requested alpha| over all cells. Computed inline rather than
    # via _worst() because it is a derived quantity, not a stored per-cell field.
    _alpha_errs = [(abs(r["alpha_world"] - r["alpha_world_requested"]),
                    f"{r['arm']}/seed={r['seed']}") for r in rows]
    alpha_err, alpha_err_cell = max(_alpha_errs) if _alpha_errs else (0.0, "none")

    preconditions = [
        {"name": "alpha_world_threaded_both_arms", "description":
         "worst cell's |alpha_world read back from the built config - requested| (must be 0)",
         "measured": float(alpha_err), "threshold": 0.0, "direction": "upper",
         "offending_cell": alpha_err_cell,
         "control": "explicit from_dims kwarg + MECH-307 assert in _build",
         "met": alpha_err <= 0.0},
        {"name": "battery_rows_all_live", "description":
         "worst cell's count of ORIGINAL-battery rows at or after agent_health <= 0 "
         "(the V3-EXQ-1075/1079 defect; must be 0)",
         "measured": float(worst_dead), "threshold": 0.0, "direction": "upper",
         "offending_cell": worst_dead_cell, "control": "reset-on-done (1073/1082 pattern)",
         "met": worst_dead <= 0},
        {"name": "battery_cf_rows_all_live", "description":
         "worst cell's count of INVERTED-MAP-battery rows at or after agent_health <= 0 "
         "(must be 0; the inverted map changes the trajectory, so this is not implied by "
         "the original battery being live)",
         "measured": float(worst_dead_cf), "threshold": 0.0, "direction": "upper",
         "offending_cell": worst_dead_cf_cell, "control": "same reset-on-done collector",
         "met": worst_dead_cf <= 0},
        {"name": "battery_rows", "description": "worst cell's original live-battery rows",
         "measured": float(worst_rows), "threshold": float(MIN_BATTERY_ROWS),
         "direction": "lower", "offending_cell": worst_rows_cell,
         "control": "fixed-length collection", "met": worst_rows >= MIN_BATTERY_ROWS},
        {"name": "battery_cf_rows", "description":
         "worst cell's inverted-map live-battery rows",
         "measured": float(worst_rows_cf), "threshold": float(MIN_BATTERY_ROWS),
         "direction": "lower", "offending_cell": worst_rows_cf_cell,
         "control": "fixed-length collection", "met": worst_rows_cf >= MIN_BATTERY_ROWS},
        {"name": "battery_distinct_actions", "description":
         "worst cell's distinct actions in the ORIGINAL battery. THE MONOSTRATEGY TRAP: at "
         "1 distinct action a permutation is a no-op and the ratio returns exactly 1.0, "
         "which is not action-blindness but an untestable battery",
         "measured": float(worst_distinct), "threshold": float(MIN_DISTINCT_ACTIONS),
         "direction": "lower", "offending_cell": worst_distinct_cell,
         "control": "uniform random actions", "met": worst_distinct >= MIN_DISTINCT_ACTIONS},
        {"name": "battery_cf_distinct_actions", "description":
         "worst cell's distinct actions in the INVERTED-MAP battery (same trap)",
         "measured": float(worst_distinct_cf), "threshold": float(MIN_DISTINCT_ACTIONS),
         "direction": "lower", "offending_cell": worst_distinct_cf_cell,
         "control": "uniform random actions",
         "met": worst_distinct_cf >= MIN_DISTINCT_ACTIONS},
        {"name": "encoder_equal_seeds", "description":
         "seeds where BOTH arms' trained cell shares the frozen encoder of its own "
         "battery agent (alpha changes the encoder, so this is checked per arm)",
         "measured": float(n_enc), "threshold": float(SEEDS_REQUIRED), "direction": "lower",
         "control": "same seed -> same init; encoder untrained in P0",
         "met": n_enc >= SEEDS_REQUIRED},
        {"name": "positive_control_cross_exceeds_blind_null_seeds", "description":
         "READINESS, SAME STATISTIC AS THE LOAD-BEARING CRITERION (C1): seeds where a ridge "
         "action-aware predictor fitted on each arm's own P0 buffer has its CROSS-BATTERY "
         "inverted-map ratio above the action-blind null on the same pair -- CI lower on "
         "(pc_cross_ratio - blind_null) > 0 -- in BOTH arms. This is the bar C1 routes on, "
         "measured on a head that is action-aware BY CONSTRUCTION. If even that head cannot "
         "clear it, the battery PAIR cannot test inversion-sensitivity and a below-bar "
         "trained head is not evidence of H-difficulty-only",
         "measured": float(n_pc_cross), "threshold": float(SEEDS_REQUIRED),
         "direction": "lower", "control": "ridge z0 + [1, z0, a, z0 x a] @ W",
         "met": n_pc_cross >= SEEDS_REQUIRED},
        {"name": "positive_control_shuffle_reads_seeds", "description":
         "RECORDED (was the option-C readiness gate; no longer C1's form): seeds where the "
         "ridge control returns same-rows shuffle status 'ready' in both arms",
         "measured": float(n_pc_shuf), "threshold": float(SEEDS_REQUIRED),
         "direction": "lower", "control": "ridge z0 + [1, z0, a, z0 x a] @ W",
         "met": n_pc_shuf >= SEEDS_REQUIRED},
        {"name": "gate_canary_reproduces", "description":
         "action_sensitivity_gate.check_canary() reproduces the pinned V3-EXQ-1073 values "
         "(the only check that catches a PARTIALLY broken gate)",
         "measured": 1.0 if canary.get("ok") else 0.0, "threshold": 1.0,
         "direction": "lower", "control": "CANARY_V3_EXQ_1073 synthetic battery",
         "met": bool(canary.get("ok"))},
    ]
    # rows_aligned gates ONLY the C2 contrast, not C1 -- scoped, not whole-run (V3-EXQ-785).
    c2_precondition = {
        "name": "rows_aligned_seeds_for_alpha_contrast", "description":
        "seeds whose two arms' battery ACTION sequences are bit-identical, which is what "
        "licenses the paired joint bootstrap for C2. APPLIES TO C2 ONLY: C1 is a "
        "within-cell statistic and is unaffected",
        "measured": float(n_aligned), "threshold": float(SEEDS_REQUIRED),
        "direction": "lower", "control": "same env seed + same battery rng -> same draws",
        "applies_to": "C2_alpha_arms_differ", "met": n_aligned >= SEEDS_REQUIRED}

    all_pre_met = all(p["met"] for p in preconditions) and n_valid >= SEEDS_REQUIRED
    c1_met = bool(all_pre_met and n_reads >= SEEDS_REQUIRED)
    c2_met = bool(all_pre_met and c2_precondition["met"] and n_differ >= SEEDS_REQUIRED)

    # LABELS ARE WRITTEN TO BE TRUE IN THE STATE THEY FIRE (2026-09-24 red-team F3 fix).
    # In particular, C1 failing does NOT mean the head is action-blind -- reanalysis
    # ...20260924T171654Z already established it reads its action at 0.9 on 3/3 seeds. It
    # means the INVERTED-MAP readout specifically does not clear its action-blind null.
    if not all_pre_met:
        label, outcome = "substrate_not_ready_requeue", "FAIL"
    elif c1_met and c2_met:
        # C2's LABEL SPLIT on the positive control (V3-EXQ-1079 :84-91, :755-758, GROWS
        # direction). Counts are unchanged; only the attribution differs.
        if n_head_gains >= SEEDS_REQUIRED:
            label = "inverted_map_exceeds_blind_null_and_head_gains_beyond_ceiling"
        elif n_ceiling_gains >= SEEDS_REQUIRED:
            label = "inverted_map_exceeds_blind_null_alpha_gain_attributable_to_battery"
        else:
            label = "inverted_map_exceeds_blind_null_alpha_attribution_undetermined"
        outcome = "PASS"
    elif c1_met:
        label, outcome = "inverted_map_exceeds_blind_null_alpha_contrast_undetermined", "FAIL"
    elif n_reads == 0 and n_valid >= SEEDS_REQUIRED:
        label, outcome = "inverted_map_ratio_within_action_blind_null_at_operating_point", "FAIL"
    else:
        label, outcome = "inverted_map_readout_undetermined", "FAIL"

    criteria = [
        {"name": "C1_inverted_map_ratio_exceeds_blind_null", "load_bearing": True,
         "measured": float(n_reads), "threshold": float(SEEDS_REQUIRED),
         "description": "valid seeds whose ARM_ALPHA_0p9 cell has the CROSS-BATTERY "
                        "inverted-map ratio above its ACTION-BLIND NULL on the same battery "
                        "pair -- readiness_verdict(counterfactual_battery=cf, "
                        "ratio_floor=blind_null, skill_floor=%.2f) == 'ready' AND the "
                        "two-sample bootstrap CI on (cross_ratio - blind_null) excluding 0. "
                        "Rejects null clause N1 as re-pointed by the user (option B)."
                        % SKILL_FLOOR,
         "passed": c1_met},
        {"name": "C2_alpha_arms_differ", "load_bearing": True,
         "measured": float(n_differ), "threshold": float(SEEDS_REQUIRED),
         "description": "valid, row-aligned seeds whose paired joint bootstrap CI on "
                        "d_act(0.9) - d_act(0.3) excludes 0. Rejects null clause N2.",
         "passed": c2_met},
        {"name": "C2b_alpha_gain_attributable_to_head", "load_bearing": False,
         "measured": float(n_head_gains), "threshold": float(SEEDS_REQUIRED),
         "description": "ATTRIBUTION, splits C2's label but does NOT gate (V3-EXQ-1079's "
                        "positive-control rule, GROWS direction): valid seeds whose "
                        "difference-of-differences CI -- [d_act_head(0.9)-d_act_head(0.3)] "
                        "minus [d_act_pc(0.9)-d_act_pc(0.3)] -- has lower bound > 0, i.e. "
                        "the head gained beyond the closed-form ridge ceiling. When the "
                        "ceiling gains MORE, the alpha contrast belongs to the battery.",
         "passed": bool(n_head_gains >= SEEDS_REQUIRED)},
        {"name": "C3_cross_battery_bars_agree", "load_bearing": False,
         "measured": float(sum(1 for s in valid if s.get("bars_agree_hi"))),
         "threshold": float(SEEDS_REQUIRED),
         "description": "RECORDED, NEVER GATES (GFLAG-0470): valid seeds where the "
                        "cross-battery raw >1.0 bar and the blind-null bar agree at 0.9. "
                        "Disagreement is the measured size of the aliasing this run's "
                        "criterion was moved off.",
         "passed": bool(sum(1 for s in valid if s.get("bars_agree_hi")) >= SEEDS_REQUIRED)},
    ]

    combination_rule = (
        "USER DECISION: option C first (2026-09-24, chip-20260924-sdppb5-invmap-ratio-bar-"
        "decision), then RE-POINTED to option B (2026-09-24, chip-20260924-sdppb5-invmap-"
        "loadbearing-recoverable, rec-20260924-5812c302) after the Step 4.5 red-team showed "
        "option C's load-bearing criterion was already answered by V3-EXQ-1082. "
        "LOAD-BEARING readout is now the CROSS-BATTERY inverted-map ratio measured against "
        "its ACTION-BLIND NULL on the same battery pair: readiness_verdict(head, orig, "
        "counterfactual_battery=inverted, ratio_floor=blind_null, skill_floor=%.2f) == "
        "'ready' AND a two-sample bootstrap (%d resamples, %.0f%% CI; original and "
        "inverted rows resampled INDEPENDENTLY because they are different row sets) on "
        "(cross_ratio - blind_null) excluding 0. The autopsy's declared null is a "
        "DISJUNCTION, so PASS must reject BOTH clauses: C1 (N1, re-pointed) = that "
        "predicate at ARM_ALPHA_0p9 on >= %d valid seeds; C2 (N2) = paired joint bootstrap "
        "on d_act(0.9) - d_act(0.3) excluding 0 on >= %d valid, row-aligned seeds. outcome "
        "PASS iff C1 AND C2. C2's LABEL is SPLIT on V3-EXQ-1079's positive-control rule "
        "carried into the GROWS direction (1079 :84-91, :755-758): the gain is attributed "
        "to the HEAD only when the difference-of-differences CI -- head contrast minus "
        "ridge-ceiling contrast -- has lower bound > 0 on >= %d valid seeds; when the "
        "ceiling gains MORE the label says the alpha gain is the BATTERY's. Counts are "
        "unchanged by the split. A seed is valid iff both arms' encoders match their "
        "battery agents AND the ridge positive control clears C1's OWN blind-null bar "
        "(pc_cross_ok) in both arms; the shuffle-form control is recorded and does not gate. "
        "rows_aligned is scoped to C2 alone (applies_to) so an unpairable contrast cannot "
        "vacate C1. Any precondition unmet -> substrate_not_ready_requeue. RECORDED AND "
        "NEVER GATING: the raw >1.0 verdict on the cross-battery ratio (so the autopsy's "
        "literal declared null stays evaluable), the SAME-ROWS action_shuffle verdict and "
        "its %d-draw spread (demoted from load-bearing -- its ratio clause is d_act > 0 "
        "restated and reanalysis ...20260924T171654Z settled it from 1082), d_act, "
        "skill_vs_identity, model_r2, persistence and the untrained head."
        % (SKILL_FLOOR, N_BOOTSTRAP, 100 * CI_LEVEL, SEEDS_REQUIRED, SEEDS_REQUIRED,
           SEEDS_REQUIRED, N_SHUFFLE_DRAWS))

    ratios = [r["cross_minus_blind"] for r in rows if r["cross_minus_blind"] is not None]
    widths = [s["alpha_contrast"]["ci_high"] - s["alpha_contrast"]["ci_low"]
              for s in valid if s.get("alpha_contrast", {}).get("ci_low") is not None
              and s["alpha_contrast"].get("ci_high") is not None]
    cb_widths = [r["cross_minus_blind_ci_high"] - r["cross_minus_blind_ci_low"]
                 for r in rows if r["cross_minus_blind_ci_low"] is not None
                 and r["cross_minus_blind_ci_high"] is not None]
    non_degenerate = bool(
        all_pre_met and len(set(round(v, 12) for v in ratios)) > 1
        and len(widths) > 0 and all(w > 0 for w in widths)
        and len(cb_widths) > 0 and all(w > 0 for w in cb_widths))

    def _f(x: Any) -> Optional[float]:
        try:
            v = float(x)
        except (TypeError, ValueError):
            return None
        return v if math.isfinite(v) else None

    flat: Dict[str, Any] = {
        "n_cells": len(rows), "n_valid_seeds": n_valid,
        "n_seeds_inverted_map_exceeds_blind_null": n_reads,
        "n_seeds_arms_differ": n_differ,
        "n_seeds_head_gains_beyond_ceiling": n_head_gains,
        "n_seeds_ceiling_gains_more": n_ceiling_gains,
        "n_seeds_rows_aligned": n_aligned,
        "n_seeds_pc_cross_exceeds_blind": n_pc_cross,
        "n_seeds_pc_shuffle_reads": n_pc_shuf,
        "n_seeds_pc_d_act_live": n_pc_dact,
        "n_seeds_cross_reads_raw_bar": sum(1 for s in valid
                                           if s.get("cross_reads_raw_hi")),
        "n_seeds_cross_reads_vs_blind_null": sum(1 for s in valid
                                                  if s.get("cross_reads_vs_blind_hi")),
        "n_seeds_cross_bars_agree": sum(1 for s in valid if s.get("bars_agree_hi")),
        "all_preconditions_met": 1 if all_pre_met else 0,
        "gate_canary_ok": 1 if canary.get("ok") else 0,
        "c1_inverted_map_exceeds_blind_null": 1 if c1_met else 0,
        "c2_alpha_arms_differ": 1 if c2_met else 0,
        "verdict_pass": 1 if outcome == "PASS" else 0,
        "verdict_c1_only": 1 if label ==
        "inverted_map_exceeds_blind_null_alpha_contrast_undetermined" else 0,
        "verdict_within_blind_null": 1 if label ==
        "inverted_map_ratio_within_action_blind_null_at_operating_point" else 0,
        "verdict_head_gains": 1 if label ==
        "inverted_map_exceeds_blind_null_and_head_gains_beyond_ceiling" else 0,
        "verdict_gain_is_battery": 1 if label ==
        "inverted_map_exceeds_blind_null_alpha_gain_attributable_to_battery" else 0,
        "verdict_undetermined": 1 if label == "inverted_map_readout_undetermined" else 0,
        "verdict_not_ready": 1 if label == "substrate_not_ready_requeue" else 0,
        "shuffle_ratio_floor": float(RATIO_FLOOR),
    }
    for s in per_seed:
        tag = f"seed{s['seed']}"
        flat[f"shuffle_ratio_alpha09_{tag}"] = _f(s.get("shuffle_ratio_hi"))
        flat[f"shuffle_ratio_alpha03_{tag}"] = _f(s.get("shuffle_ratio_lo"))
        ac = s.get("alpha_contrast") or {}
        flat[f"d_act_delta_09_minus_03_{tag}"] = _f(ac.get("delta"))
        flat[f"d_act_delta_ci_low_{tag}"] = _f(ac.get("ci_low"))
        flat[f"d_act_delta_ci_high_{tag}"] = _f(ac.get("ci_high"))
        pcc = s.get("alpha_contrast_pc") or {}
        flat[f"pc_d_act_delta_09_minus_03_{tag}"] = _f(pcc.get("delta"))
        flat[f"pc_d_act_delta_ci_low_{tag}"] = _f(pcc.get("ci_low"))
        flat[f"pc_d_act_delta_ci_high_{tag}"] = _f(pcc.get("ci_high"))
        att = s.get("alpha_contrast_attribution") or {}
        flat[f"head_minus_ceiling_dd_{tag}"] = _f(att.get("dd"))
        flat[f"head_minus_ceiling_dd_ci_low_{tag}"] = _f(att.get("ci_low"))
        flat[f"head_minus_ceiling_dd_ci_high_{tag}"] = _f(att.get("ci_high"))
        flat[f"cross_minus_blind_alpha09_{tag}"] = _f(s.get("cross_minus_blind_hi"))
        flat[f"cross_minus_blind_ci_low_alpha09_{tag}"] = _f(s.get("cross_minus_blind_ci_low_hi"))
        flat[f"cross_minus_blind_ci_high_alpha09_{tag}"] = _f(s.get("cross_minus_blind_ci_high_hi"))
        flat[f"cross_ratio_alpha09_{tag}"] = _f(s.get("cross_ratio_hi"))
        flat[f"cross_blind_null_alpha09_{tag}"] = _f(s.get("cross_blind_null_hi"))
    for r in rows:
        tag = f"{r['arm'].lower()}_seed{r['seed']}"
        for k in ("shuffle_ratio", "shuffle_skill", "cross_ratio", "cross_blind_null",
                  "cross_minus_blind", "cross_minus_blind_ci_low",
                  "cross_minus_blind_ci_high",
                  "cross_ratio_minus_blind_null", "d_act", "d_act_ci_low", "d_act_ci_high",
                  "pc_d_act", "pc_shuffle_ratio", "pc_cross_ratio",
                  "pc_cross_minus_blind", "d_act_untrained_head", "model_r2",
                  "persistence_r2", "persistence_relative_skill", "skill_vs_identity",
                  "conv_rel_drop", "rms_dz_per_dim_battery", "rms_dz_per_dim_battery_cf",
                  "transition_l2_mean", "transition_l2_mean_cf",
                  "head_action_separation_l2_mean", "alpha_world"):
            flat[f"{k}_{tag}"] = _f(r.get(k))
        flat[f"cross_reads_raw_bar_1p0_{tag}"] = int(r["cross_reads_raw_bar_1p0"])
        flat[f"cross_reads_vs_blind_null_{tag}"] = int(r["cross_reads_vs_blind_null"])
        flat[f"pc_shuffle_reads_{tag}"] = int(r["pc_shuffle_reads"])
        flat[f"pc_cross_exceeds_blind_{tag}"] = int(r["pc_cross_exceeds_blind"])
        flat[f"reads_inverted_map_{tag}"] = int(r["reads_inverted_map"])
    flat = {k: v for k, v in flat.items() if v is not None}

    manifest: Dict[str, Any] = {
        "queue_id": QUEUE_ID,
        "validates_substrate": VALIDATES_SUBSTRATE,
        "bears_on": BEARS_ON,
        "bears_on_provenance": (
            "SD-PP-B5 + MECH-573 from chip-20260924-sdppb5-inverted-map-probe; "
            "SD-PP-B10 + SD-008 because the 0.3-vs-0.9 alpha contrast is that pair's "
            "subject and the 1082 autopsy's own bears_on lists both."),
        "autopsy": AUTOPSY,
        "run_id": f"{EXPERIMENT_TYPE}_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}_v3",
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": ARCHITECTURE_EPOCH,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "claim_ids": CLAIM_IDS,
        "evidence_direction": "non_contributory",
        "outcome": outcome,
        "timestamp_utc": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
        "sleep_driver_pattern": "not_applicable",
        "arms": [{"arm": a, "alpha_world": al} for a, al in ARMS],
        "operating_point_arm": OPERATING_POINT_ARM,
        "non_degenerate": non_degenerate,
        "degeneracy_reason": (
            None if non_degenerate else
            "preconditions unmet, or (cross_ratio - blind_null) identical across cells / "
            "zero-width alpha-contrast or blind-null CIs"),
        "gate_canary": canary,
        "reanalysis_settling_the_recoverable_half": (
            "reanalysis_sdppb5_off_head_action_read_alpha09_live_battery_20260924T171654Z"),
        "criterion_provenance": (
            "TWO user decisions. Stop 1 (option C, chip-20260924-sdppb5-invmap-ratio-bar-"
            "decision) load-bore the same-rows shuffle form at 1.0; the Step 4.5 red-team "
            "showed that criterion is d_act > 0 restated and already answered by V3-EXQ-1082 "
            "(this driver's 0.9 arm reproduced 1082's ARM_OFF bit-for-bit). Stop 2 "
            "(chip-20260924-sdppb5-invmap-loadbearing-recoverable, rec-20260924-5812c302) "
            "the USER chose '(2) then (1)': emit the reanalysis named above, then RE-POINT "
            "the load-bearing criterion at the CROSS-BATTERY ratio vs its action-blind null "
            "(option B), add V3-EXQ-1079's positive-control contrast and label split, and "
            "fix the false label. GOV-REUSE-1 was re-run against the NEW statistic: 1 of "
            "1076 manifests carries it and its rows are post-death, so the live-battery form "
            "is not recoverable. Superseded first-decision text follows. "
            "Bar chosen by the USER (option C, 2026-09-24) on decision chip "
            "chip-20260924-sdppb5-invmap-ratio-bar-decision, raised because the 1082 "
            "autopsy's ~1.0 bar is the null of the SAME-ROWS shuffle form, not of the "
            "cross-battery form its sketch specifies. Resolves GFLAG-0470's contested "
            "disposition as C; /governance applies the flag. Analysis: REE_assembly "
            "evidence/planning/"
            "sdppb5_inverted_map_ratio_criterion_aliasing_staged_20260924.md (e9e37e46c2)."),
        "ethics_preflight": {
            "involves_negative_valence": False, "involves_suffering_like_state": False,
            "involves_self_model": False, "involves_inescapability_or_helplessness": False,
            "involves_offline_replay_over_harm": False,
            "involves_social_mind_or_language": False,
            "involves_human_data_or_clinical_context": False, "decision": "allow"},
        "readout": flat,
        "arm_results": rows,
        "per_seed_analysis": per_seed,
        "interpretation": {
            "label": label,
            "combination_rule": combination_rule,
            "preconditions": preconditions + [c2_precondition],
            "criteria": criteria,
            "criteria_non_degenerate": {c["name"]: non_degenerate for c in criteria},
        },
    }
    return manifest, t0


if __name__ == "__main__":
    _ap = argparse.ArgumentParser()
    _ap.add_argument("--dry-run", action="store_true")
    _args = _ap.parse_args()

    _manifest, _t0 = run_experiment(dry_run=_args.dry_run)
    _out_path = write_flat_manifest(
        _manifest, dry_run=_args.dry_run,
        config={"arms": [{"arm": a, "alpha_world": al} for a, al in ARMS],
                "operating_point_arm": OPERATING_POINT_ARM, "seeds": SEEDS,
                "p0_steps": P0_STEPS, "steps_per_episode": STEPS_PER_EPISODE,
                "episodes_per_run": EPISODES_PER_RUN, "batch_k": BATCH_K, "lr": LR,
                "min_buf_before_train": MIN_BUF_BEFORE_TRAIN, "buf_cap": BUF_CAP,
                "max_grad_norm": MAX_GRAD_NORM,
                "world_dim": WORLD_DIM, "self_dim": SELF_DIM,
                "live_battery_n": LIVE_BATTERY_N, "post_reset_skip": POST_RESET_SKIP,
                "battery_rng_xor": BATTERY_RNG_XOR,
                "shuffle_gen_seed": SHUFFLE_GEN_SEED,
                "n_shuffle_draws": N_SHUFFLE_DRAWS,
                "ratio_floor": RATIO_FLOOR, "skill_floor": SKILL_FLOOR,
                "min_battery_rows": MIN_BATTERY_ROWS,
                "min_distinct_actions": MIN_DISTINCT_ACTIONS,
                "n_bootstrap": N_BOOTSTRAP, "ci_level": CI_LEVEL,
                "ridge_rel_lambda": RIDGE_REL_LAMBDA,
                "seeds_required": SEEDS_REQUIRED,
                "env": {"size": GRID_SIZE, "num_hazards": N_HAZARDS,
                        "num_resources": N_RESOURCES, "use_proxy_fields": True}},
        seeds=SEEDS, script_path=Path(__file__), started_at=_t0,
        z_goal_stream_stats=_ZG.stats())

    print(f"[{EXPERIMENT_TYPE}] outcome={_manifest['outcome']} "
          f"label={_manifest['interpretation']['label']}", flush=True)
    _o = str(_manifest["outcome"]).upper()
    emit_outcome(outcome=_o if _o in ("PASS", "FAIL") else "FAIL",
                 manifest_path=_out_path, dry_run=_args.dry_run)
