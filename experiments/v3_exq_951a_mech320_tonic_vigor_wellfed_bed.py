"""
V3-EXQ-951 -- MECH-320 (tonic_vigor_coupling_score_bias) selection-authority
retest on the NOW-READY scaffolded_sd054_onboarding substrate.

BACKGROUND (why this exists, and why NOT a simple "V3-EXQ-624d"):
V3-EXQ-624/624a/624b/624c (2026-06-02 through 2026-06-07) tested MECH-320's
w_passive score-bias with the modulatory-bias-selection-authority substrate
already ON (landed 2026-06-03; float32-cancellation amend V3-EXQ-643a landed
2026-06-06). 624c -- the last of that lineage -- ran with
use_modulatory_selection_authority=True in ALL arms and measured a REAL,
non-byte-identical action_density lift on the seeds where the positive
control fired (ARM_1 0.9785 vs ARM_0 0.9076, seed-level lift up to +0.259 in
earlier letters). So the premise "the authority substrate has never been
exercised for MECH-320" is FALSE as of 624c; a bare re-run with
use_modulatory_selection_authority=True would repeat work already done.

What 624c actually found (failure_autopsy_V3-EXQ-603g-624c-651a_2026-06-07,
CONFIRMED): positive control (a competitive no-op candidate) validated on
only 2/5 seeds (majority >=3 not met) -- an ad-hoc hand-tuned env regime
(size=8, 1 hazard, sparse resources, P0=100 no-grad warmup) could not
reliably produce a policy with genuine no-op opportunity. Per EXPLICIT USER
ADJUDICATION recorded in that autopsy: "block on the foraging-competence
substrate rather than re-queue a regime tweak. The positive-control headroom
... depends on genuine foraging competence (GAP-2)." recommended_substrate_
queue_entry names target_sd_id="scaffolded_sd054_onboarding", action="none"
(no further regime-tuning owed -- the residual IS that substrate).

That substrate (evidence/planning/substrate_queue.json,
sd_id="scaffolded_sd054_onboarding") has read ready:true since 2026-06-11
(V3-EXQ-603n PASS, all four G0-G3 legs cleared >=2/3 seeds). No MECH-320
retest has used it. The re-derive brake (skill Step 2.5b) counts 4 prior
substrate_ceiling/non_contributory autopsy hits for MECH-320 (>= threshold
2), so a bare same-design re-queue is braked -- but the brake is RELEASED
here because the specific substrate the most recent counted autopsy (624c)
named as the residual gate is now built. This queue entry records that
release explicitly (see `note` in the queue entry).

WHY A NEW NUMBER, NOT "624d": 624/624a/b/c tested MECH-320 JOINTLY with
ARC-068 (the Niv-vs-Salamone effort-cost-vs-opportunity-cost dissociation,
via a parametric movement-cost manipulation on top of the vigor toggle).
This experiment does NOT reproduce that dissociation -- composing the
Salamone movement-cost env variant with the scaffolded_sd054_onboarding
curriculum's own env builder (_build_env, which does not expose
damage_increment/failure_prob_scale overrides per phase) is a separate,
non-trivial integration surface, and is left for a genuine "624d" successor
if this run clears its own gate. Instead this experiment tests TWO questions
that the 624-series never resolved conclusively because it could not get
genuine substrate competence:
  (a) PRIMARY (the brief's own stated primary DV): does P2 action_density
      rise under MECH-320 vigor-ON relative to vigor-OFF, on a policy that
      has actually learned to forage/survive (so a competitive no-op
      candidate genuinely exists)?
  (b) SECONDARY, EXPLORATORY (R3 falsifiable alternative, never tested by
      ANY prior V3-EXQ under MECH-320 -- additive is the sole form the 624-
      series ever exercised): does the ON-arm lift look additive (roughly
      constant vs the seed's own baseline action-preference) or
      multiplicative (scaling with it)? Reported as INFORMATIVE ONLY -- 3
      seeds cannot power a real regression, and this is explicitly NOT a
      pre-registered gating criterion.
claim_ids = ["MECH-320"] only (ARC-068 is NOT re-tested here; the Niv-vs-
Salamone dissociation on this substrate is explicitly out of scope -- see
above).

DESIGN: train ONE agent per seed through the FULL landed curriculum (Stage0
nursery -> Stage0b consolidation -> P0 -> Stage-H hazard-avoidance -> P1
foraging), mirroring V3-EXQ-812's already-validated integration of MECH-295
onto this exact substrate (same curriculum config, same base REEConfig
build, same abort-checkpoint pattern). MECH-320's vigor module is
CONSTRUCTED and LIVE (use_tonic_vigor=True, w_action=w_passive=0.1,
v_t_floor=0.05, form="additive") for the ENTIRE training trajectory --
severing it during training would confound "does vigor have selection
authority in a competently-trained policy" with "what did a differently-
trained policy learn to compensate" (the same reasoning 812 used for the
MECH-295 cue). modulatory-bias-selection-authority is ON throughout
(use_modulatory_selection_authority=True, gain=0.5 -- the fixed, already-
landed substrate 624c exercised, unchanged here).

At P2 (frozen policy: agent.eval(), no gradient steps), run THREE arms on
the SAME trained weights, toggling ONLY the TonicVigor module's live config
object (a plain mutable dataclass -- agent.tonic_vigor.config), with
reset_all_rng(seed) immediately before EACH arm so all three see an
identical env realisation sequence (matched comparison; same idiom 812 uses
for its 2-arm cue toggle):
  ARM_0_baseline:      w_action=0.0, w_passive=0.0  (bias forced to exactly
                        zero regardless of v_t; v_t_floor/form irrelevant).
  ARM_1_vigor_additive: w_action=0.1, w_passive=0.1, form="additive"
                        (matches the trained-with config exactly).
  ARM_2_vigor_multiplicative: w_action=0.1, w_passive=0.1,
                        form="multiplicative" (TonicVigor.compute_score_bias
                        reads self.config.form fresh on every call -- this is
                        a pure runtime toggle on the same live module, not a
                        different code path per arm).
Per-tick instrumentation is a single monkeypatch on agent.select_action
(mirroring 812's _CueAuthorityProbe pattern exactly, but simpler): records
action_idx = argmax(action) and increments a non-noop counter. NOOP_CLASS=4
(CausalGridWorldV2.ACTIONS index 4 == (0, 0), the built-in stay/no-op move;
confirmed the scaffolded env builder does not override action_dim, so this
is the SAME convention V3-EXQ-624c used, NOT the TonicVigorConfig library
default of 0, which would misclassify action-4 no-ops as an ordinary move
in this env's index convention).

SUBSTRATE-READINESS GATE (why a seed can be excluded from scoring without
that being a claim verdict -- the 624c lesson, generalised): a seed is VALID
only if BOTH (i) P1 survival_gate_passed (the same G1 precondition 603n/812
already validate on this exact curriculum -- reused verbatim, per the
skill's precedent for reusing an already-landed precondition shape) AND
(ii) ARM_0's own P2 action_density is NOT already saturated
(< BASELINE_SATURATION_CEILING = 0.95) -- a saturated baseline has no
no-op-opportunity headroom for a lift to appear in, structurally, regardless
of whether MECH-320 has real authority (this is precisely the ceiling
condition that starved 3/5 seeds in 624c; recording it explicitly here
rather than re-discovering it after the fact).

MANDATORY NON-VACUITY GUARD (mirrors the V3-EXQ-643 lesson): ARM_0's tonic-
vigor bias magnitude (last_bias_max_abs from TonicVigor.get_state(), the
SAME statistic the mechanism acts through) must read exactly 0.0 -- sanity-
checked in the smoke test and again in run_experiment -- confirming the
w_action=w_passive=0.0 toggle actually zeroed the channel rather than
silently no-op'ing.

PRE-REGISTERED THRESHOLDS:
  C1 (LOAD-BEARING, the brief's Primary DV): on VALID seeds, ARM_1
     action_density - ARM_0 action_density >= C1_LIFT_MIN (0.03), on a
     majority (>= VALID_SEED_MAJORITY_FRAC = 2/3) of valid seeds, AND valid
     seeds must themselves be >= 2/3 of all seeds run (else the test could
     not evaluate the claim on enough seeds -- non_contributory, not a
     verdict).
  C2 (INFORMATIVE ONLY, R3 form question -- NOT gating; explicitly reported
     as underpowered): per-seed (ARM_2 - ARM_0) vs (ARM_1 - ARM_0), and each
     arm's lift vs ARM_0's own baseline density (the natural pre-existing-
     action-preference proxy this run's small seed count can supply, in
     place of the parametric preference sweep the original claim
     registration envisioned).

experiment_purpose = "evidence". claim_ids = ["MECH-320"].
Predecessor (not supersedes -- different scope, see above): V3-EXQ-624c
(non_contributory/substrate_conditional, majority-of-5 positive control not
met; failure_autopsy_V3-EXQ-603g-624c-651a_2026-06-07 names
scaffolded_sd054_onboarding as the residual gate this experiment clears).

SLEEP DRIVER: N/A (no sleep loop; scaffolded_sd054_onboarding is a waking
goal-pipeline onboarding scheduler, same as 812/603n).

Run with:
  /opt/local/bin/python3 experiments/v3_exq_951_mech320_tonic_vigor_authority_sd054.py
or:
  /opt/local/bin/python3 experiments/v3_exq_951_mech320_tonic_vigor_authority_sd054.py --dry-run
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "experiments") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "experiments"))

from experiment_protocol import emit_outcome  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402
from scaffolded_sd054_onboarding import (  # noqa: E402
    ScaffoldedSD054OnboardingConfig,
    ScaffoldedSD054OnboardingScheduler,
)
from experiments.pack_writer import write_flat_manifest  # noqa: E402
from experiments._lib.z_goal_stream import ZGoalStreamAccumulator  # noqa: E402
from experiments._lib.arm_fingerprint import arm_cell, reset_all_rng  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_951a_mech320_tonic_vigor_wellfed_bed"
QUEUE_ID = "V3-EXQ-951a"
CLAIM_IDS: List[str] = ["MECH-320"]
EXPERIMENT_PURPOSE = "evidence"

# Seed 44 REMOVED. It is a documented recurring per-seed instability on
# reef-config envs (early episode death ~step 40), confirmed across at least two
# independent autopsies (EXQ-539-540, V3-EXQ-538a); the standing instruction is
# to substitute 45 rather than debug the same truncation a third time. 951 shipped
# [42, 43, 44] and this run's curriculum keeps scaffold_p1_reef_spawn_hold_fraction
# > 0, so the reef config is live and the caution applies.
SEEDS = [42, 43, 45]
MIN_FRACTION = 2.0 / 3.0  # matches 603n / 812's own validated seed-majority convention

# ---- Curriculum config: mirror V3-EXQ-812 / 603n exactly (the landed readiness config) ----
WORLD_DIM = 32
HARM_A_DIM = 16
HARM_OBS_A_DIM = 7
HARM_HISTORY_LEN = 10
DRIVE_WEIGHT = 2.0

STAGE0_BUDGET = 20
STAGE0B_BUDGET = 10
P0_BUDGET = 100
HAZARD_STAGE_BUDGET = 40
P1_BUDGET = 50
P2_BUDGET = 15
TRAIN_STEPS = 200
P1_HOLD_FRACTION = 0.3
P0_NUM_HAZARDS = 1
# ---- 951a: the WELL-FED / SAFE / FAMILIAR P2 MEASUREMENT BED -------------
# THE ROOT CAUSE 951a EXISTS TO FIX. 951 left the scaffold's P2 env knobs at
# their defaults -- scaffold_p2_num_hazards=4, scaffold_p2_proximity_harm_scale
# =0.1 -- and set the hazard-food-attraction guard to 0.3. That is a THREAT bed,
# and on a threat bed the agent moves on every single step, which is exactly what
# action_density == 1.0000 in 9/9 cells means. A tonic-vigor bias TOWARD action
# is unmeasurable in a regime where the agent already acts unconditionally: the
# DV sits on its ceiling and no lift can be observed in either direction. The
# readiness gate `baseline_no_op_opportunity` correctly refused to score, so the
# instrument caught it.
#
# *** PILOTED 2026-09-01, AND THIS DIAGNOSIS WAS REFUTED. READ BEFORE QUEUEING. ***
# The bed change below was built and piloted. On the CALMED bed -- 0 hazards, 0
# proximity harm, HFA guard 0.0 -- action_density is STILL exactly 1.0000 in all
# three arms (ARM_0 1.0000, ARM_1 1.0000, ARM_2 1.0000), which is the same
# signature as 951's full run at exactly 1.0000 in 9/9 cells. Removing the threat
# entirely did NOT create no-op opportunity, so THE BED WAS NOT THE BINDING
# CONSTRAINT and this driver must NOT be queued as-is.
# CONFIRMED NOT A DRY-RUN LENGTH ARTIFACT. The pilot was re-run at ~5x the
# budgets (stage0/stage0b/p0/hazard/p1/p2 = 4/3/12/10/12/10 at 90 steps, vs
# 2/2/5/5/5/2 at 30) and every number came back BIT-IDENTICAL: action_density
# 1.0000 in all three arms, v_t_mean 0.0500, bias_max_abs_mean 0.0000 / 0.0050 /
# 0.0000, same label. Do not spend another pilot on "maybe it just needed longer". Three things the pilot
# established:
#   (A) C1_LIFT_MIN = 0.03 is UNATTAINABLE BY CONSTRUCTION. It asks for
#       ARM_n minus ARM_0 action_density >= 0.03 while ARM_0 sits at 1.0000,
#       i.e. action_density 1.03. The criterion can only ever fire negative.
#   (B) v_t IS PINNED AT ITS FLOOR: v_t_mean = 0.0500 = exactly V_T_FLOOR in
#       every arm, so the tonic-vigor EWMA never rises above the forced probe
#       floor. The additive arm's bias is then w_action * v_t = 0.1 * 0.05 =
#       0.005, matching the observed bias_max_abs_mean 0.0050 exactly -- a
#       5%-of-scale nudge against bias_scale 0.1 that never flips an argmax to
#       the no-op class.
#   (C) ARM_2 (multiplicative) IS INERT BY CONSTRUCTION here: bias_max_abs_mean
#       0.0000. ree_core/agent.py (~8226) states the multiplicative form anchors
#       on dacc_score_bias and yields zero bias when that is None.
# RULED OUT: the no-op IS reachable -- ACTIONS[4] == (0, 0), action_dim 5, and
# this driver passes action_dim=env.action_dim -- so 1.0000 is genuine behaviour,
# not an unreachable-class artifact. (V3-EXQ-547 used action_dim=4, where class 4
# WOULD be unreachable; do not copy its from_dims call.)
# TWO OF THE ORIGINATING CHIP'S PREMISES WERE ALSO WRONG: ARC-068 WAS already
# enabled (config.py ~4356 documents tonic_vigor_w_passive as the ARC-068
# opportunity-cost no-op weight, set to 0.1 in ARM_1/ARM_2), and ARC-067 CANNOT
# be enabled -- it has NO implementation anywhere in ree_core.
# The open fork -- change the DV to the no-op candidate's SCORE MARGIN, fix the
# v_t-floor manipulation, or both as a small portfolio -- is carried by
# chip-20260901-mech320-dv-headroom-and-vt-floor and is a decision for the user,
# not something to guess. The bed change below is still CORRECT and worth keeping;
# it is simply not sufficient.
#
# The predecessor diagnostic V3-EXQ-547 had ALREADY pre-specified the right bed
# ("a well-fed-safe-familiar substrate"); 951 simply did not run on it. 547's own
# concrete params (size=5, num_hazards=1, num_resources=1) were chosen for
# one-tick unit probes, not for a behavioural measurement, so 951a transplants the
# REGIME rather than the literal numbers, and says so:
#   SAFE       -> P2 hazards 4 -> 0, proximity-harm scale 0.1 -> 0.0, and the
#                 hazard-food-attraction guard 0.3 -> 0.0, so nothing in the
#                 measurement window couples food to threat.
#   WELL-FED   -> P2 resources left plentiful (5) against 0 hazards, so drive
#                 pressure cannot itself force continuous movement.
#   FAMILIAR   -> unchanged: the agent arrives at P2 having trained through the
#                 full scaffolded curriculum, including the hazard stage, on the
#                 same world. Competence is preserved; only the MEASUREMENT bed
#                 is calmed. This is the minimal change that satisfies 547's
#                 specification without discarding the SD-054 onboarding the
#                 substrate-readiness gate depends on.
P2_HFA_GUARD = 0.0
P2_NUM_HAZARDS = 0
P2_NUM_RESOURCES = 5
P2_PROXIMITY_HARM = 0.0
P1_REEF_SPAWN_HOLD_FRACTION = 0.4

HAZARD_STAGE_NUM_HAZARDS = 4
HAZARD_STAGE_NUM_RESOURCES = 2
HAZARD_STAGE_HFA = 0.0
HAZARD_STAGE_PROXIMITY_HARM = 0.1
HAZARD_STAGE_SPAWN_IN_REEF = True
HAZARD_STAGE_SURVIVAL_GATE_STEPS = 75
HAZARD_STAGE_STABILITY_WINDOW = 10

SEED_GAIN = 1.5
SEED_BENEFIT_THRESHOLD = 0.02
SEED_DRIVE_FLOOR = 0.9
N_RESOURCE_TYPES = 3
CUE_RECALL_GAIN = 0.2

AVOIDANCE_SCAFFOLD_FLOOR_START = 0.8
AVOIDANCE_SCAFFOLD_FLOOR_END = 0.0
AVOIDANCE_THREAT_REF = 0.35
PAG_THETA_FREEZE = 0.8
PAG_DURATION_INPUT_THRESHOLD = 0.2

HARM_PATHWAY_LR = 1e-3

# ---- 951-specific: MECH-320 vigor + selection-authority config ----
ACTION_DIM = 5  # CausalGridWorldV2.ACTIONS: 0=up,1=down,2=left,3=right,4=noop(stay)
NOOP_CLASS = 4  # see module docstring: NOT the TonicVigorConfig library default (0)
V_T_FLOOR = 0.05  # forced-vigor probe (V3-EXQ-549 fix), matches 624-series
VIGOR_W_ACTION_TRAINED = 0.1
VIGOR_W_PASSIVE_TRAINED = 0.1
VIGOR_HALF_LIFE = 100.0
USE_MODULATORY_AUTHORITY = True
MODULATORY_AUTHORITY_GAIN = 0.5

# ---- 951-specific gates ----
C1_LIFT_MIN = 0.03  # matches 624-series C1
BASELINE_SATURATION_CEILING = 0.95  # ARM_0 P2 action_density must stay below this
VALID_SEED_MAJORITY_FRAC = MIN_FRACTION

ARM_LABELS = [
    "ARM_0_baseline",
    "ARM_1_vigor_additive",
    "ARM_2_vigor_multiplicative",
]

CONFIG_SLICE = {
    "scaffold_cfg": "see _make_scaffold_cfg (curriculum budgets + landed levers, mirrors 603n/812)",
    "world_dim": WORLD_DIM, "drive_weight": DRIVE_WEIGHT,
    "seed_gain": SEED_GAIN, "seed_benefit_threshold": SEED_BENEFIT_THRESHOLD,
    "seed_drive_floor": SEED_DRIVE_FLOOR, "cue_recall_gain": CUE_RECALL_GAIN,
    "harm_pathway_lr": HARM_PATHWAY_LR,
    "v_t_floor": V_T_FLOOR,
    "vigor_w_action_trained": VIGOR_W_ACTION_TRAINED,
    "vigor_w_passive_trained": VIGOR_W_PASSIVE_TRAINED,
    "use_modulatory_selection_authority": USE_MODULATORY_AUTHORITY,
    "modulatory_authority_gain": MODULATORY_AUTHORITY_GAIN,
}


def _make_scaffold_cfg(dry_run: bool) -> ScaffoldedSD054OnboardingConfig:
    if dry_run:
        stage0, stage0b, p0, hazard, p1, p2, steps = 2, 2, 5, 5, 5, 2, 30
    else:
        stage0, stage0b, p0, hazard, p1, p2, steps = (
            STAGE0_BUDGET, STAGE0B_BUDGET, P0_BUDGET, HAZARD_STAGE_BUDGET,
            P1_BUDGET, P2_BUDGET, TRAIN_STEPS,
        )
    cfg = ScaffoldedSD054OnboardingConfig(
        use_scaffolded_sd054_onboarding_scheduler=True,
        scaffold_stage0_enabled=True,
        scaffold_stage0_episode_budget=stage0,
        scaffold_p0_episode_budget=p0,
        scaffold_p1_episode_budget=p1,
        scaffold_p2_episode_budget=p2,
        scaffold_steps_per_episode=steps,
        scaffold_p0_num_hazards=P0_NUM_HAZARDS,
        scaffold_p1_anneal_hold_fraction=P1_HOLD_FRACTION,
        scaffold_p2_hazard_food_attraction_guard=P2_HFA_GUARD,
        # 951a: the well-fed-safe-familiar measurement bed (see the block above).
        # 951 left all three of these at their defaults (4 hazards, 5 resources,
        # 0.1 proximity harm), which is what saturated action_density.
        scaffold_p2_num_hazards=P2_NUM_HAZARDS,
        scaffold_p2_num_resources=P2_NUM_RESOURCES,
        scaffold_p2_proximity_harm_scale=P2_PROXIMITY_HARM,
        scaffold_developmental_window_enabled=True,
        scaffold_stage0b_enabled=True,
        scaffold_stage0b_episode_budget=stage0b,
        scaffold_stage0b_retention_gate=0.75,
        scaffold_contact_gated_goal_updates=True,
        scaffold_z_goal_seeding_gain=SEED_GAIN,
        scaffold_benefit_threshold=SEED_BENEFIT_THRESHOLD,
        scaffold_drive_floor=SEED_DRIVE_FLOOR,
        scaffold_auto_reconcile_gating_to_seeding=True,
        scaffold_p1_reef_spawn_hold_fraction=P1_REEF_SPAWN_HOLD_FRACTION,
        scaffold_cue_recall_bridge_enabled=True,
        scaffold_cue_n_resource_types=N_RESOURCE_TYPES,
        scaffold_stage0_bind_incentive_token=True,
        scaffold_hazard_stage_enabled=True,
        scaffold_hazard_stage_episode_budget=hazard,
        scaffold_hazard_stage_num_hazards=HAZARD_STAGE_NUM_HAZARDS,
        scaffold_hazard_stage_num_resources=HAZARD_STAGE_NUM_RESOURCES,
        scaffold_hazard_stage_hazard_food_attraction=HAZARD_STAGE_HFA,
        scaffold_hazard_stage_proximity_harm_scale=HAZARD_STAGE_PROXIMITY_HARM,
        scaffold_hazard_stage_spawn_in_reef_half=HAZARD_STAGE_SPAWN_IN_REEF,
        scaffold_hazard_stage_survival_gate_steps=HAZARD_STAGE_SURVIVAL_GATE_STEPS,
        scaffold_hazard_stage_stability_window=HAZARD_STAGE_STABILITY_WINDOW,
        scaffold_avoidance_driver_enabled=True,
        scaffold_avoidance_scaffold_floor_start=AVOIDANCE_SCAFFOLD_FLOOR_START,
        scaffold_avoidance_scaffold_floor_end=AVOIDANCE_SCAFFOLD_FLOOR_END,
        scaffold_feed_harm_stream=True,
        scaffold_train_harm_pathway=True,
        scaffold_harm_pathway_lr=HARM_PATHWAY_LR,
        scaffold_harm_pathway_in_p0=True,
    )
    if steps < 75:
        cfg.scaffold_p1_survival_gate_steps = max(1, steps // 4)
        cfg.scaffold_hazard_stage_survival_gate_steps = max(1, steps // 4)
    return cfg


def _make_config(env) -> REEConfig:
    """Base REEConfig mirroring V3-EXQ-812's landed-substrate config (proven to
    carry an agent through the full scaffolded_sd054_onboarding curriculum),
    with MECH-320 tonic-vigor + modulatory-selection-authority layered on top.
    Other flags (harm stream, mech295 bridge, avoidance, pag freeze gate) are
    kept ON exactly as 812/603n use them -- they are part of the FIXED,
    already-validated background substrate this experiment reuses, not
    manipulated variables; MECH-320's vigor bias composes orthogonally with
    all of them (target-free, same-call-site design per tonic_vigor.py).
    """
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=32,
        world_dim=WORLD_DIM,
        alpha_world=0.9,
        use_harm_stream=True,
        use_affective_harm_stream=True,
        z_harm_a_dim=HARM_A_DIM,
        harm_obs_a_dim=HARM_OBS_A_DIM,
        harm_history_len=HARM_HISTORY_LEN,
        use_e2_harm_s_forward=True,
        use_support_preserving_cem=True,
        support_preserving_stratified_elites=True,
        support_preserving_ao_std_floor=0.2,
        support_preserving_min_first_action_classes=2,
        z_goal_enabled=True,
        drive_weight=DRIVE_WEIGHT,
        use_mech295_liking_bridge=True,
        mech295_liking_to_approach_cue_gain=0.5,
        use_mech307_conjunction=True,
        use_incentive_token_bank=True,
        use_cue_recall=True,
        cue_recall_gain=CUE_RECALL_GAIN,
        e2_action_contrastive_enabled=True,
        use_pag_freeze_gate=True,
        pag_theta_freeze=PAG_THETA_FREEZE,
        pag_duration_input_threshold=PAG_DURATION_INPUT_THRESHOLD,
        use_instrumental_avoidance=True,
        avoidance_threat_ref=AVOIDANCE_THREAT_REF,
        # MECH-320 tonic vigor: LIVE for the entire training trajectory (see
        # module docstring -- severing during training would confound the
        # authority test with a differently-trained policy). Trained-with
        # values are the arm-toggle's "ON" state; ARM_0 zeroes w_action/
        # w_passive post-hoc at P2 via the live config object.
        use_tonic_vigor=True,
        tonic_vigor_v_t_floor=V_T_FLOOR,
        tonic_vigor_noop_class=NOOP_CLASS,
        # Modulatory-bias-selection-authority: fixed, already-landed substrate
        # (2026-06-03; float32 amend V3-EXQ-643a 2026-06-06), ON throughout,
        # unchanged from the 624-series. Gives MECH-320's bias bounded
        # authority over the E3.select argmin.
        use_modulatory_selection_authority=USE_MODULATORY_AUTHORITY,
        modulatory_authority_gain=MODULATORY_AUTHORITY_GAIN,
    )
    cfg.latent.use_resource_encoder = True
    return cfg


class _ActionDensityProbe:
    """Wraps agent.select_action to count P2 ticks where the committed action
    is non-noop (action_density), plus the tonic-vigor per-tick gate state.

    Mirrors V3-EXQ-812's _CueAuthorityProbe pattern (install once, read
    agent._last e3 state immediately after the wrapped call returns) but is
    simpler: no candidate-proximity surface to capture, just the committed
    action index scaffolded_sd054_onboarding._eval_episode already computes
    via action.argmax(dim=-1) after calling agent.select_action(candidates,
    ticks) every P2 step.
    """

    def __init__(self, agent: REEAgent) -> None:
        self.agent = agent
        self.n_ticks = 0
        self.n_nonnoop_ticks = 0
        self.gate_energy_sum = 0.0
        self.gate_drive_sum = 0.0
        self.gate_pe_sum = 0.0
        self.v_t_sum = 0.0
        self.bias_max_abs_sum = 0.0
        self._installed = False

    def install(self) -> None:
        if self._installed:
            return
        orig_select_action = self.agent.select_action

        def wrapped_select_action(*args, **kwargs):
            action = orig_select_action(*args, **kwargs)
            try:
                action_idx = int(action.argmax(dim=-1).item())
                self.n_ticks += 1
                if action_idx != NOOP_CLASS:
                    self.n_nonnoop_ticks += 1
                tv = getattr(self.agent, "tonic_vigor", None)
                if tv is not None:
                    st = tv.get_state()
                    self.gate_energy_sum += float(st["last_gate_energy"])
                    self.gate_drive_sum += float(st["last_gate_drive"])
                    self.gate_pe_sum += float(st["last_gate_pe"])
                    self.v_t_sum += float(st["last_v_t"])
                    self.bias_max_abs_sum += float(st["last_bias_max_abs"])
            except Exception:
                pass
            return action

        self.agent.select_action = wrapped_select_action
        self._installed = True

    def summary(self) -> Dict[str, Any]:
        n = self.n_ticks
        return {
            "n_ticks": n,
            "action_density": (self.n_nonnoop_ticks / n) if n > 0 else 0.0,
            "gate_product_mean": (
                (self.gate_energy_sum / n) * (self.gate_drive_sum / n) * (self.gate_pe_sum / n)
                if n > 0 else 0.0
            ),
            "v_t_mean": (self.v_t_sum / n) if n > 0 else 0.0,
            "bias_max_abs_mean": (self.bias_max_abs_sum / n) if n > 0 else 0.0,
        }


def _run_p2_arm(
    scheduler: ScaffoldedSD054OnboardingScheduler,
    agent: REEAgent,
    device: torch.device,
    seed: int,
    arm_label: str,
    w_action: float,
    w_passive: float,
    form: str,
    dry_run: bool,
) -> Dict[str, Any]:
    """Run one frozen-policy P2 pass at a fixed vigor-weight/form setting,
    instrumented for action_density. reset_all_rng(seed) is called first so
    every arm of this seed sees an identical env realisation sequence (matches
    V3-EXQ-812's arm-toggle idiom).
    """
    reset_all_rng(seed)
    tv = getattr(agent, "tonic_vigor", None)
    if tv is not None:
        tv.config.w_action = float(w_action)
        tv.config.w_passive = float(w_passive)
        tv.config.form = str(form)

    probe = _ActionDensityProbe(agent)
    probe.install()
    agent.eval()
    scheduler.run_p2(agent, device)
    summary = probe.summary()

    row: Dict[str, Any] = {
        "arm": arm_label,
        "seed": int(seed),
        "w_action": float(w_action),
        "w_passive": float(w_passive),
        "form": str(form),
        **summary,
    }
    print(
        f"  [p2_arm] seed={seed} arm={arm_label} w_action={w_action:.2f}"
        f" w_passive={w_passive:.2f} form={form}"
        f" action_density={summary['action_density']:.4f}"
        f" v_t_mean={summary['v_t_mean']:.4f}"
        f" bias_max_abs_mean={summary['bias_max_abs_mean']:.4f}",
        flush=True,
    )

    with arm_cell(
        seed,
        config_slice=CONFIG_SLICE,
        script_path=Path(__file__),
        config_slice_declared=True,
        extra_ineligible_reasons=[
            "shared_trained_agent_eval_time_toggle_not_independently_trained",
        ],
        do_reset=False,  # RNG already reset above (same seed, matched-comparison purpose)
    ) as cell:
        cell.stamp(row)
    return row


_ZG = ZGoalStreamAccumulator()


def _run_seed(seed: int, dry_run: bool, total_eps: int) -> Dict[str, Any]:
    torch.manual_seed(seed)
    scaffold_cfg = _make_scaffold_cfg(dry_run)
    device = torch.device("cpu")

    from scaffolded_sd054_onboarding import _build_env
    probe_env = _build_env(scaffold_cfg, "p2")
    probe_env.reset()
    agent = REEAgent(_make_config(probe_env)).to(device)
    scheduler = ScaffoldedSD054OnboardingScheduler(scaffold_cfg)

    print(f"Seed {seed} Condition MECH320_TONIC_VIGOR_AUTHORITY_SD054", flush=True)

    def _fail(stage: str, reason: str) -> Dict[str, Any]:
        print(f"verdict: FAIL seed={seed} aborted_at={stage} reason={reason}", flush=True)
        _ZG.observe(agent)
        return {
            "seed": seed, "aborted_at": stage, "abort_reason": reason,
            "arms": [], "g1_p1_survival": False, "seed_pass": False,
        }

    s0 = scheduler.run_stage0_nursery(agent, device)
    done = s0.n_episodes
    print(f"  [train] stage0_nursery seed={seed} ep {done}/{total_eps}", flush=True)
    if s0.aborted:
        return _fail("stage0", s0.abort_reason)

    s0b = scheduler.run_stage0b_consolidation(agent, device, stage0_baseline_norm=s0.z_goal_norm_peak)
    done += s0b.n_episodes
    print(f"  [train] stage0b_consolidate seed={seed} ep {done}/{total_eps}", flush=True)
    if s0b.aborted:
        return _fail("stage0b", s0b.abort_reason)

    p0 = scheduler.run_p0(agent, device)
    done += p0.n_episodes
    print(f"  [train] p0_guided seed={seed} ep {done}/{total_eps}"
          f" mean_len={p0.mean_episode_length:.1f}", flush=True)
    if p0.aborted:
        return _fail("p0", p0.abort_reason)

    hz = scheduler.run_hazard_avoidance(agent, device)
    done += hz.n_episodes
    print(f"  [train] hazard_avoidance seed={seed} ep {done}/{total_eps}"
          f" survival_gate={'pass' if hz.survival_gate_passed else 'FAIL'}", flush=True)
    if hz.aborted:
        return _fail("hazard", hz.abort_reason)

    p1 = scheduler.run_p1(agent, device)
    done += p1.n_episodes
    print(f"  [train] p1_foraging seed={seed} ep {done}/{total_eps}"
          f" survival_gate={'pass' if p1.survival_gate_passed else 'FAIL'}", flush=True)

    # ---- Frozen-policy P2: run ALL THREE arms on the SAME trained agent ----
    arm0 = _run_p2_arm(scheduler, agent, device, seed, "ARM_0_baseline",
                        0.0, 0.0, "additive", dry_run)
    done += P2_BUDGET if not dry_run else 2
    print(f"  [train] p2_arm0 seed={seed} ep {done}/{total_eps}", flush=True)
    arm1 = _run_p2_arm(scheduler, agent, device, seed, "ARM_1_vigor_additive",
                        VIGOR_W_ACTION_TRAINED, VIGOR_W_PASSIVE_TRAINED, "additive", dry_run)
    done += P2_BUDGET if not dry_run else 2
    print(f"  [train] p2_arm1 seed={seed} ep {done}/{total_eps}", flush=True)
    arm2 = _run_p2_arm(scheduler, agent, device, seed, "ARM_2_vigor_multiplicative",
                        VIGOR_W_ACTION_TRAINED, VIGOR_W_PASSIVE_TRAINED, "multiplicative", dry_run)
    done += P2_BUDGET if not dry_run else 2
    print(f"  [train] p2_arm2 seed={seed} ep {done}/{total_eps}", flush=True)

    g1 = bool(p1.survival_gate_passed)
    baseline_density = float(arm0["action_density"])
    baseline_not_saturated = bool(baseline_density < BASELINE_SATURATION_CEILING)
    seed_valid = bool(g1 and baseline_not_saturated)
    seed_pass = bool(g1)  # harness completed end-to-end; scientific gate is separate
    print(
        f"verdict: {'PASS' if seed_pass else 'FAIL'} seed={seed} g1={g1}"
        f" baseline_density={baseline_density:.4f} valid={seed_valid}",
        flush=True,
    )

    # Restore trained-with vigor weights on the live agent before moving on
    # (hygiene only -- the agent is not reused past this seed).
    tv = getattr(agent, "tonic_vigor", None)
    if tv is not None:
        tv.config.w_action = VIGOR_W_ACTION_TRAINED
        tv.config.w_passive = VIGOR_W_PASSIVE_TRAINED
        tv.config.form = "additive"

    _ZG.observe(agent)
    return {
        "seed": seed, "aborted_at": None, "abort_reason": "",
        "g1_p1_survival": g1,
        "baseline_action_density": baseline_density,
        "baseline_not_saturated": baseline_not_saturated,
        "seed_valid": seed_valid,
        "arms": [arm0, arm1, arm2],
        "seed_pass": seed_pass,
    }


def _frac(flags: List[bool]) -> float:
    return float(sum(1 for f in flags if f)) / float(len(flags)) if flags else 0.0


def run_experiment(dry_run: bool = False) -> Dict[str, Any]:
    print(f"[{EXPERIMENT_TYPE}] starting (dry_run={dry_run})", flush=True)
    seeds = SEEDS[:1] if dry_run else SEEDS
    if dry_run:
        total_eps = 2 + 2 + 5 + 5 + 5 + 3 * 2
    else:
        total_eps = (
            STAGE0_BUDGET + STAGE0B_BUDGET + P0_BUDGET + HAZARD_STAGE_BUDGET
            + P1_BUDGET + 3 * P2_BUDGET
        )

    per_seed: List[Dict[str, Any]] = []
    for s in seeds:
        per_seed.append(_run_seed(s, dry_run, total_eps))

    reached_p2 = [r for r in per_seed if r.get("arms")]
    n_seeds = len(per_seed)

    # ---- Readiness precondition: P1 survival gate on this exact curriculum
    # (reused verbatim from 603n/812's already-validated shape). ----
    g1_frac = _frac([r.get("g1_p1_survival", False) for r in per_seed])
    reached_p2_alive = bool(g1_frac >= MIN_FRACTION)

    # ---- Non-vacuity sanity: ARM_0's bias channel must read exactly zero
    # (confirms the w_action=w_passive=0.0 toggle actually silenced the
    # channel -- the V3-EXQ-643 lesson, applied to the OFF arm instead of a
    # readiness precondition). ----
    arm0_bias_zero_per_seed = []
    for r in reached_p2:
        a0 = next((a for a in r["arms"] if a["arm"] == "ARM_0_baseline"), None)
        arm0_bias_zero_per_seed.append(bool(a0 is not None and a0["bias_max_abs_mean"] == 0.0))
    arm0_bias_zero_ok = bool(all(arm0_bias_zero_per_seed)) if arm0_bias_zero_per_seed else False

    # ---- Substrate-competence validity: G1 survival AND ARM_0 baseline not
    # ceiling-saturated (the exact headroom failure that starved 3/5 seeds in
    # 624c). ----
    valid_seeds = [r for r in per_seed if r.get("seed_valid")]
    valid_frac = _frac([bool(r.get("seed_valid")) for r in per_seed])
    majority_valid = bool(valid_frac >= VALID_SEED_MAJORITY_FRAC)

    # ---- C1 (LOAD-BEARING primary DV): ARM_1 - ARM_0 action_density lift,
    # evaluated only on valid seeds. ----
    per_seed_c1_lift: List[float] = []
    per_seed_c1_pass: List[bool] = []
    per_seed_c2_multiplicative_lift: List[float] = []
    for r in valid_seeds:
        a0 = next(a for a in r["arms"] if a["arm"] == "ARM_0_baseline")
        a1 = next(a for a in r["arms"] if a["arm"] == "ARM_1_vigor_additive")
        a2 = next(a for a in r["arms"] if a["arm"] == "ARM_2_vigor_multiplicative")
        lift1 = float(a1["action_density"] - a0["action_density"])
        lift2 = float(a2["action_density"] - a0["action_density"])
        per_seed_c1_lift.append(lift1)
        per_seed_c1_pass.append(bool(lift1 >= C1_LIFT_MIN))
        per_seed_c2_multiplicative_lift.append(lift2)

    c1_frac = _frac(per_seed_c1_pass) if per_seed_c1_pass else 0.0
    c1_pass = bool(majority_valid and c1_frac >= VALID_SEED_MAJORITY_FRAC)

    # ---- Routing ----
    if not reached_p2_alive:
        outcome = "FAIL"
        label = "substrate_not_ready_requeue"
        evidence_direction = "non_contributory"
    elif not majority_valid:
        # The scaffolded curriculum reliably produces survival-competent
        # policies (per 603n/812) but this run's baselines still saturated on
        # too many seeds for the lift to be measurable -- an env-headroom gap,
        # not a verdict on MECH-320.
        outcome = "FAIL"
        label = "baseline_action_density_saturated_insufficient_headroom"
        evidence_direction = "non_contributory"
    elif not arm0_bias_zero_ok:
        outcome = "FAIL"
        label = "internal_inconsistency_arm0_bias_nonzero"
        evidence_direction = "non_contributory"
    elif c1_pass:
        outcome = "PASS"
        label = "mech320_selection_authority_established_with_genuine_no_op_opportunity"
        evidence_direction = "supports"
    else:
        # Majority of seeds had genuine no-op opportunity (unlike 624c) AND
        # the vigor channel was confirmed silenced correctly in ARM_0, yet the
        # lift did not clear threshold on a majority of those seeds -- real
        # negative evidence under conditions the 624-series could not
        # establish, not an instrument failure.
        outcome = "FAIL"
        label = "mech320_no_selection_authority_despite_no_op_opportunity"
        evidence_direction = "weakens"

    print(
        f"[{EXPERIMENT_TYPE}] g1_frac={g1_frac:.2f} valid_frac={valid_frac:.2f}"
        f" c1_frac={c1_frac:.2f} arm0_bias_zero_ok={arm0_bias_zero_ok}"
        f" -> outcome={outcome} label={label}",
        flush=True,
    )

    preconditions = [
        {
            "name": "reached_p2_alive", "kind": "readiness",
            "description": "P1 survival >= 2/3 seeds so the agent reaches P2 alive "
                            "(same precondition shape 603n/812 already validated on this "
                            "exact curriculum).",
            "control": "P1 survival gate (median episode length last window >= "
                       "scaffold_p1_survival_gate_steps).",
            "measured": float(g1_frac), "threshold": float(MIN_FRACTION),
            "direction": "lower",
            "met": bool(reached_p2_alive),
        },
        {
            "name": "baseline_no_op_opportunity", "kind": "readiness",
            "description": "ARM_0 P2 action_density stays below the saturation ceiling on "
                            "a majority of seeds -- the exact headroom condition that "
                            "starved 3/5 seeds in V3-EXQ-624c.",
            "control": "measured directly from ARM_0's own P2 pass (vigor bias forced to "
                       "zero via w_action=w_passive=0.0).",
            "measured": float(valid_frac), "threshold": float(VALID_SEED_MAJORITY_FRAC),
            "direction": "lower",
            "met": bool(majority_valid),
        },
        {
            "name": "arm0_bias_channel_silenced", "kind": "sanity",
            "description": "ARM_0's tonic-vigor bias_max_abs reads exactly 0.0 on every "
                            "seed that reached P2 -- confirms the w_action=w_passive=0.0 "
                            "toggle actually zeroed the channel (the V3-EXQ-643 lesson, "
                            "applied to the reference arm).",
            "control": "TonicVigor.get_state()['last_bias_max_abs'], accumulated over "
                       "every P2 tick in ARM_0.",
            "measured": 1.0 if arm0_bias_zero_ok else 0.0, "threshold": 1.0,
            "direction": "lower",
            "met": bool(arm0_bias_zero_ok),
        },
    ]
    criteria_non_degenerate = {
        "C1_selection_authority_lift": bool(per_seed_c1_pass),
    }
    criteria = [
        {"name": "C1_selection_authority_lift", "load_bearing": True, "passed": bool(c1_pass)},
    ]

    mean_c1_lift = (sum(per_seed_c1_lift) / len(per_seed_c1_lift)) if per_seed_c1_lift else 0.0
    mean_c2_lift = (
        sum(per_seed_c2_multiplicative_lift) / len(per_seed_c2_multiplicative_lift)
        if per_seed_c2_multiplicative_lift else 0.0
    )

    return {
        "outcome": outcome,
        "evidence_direction": evidence_direction,
        "interpretation": {
            "label": label,
            "preconditions": preconditions,
            "criteria_non_degenerate": criteria_non_degenerate,
            "criteria": criteria,
            "per_seed_c1_lift": per_seed_c1_lift,
            "per_seed_c1_pass": per_seed_c1_pass,
            "mean_c1_lift_additive": mean_c1_lift,
            "c2_informative_form_discrimination": {
                "note": "EXPLORATORY / NOT GATING -- n<=3 seeds cannot power a real "
                        "regression against a pre-existing-action-preference axis. "
                        "Reported as raw per-seed lifts only; a genuine additive-vs-"
                        "multiplicative discrimination needs the parametric sweep the "
                        "original MECH-320 registration (claims.yaml, R3) describes and "
                        "this run does not attempt.",
                "per_seed_multiplicative_lift": per_seed_c2_multiplicative_lift,
                "per_seed_additive_lift": per_seed_c1_lift,
                "mean_multiplicative_lift": mean_c2_lift,
                "mean_additive_lift": mean_c1_lift,
            },
        },
        "gate_summary": {
            "g1_p1_survival_frac": g1_frac,
            "reached_p2_alive": reached_p2_alive,
            "valid_seed_frac": valid_frac,
            "majority_valid": majority_valid,
            "arm0_bias_zero_ok": arm0_bias_zero_ok,
            "c1_frac": c1_frac,
            "c1_pass": c1_pass,
            "c1_lift_min": C1_LIFT_MIN,
            "baseline_saturation_ceiling": BASELINE_SATURATION_CEILING,
            "valid_seed_majority_frac": VALID_SEED_MAJORITY_FRAC,
        },
        "per_seed": per_seed,
        "arm_results": [a for r in per_seed for a in r.get("arms", [])],
    }


def main(dry_run: bool = False) -> Dict[str, Any]:
    t0 = time.perf_counter()
    result = run_experiment(dry_run=dry_run)
    if dry_run:
        print(f"[{EXPERIMENT_TYPE}] dry-run complete; manifest not written.", flush=True)
        return {"outcome": result["outcome"], "manifest_path": None}

    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{timestamp}_v3"
    out_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: Dict[str, Any] = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": timestamp,
        "outcome": result["outcome"],
        "evidence_direction": result["evidence_direction"],
        "sleep_driver_pattern": "N/A (waking goal-pipeline onboarding scheduler; no sleep loop)",
        "substrate": "scaffolded_sd054_onboarding (full curriculum: Stage-0 -> Stage-0b -> P0 -> "
                     "Stage-H -> P1, MECH-320 vigor ON throughout training; frozen-policy P2 run "
                     "THREE times per seed toggling only the live TonicVigor.config object: "
                     "w_action=w_passive=0 baseline, additive form, multiplicative form)",
        "predecessor": "V3-EXQ-624c (non_contributory/substrate_conditional; positive control "
                       "valid on only 2/5 seeds under an ad-hoc hand-tuned regime; confirmed "
                       "autopsy failure_autopsy_V3-EXQ-603g-624c-651a_2026-06-07 names "
                       "scaffolded_sd054_onboarding, ready since 2026-06-11, as the residual "
                       "gate this experiment clears). NOT a supersession -- this experiment "
                       "does not re-test the 624-series' ARC-068 Niv-vs-Salamone dissociation "
                       "(out of scope; see module docstring).",
        "design_note": "3 arms (baseline / additive / multiplicative) on the SAME per-seed "
                       "trained agent (vigor module constructed and live for the entire "
                       "training trajectory, matching V3-EXQ-812's confound-avoidance "
                       "reasoning for MECH-295's cue). RNG reset to the seed value before EACH "
                       "P2 arm pass so all three arms see an identical env realisation "
                       "sequence.",
        "re_derive_brake": {
            "prior_count": 4,
            "threshold": 2,
            "fired_prior": True,
            "released_here": True,
            "released_because": "target substrate named by the most recent counted autopsy "
                                "(V3-EXQ-624c, target_sd_id=scaffolded_sd054_onboarding) is now "
                                "ready (substrate_queue.json ready:true since 2026-06-11, "
                                "V3-EXQ-603n PASS).",
            "counted_autopsy_slugs": [
                "failure_autopsy_604a-624a-630_2026-06-03",
                "failure_autopsy_V3-EXQ-624b_2026-06-07",
                "failure_autopsy_V3-EXQ-603g-624c-651a_2026-06-07",
            ],
        },
        "pre_registered_gates": {
            "reached_p2_alive": f"P1 survival gate >= {MIN_FRACTION:.3f} seeds",
            "baseline_no_op_opportunity": f"ARM_0 P2 action_density < "
                                          f"{BASELINE_SATURATION_CEILING} on >= "
                                          f"{VALID_SEED_MAJORITY_FRAC:.3f} seeds",
            "arm0_bias_channel_silenced": "ARM_0 bias_max_abs_mean == 0.0 (sanity)",
            "C1_selection_authority_lift": f"ARM_1 - ARM_0 action_density >= {C1_LIFT_MIN}, on "
                                           f">= {VALID_SEED_MAJORITY_FRAC:.3f} of VALID seeds "
                                           f"(load-bearing)",
            "C2_form_discrimination": "informative only, NOT gating -- see interpretation block",
        },
        "scaffold_curriculum": {
            "stage0_budget": STAGE0_BUDGET, "stage0b_budget": STAGE0B_BUDGET,
            "p0_budget": P0_BUDGET, "hazard_stage_budget": HAZARD_STAGE_BUDGET,
            "p1_budget": P1_BUDGET, "p2_budget_per_arm": P2_BUDGET, "n_p2_arms": 3,
            "train_steps": TRAIN_STEPS,
            "v_t_floor": V_T_FLOOR,
            "vigor_w_action_trained": VIGOR_W_ACTION_TRAINED,
            "vigor_w_passive_trained": VIGOR_W_PASSIVE_TRAINED,
            "use_modulatory_selection_authority": USE_MODULATORY_AUTHORITY,
            "modulatory_authority_gain": MODULATORY_AUTHORITY_GAIN,
            "noop_class": NOOP_CLASS,
        },
    }
    manifest.update(result)
    out_path = write_flat_manifest(
        manifest,
        out_dir,
        dry_run=False,
        config=CONFIG_SLICE,
        seeds=SEEDS,
        script_path=Path(__file__),
        started_at=t0,
        z_goal_stream_stats=_ZG.stats(),
    )
    print(f"[{EXPERIMENT_TYPE}] manifest -> {out_path}", flush=True)
    print(f"Done. Outcome: {result['outcome']}", flush=True)
    return {"outcome": result["outcome"], "manifest_path": str(out_path)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    _res = main(dry_run=args.dry_run)
    if _res.get("manifest_path"):
        _outcome_raw = str(_res["outcome"]).upper()
        emit_outcome(
            outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
            manifest_path=_res["manifest_path"],
            dry_run=args.dry_run,
        )
