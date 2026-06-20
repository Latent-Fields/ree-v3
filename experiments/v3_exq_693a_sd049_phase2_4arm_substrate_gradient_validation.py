"""
V3-EXQ-693a -- SD-049 Phase-2 4-ARM SUBSTRATE-GRADIENT behavioural validation, MEASUREMENT
RE-ISSUE of V3-EXQ-693 (NEW letter; same scientific question). The WL-scoring harness is
ported from the working V3-EXQ-514r/s/t lineage so the R3 non-vacuity guard can fire.

WHY A RE-ISSUE (failure_autopsy_V3-EXQ-693_2026-06-20, confirmed + user-adjudicated):
V3-EXQ-693 FAILed self-routing substrate_not_ready_requeue / non_vacuity_unmet. The SOLE unmet
guard was R3 (WL-channel-fireable): n_scored_wl_steps_total=0 and run_bank_populated_frac=0.0
across all 4 arms / 3 seeds. The autopsy established this is an INSTRUMENTATION gap, NOT a
substrate ceiling and NOT a falsification: 693 forked from V3-EXQ-514l (pre-514n) per the queue
commit "514l successor", so its wanting!=liking (WL) scoring harness is stale. In 693 tokens DO
bind (distinct_tokens_max = n_resource_types 2/3/5), per-axis drive spread is present
(0.009-0.015), the identity probe PASSES (0.71-0.77, ~4.4k samples) and the ANOVA PASSES (F up
to 1480) -- only the WL scoring leg is hard-zero. The contemporaneous V3-EXQ-514t scores the
SAME ree_core/goal.py IncentiveTokenBank.most_wanted object-bound WL read cleanly
(run_bank_populated_frac=1.0, 72 scored WL steps).

ROOT CAUSE (code trace): 693's WL block read the liking target via
_contacted_resource_type(obs_dict) -- the POST-STEP obs, where the SD-049 consumed-cell tag has
already been cleared by env.step(), so it returns None at the very consumption tick -- and gated
contact on benefit > SEED_BENEFIT_THRESHOLD. So the per-contact scoring conjunction
(most_wanted != None AND rtype != None AND n_distinct >= 2) never co-activated: rtype was None
at the consumption tick. The 514r/s/t lineage reads the AUTHORITATIVE consumed (liking) tag from
the INFO dict (info["sd049_consumed_type_tag_this_tick"], the 681-C4 fix), which the env caches
BEFORE clearing the cell tag (causal_grid_world.py step()), and gates on a genuine consumption
event (consumed_tag is not None). 514t scores 72 WL steps cleanly with this read.

THE FIX (harness port, this script): in _run_arm_eval, the liking target + the L2 token-bind type
are read from info via _consumed_type_tag_from_info(info) (514t/681-C4), and the WL scoring +
token-bind block is gated on consumed_tag is not None. The R1 foraging-contact counter keeps the
benefit-OR-consumed gate so ARM_0 (SD-049 OFF -> no consumed tag emitted) preserves its contact
rate and R1's pass/fail is NOT regressed by the port. The 4-arm substrate gradient
(OFF / 2-type / 3-type+novelty / 5-type), the identity-recovery probe (C_ID), the
discrimination-margin (C_GR), and the per-axis drive ANOVA (C_ANOVA) are UNCHANGED. The three
non-vacuity guards (R1 consumption / R2 identity-probe-fireable / R3 WL-channel-fireable) still
self-route substrate_not_ready_requeue if any is unmet -- NEVER a false weakens.

WATCH ITEM (gated behind R3 in 693, so never adjudicated -- carried into 693a per the autopsy
Section 5 secondary flag): the C_GR discrimination-margin lift was near-zero (ARM_2-ARM_0 =
0.0067 vs threshold 0.4; negative -0.0087 in ARM_3) even though it could not fire under 693's
vacuity. Once the WL channel scores, this near-zero margin may be the REAL SD-049 Phase-2
conversion question. The C_GR margin readout is RETAINED and a margin-near-zero outcome is
PRE-REGISTERED as a likely conversion-side substrate_ceiling / substrate_conditional re-route
candidate (route via /failure-autopsy), NOT a standalone falsification.

THE FOUR PRE-REGISTERED DVs (measured frozen-policy on each arm's P2 env; thresholds are
constants, NOT post-hoc):
  C_ID    encoder identity-recovery: linear-probe accuracy on z_resource (neighborhood-labelled).
          ARM_2 mean >= ID_PROBE_FLOOR (0.6), pooled n >= N_ID_SAMPLES_FLOOR (30).   [SD-049 + SD-015]
  C_GR    goal_resource_r as an IDENTITY-DISCRIMINATION MARGIN (non-saturating; the 514 C5 fix).
          Lift criterion: margin_ARM2 - margin_ARM0 >= GR_LIFT (0.4).                 [SD-049 + SD-015]
  C_WL    object-bound wanting!=liking dissociation fraction (SD-057 read; the ported 514t harness):
          fraction of scored consumption events where MECH-346 most-wanted type != the
          info-tag consumed (liking) type. ARM_2 mean >= WL_FRACTION (0.6).            [SD-049]
  C_ANOVA per-axis homeostatic drive one-way ANOVA: F > F_CRIT_P01 (4.605) in ARM_2.   [SD-049]

THREE NON-VACUITY SELF-ROUTE GUARDS (an unmet precondition -> substrate_not_ready_requeue, FAIL,
BOTH claims non_contributory -- NEVER a false weakens):
  R1 CONSUMPTION: ARM_0 AND ARM_2 each have >= MIN_FRACTION (2/3) seeds that pass the P2 contact
     guard AND reach behav_contact_rate > CONSUMPTION_FLOOR.
  R2 IDENTITY-PROBE-FIREABLE: ARM_2 pooled neighborhood identity samples >= N_ID_SAMPLES_FLOOR.
  R3 WL-CHANNEL-FIREABLE: leg-1 positive control (constructed 2-token bank produces a
     drive-favored most-wanted differing from a designated last-consumed type, pc_separation == 1.0)
     AND leg-2 in-run population (ARM_2 has >= MIN_FRACTION seeds where the bank reached >= 2
     distinct tokens AND >= MIN_SCORED_STEPS scored WL events AND a per-axis-drive spread
     > DRIVE_SPREAD_FLOOR). The 514t info-tag harness is what makes leg-2 fire.

PER-CLAIM EVIDENCE DIRECTION (len(claim_ids) > 1 -> evidence_direction_per_claim REQUIRED):
  Non-vacuity (R1/R2/R3) NOT met -> SD-049 = SD-015 = non_contributory (substrate_not_ready_requeue).
  Non-vacuity met:
    SD-015 = supports if (C_ID and C_GR); mixed if exactly one; weakens if NEITHER.
    SD-049 = supports if (C_ID and C_ANOVA and C_GR and C_WL); mixed if some-but-not-all (a
             WL-flat-only OR C_GR-near-zero outcome is the SD-049 falsifier-branch
             substrate_conditional re-route candidate, NOT a standalone weakens -- routed by
             /failure-autopsy); weakens only if ALL FOUR discrimination criteria fail on a
             forage-competent rich substrate.

claim_ids: SD-049, SD-015
experiment_purpose: evidence
supersedes: V3-EXQ-693 (FAIL/non_contributory; stale 514l WL-scoring harness -> R3 vacuous).
predecessor: V3-EXQ-693 (SD-049 Phase-2 4-arm; instrumentation re-issue) / V3-EXQ-514t
             (working WL-scoring harness ported here).
SLEEP DRIVER: N/A (waking goal-pipeline onboarding scheduler; no sleep loop).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "experiments") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "experiments"))

from experiment_protocol import emit_outcome  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.goal import IncentiveTokenBank  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402
from scaffolded_sd054_onboarding import (  # noqa: E402
    ScaffoldedSD054OnboardingConfig,
    ScaffoldedSD054OnboardingScheduler,
    _build_env,
    _sense_with_optional_harm,
    stage_plan,
)

EXPERIMENT_TYPE = "v3_exq_693a_sd049_phase2_4arm_substrate_gradient_validation"
QUEUE_ID = "V3-EXQ-693a"
CLAIM_IDS: List[str] = ["SD-049", "SD-015"]
EXPERIMENT_PURPOSE = "evidence"
SUPERSEDES = "V3-EXQ-693"

SEEDS = [42, 43, 44]

# --- 4-arm substrate gradient (matched to the V3-EXQ-513 / 514l / 693 validated arm configs) ---
# (arm_id, label, sd049_on, n_resource_types)
ARM_SPECS: List[Tuple[str, str, bool, int]] = [
    ("ARM_0", "OFF_single_resource", False, 1),
    ("ARM_1", "2type_homeostatic", True, 2),
    ("ARM_2", "3type_with_novelty", True, 3),   # PRIMARY measurement arm
    ("ARM_3", "5type_overshoot", True, 5),
]
PRIMARY_ARM = "ARM_2"
CONTROL_ARM = "ARM_0"

# --- Goal-pipeline / encoder dims (mirror 603n / 514t / 693 exactly) ---
WORLD_DIM = 32
HARM_A_DIM = 16
HARM_OBS_A_DIM = 7
HARM_HISTORY_LEN = 10
DRIVE_WEIGHT = 2.0

# --- Curriculum budgets (mirror 603n / 693 exactly) ---
STAGE0_BUDGET = 20
STAGE0B_BUDGET = 10
P0_BUDGET = 100
HAZARD_STAGE_BUDGET = 40
P1_BUDGET = 50
P2_BUDGET = 15            # the 603n-canonical contact-guard measurement (run_p2)
BEHAV_EVAL_EPISODES = 15  # the SD-049 Phase-2 behavioural DV measurement (this script)
TRAIN_STEPS = 200
P1_HOLD_FRACTION = 0.3
P0_NUM_HAZARDS = 1
P2_HFA_GUARD = 0.3
P1_REEF_SPAWN_HOLD_FRACTION = 0.4

HAZARD_STAGE_NUM_HAZARDS = 4
HAZARD_STAGE_NUM_RESOURCES = 2
HAZARD_STAGE_HFA = 0.0
HAZARD_STAGE_PROXIMITY_HARM = 0.1
HAZARD_STAGE_SPAWN_IN_REEF = True
HAZARD_STAGE_SURVIVAL_GATE_STEPS = 75
HAZARD_STAGE_STABILITY_WINDOW = 10

# --- 634c seeding calibration + SD-057 cue-recall bridge (mirror 603n / 514t) ---
SEED_GAIN = 1.5
SEED_BENEFIT_THRESHOLD = 0.02
SEED_DRIVE_FLOOR = 0.9
CUE_RECALL_GAIN = 0.2

# --- SD-058 / MECH-357 protective-scaffold anneal (mirror 603n / 514t) ---
AVOIDANCE_SCAFFOLD_FLOOR_START = 0.8
AVOIDANCE_SCAFFOLD_FLOOR_END = 0.0
AVOIDANCE_THREAT_REF = 0.35
PAG_THETA_FREEZE = 0.8
PAG_DURATION_INPUT_THRESHOLD = 0.2
HARM_PATHWAY_LR = 1e-3
STAGE0B_RETENTION_GATE = 0.75

# --- Pre-registered acceptance thresholds (NOT derived from the run) ---
P2_ZGOAL_GATE = 0.4          # per-seed contact-guard: z_goal_norm_at_contact_peak floor (603n G3)
CONTACT_GATE = 0.0           # per-seed contact-guard: P2 contact_rate floor (603n G2)
CONSUMPTION_FLOOR = 0.02     # R1: behav-eval contact rate floor (>> the 514l ~0.002 degeneracy)
MIN_FRACTION = 2.0 / 3.0     # >= 2/3 seeds for any guard / aggregate
ID_PROBE_FLOOR = 0.6         # C_ID: identity-recovery linear-probe accuracy (514l C2b / SD-049 spec)
N_ID_SAMPLES_FLOOR = 30      # R2 + C_ID: pooled neighborhood identity-sample floor (514l C2c)
GR_LIFT = 0.4                # C_GR: goal_resource_r DISCRIMINATION-MARGIN lift ARM_2 - ARM_0 (SD-049 spec)
WL_FRACTION = 0.6            # C_WL: object-bound wanting!=liking dissoc fraction ARM_2 (SD-049 spec)
# C_ANOVA: one-way F-critical at p=0.01, df1 = n_axes-1, df2 ~ inf. F(0.01, 2, inf) = 4.605.
# scipy unavailable on runners -> pre-registered critical gate (514m precedent).
F_CRIT_P01 = 4.605

# --- WL same-statistic non-vacuity readiness gate (R3; 514n precedent) ---
PC_SEPARATION_FLOOR = 1.0    # leg 1: the constructed positive control MUST separate (deterministic)
MIN_SCORED_STEPS = 5         # leg 2: min consumption events with both targets defined per seed
DRIVE_SPREAD_FLOOR = 1e-3    # leg 2: min per-axis-drive spread for a genuine drive-differentiated test


def _consumed_type_tag_from_info(info: Dict[str, Any]) -> Optional[int]:
    """681-C4 / 514t fix: the AUTHORITATIVE consumed (liking) tag is in the INFO dict.

    The env (causal_grid_world.py step()) caches the consumed-cell type tag BEFORE clearing
    the cell, surfacing it as info["sd049_consumed_type_tag_this_tick"] (1..n on a consumption
    tick; 0 / absent otherwise). 693's stale read of obs_dict (post-clear) returned None at
    the consumption tick, hard-zeroing the WL scoring conjunction; this read fixes it."""
    if not isinstance(info, dict):
        return None
    raw = info.get("sd049_consumed_type_tag_this_tick", 0)
    try:
        tag = int(raw[0] if hasattr(raw, "__len__") else raw)
    except (TypeError, ValueError):
        return None
    return tag if tag > 0 else None


def _make_scaffold_cfg(dry_run: bool, sd049_on: bool, n_resource_types: int) -> ScaffoldedSD054OnboardingConfig:
    """Full 603n curriculum config, parametrised per arm. SD-049 substrate (and therefore the
    cue-recall bridge) is enabled with n_resource_types for ARM_1/2/3; OFF for ARM_0."""
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
        # developmental-window / consolidation amend (2026-06-03b)
        scaffold_developmental_window_enabled=True,
        scaffold_stage0b_enabled=True,
        scaffold_stage0b_episode_budget=stage0b,
        scaffold_stage0b_retention_gate=STAGE0B_RETENTION_GATE,
        scaffold_contact_gated_goal_updates=True,
        # 634c seeding calibration (2026-06-03c)
        scaffold_z_goal_seeding_gain=SEED_GAIN,
        scaffold_benefit_threshold=SEED_BENEFIT_THRESHOLD,
        scaffold_drive_floor=SEED_DRIVE_FLOOR,
        # foraging-competence residual amend (2026-06-05)
        scaffold_auto_reconcile_gating_to_seeding=True,
        scaffold_p1_reef_spawn_hold_fraction=P1_REEF_SPAWN_HOLD_FRACTION,
        # SD-057 cue-recall bridge -- enables SD-049 in the scaffold envs (per arm)
        scaffold_cue_recall_bridge_enabled=bool(sd049_on),
        scaffold_cue_n_resource_types=int(n_resource_types),
        scaffold_stage0_bind_incentive_token=True,
        # curriculum-decomposition amend (2026-06-07): isolated Stage-H
        scaffold_hazard_stage_enabled=True,
        scaffold_hazard_stage_episode_budget=hazard,
        scaffold_hazard_stage_num_hazards=HAZARD_STAGE_NUM_HAZARDS,
        scaffold_hazard_stage_num_resources=HAZARD_STAGE_NUM_RESOURCES,
        scaffold_hazard_stage_hazard_food_attraction=HAZARD_STAGE_HFA,
        scaffold_hazard_stage_proximity_harm_scale=HAZARD_STAGE_PROXIMITY_HARM,
        scaffold_hazard_stage_spawn_in_reef_half=HAZARD_STAGE_SPAWN_IN_REEF,
        scaffold_hazard_stage_survival_gate_steps=HAZARD_STAGE_SURVIVAL_GATE_STEPS,
        scaffold_hazard_stage_stability_window=HAZARD_STAGE_STABILITY_WINDOW,
        # SD-058 / MECH-357 avoidance-learning driver (mirror 603n)
        scaffold_avoidance_driver_enabled=True,
        scaffold_avoidance_scaffold_floor_start=AVOIDANCE_SCAFFOLD_FLOOR_START,
        scaffold_avoidance_scaffold_floor_end=AVOIDANCE_SCAFFOLD_FLOOR_END,
        # PREREQUISITE: feed the env harm stream so z_harm / z_harm_a populate
        scaffold_feed_harm_stream=True,
        # harm-pathway training (2026-06-09 amend; ON, validated by 603k/603n)
        scaffold_train_harm_pathway=True,
        scaffold_harm_pathway_lr=HARM_PATHWAY_LR,
        scaffold_harm_pathway_in_p0=True,
    )
    if steps < 75:
        cfg.scaffold_p1_survival_gate_steps = max(1, steps // 4)
        cfg.scaffold_hazard_stage_survival_gate_steps = max(1, steps // 4)
    return cfg


def _make_config(env) -> REEConfig:
    """Mirror 514t: full SD-049 Phase-2 + SD-057 substrate. z_resource encoder on (SD-015);
    incentive token bank on (SD-057, the object-bound WL read)."""
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
    )
    cfg.latent.use_resource_encoder = True   # SD-015 (z_resource -> bank L2 bind + identity probe)
    return cfg


def _ben_drive(obs_body: torch.Tensor) -> Tuple[float, float]:
    b = obs_body.reshape(-1)
    benefit = float(b[11].item()) if b.shape[0] > 11 else 0.0
    energy = float(b[3].item()) if b.shape[0] > 3 else 0.5
    drive = max(0.0, min(1.0, 1.0 - energy))
    return benefit, drive


def _per_axis_drive_from_obs(obs_dict: Dict[str, Any], device: torch.device) -> Optional[torch.Tensor]:
    pad = obs_dict.get("per_axis_drive", None)
    if pad is None:
        return None
    if hasattr(pad, "to"):
        return pad.to(device)
    return torch.as_tensor(np.asarray(pad, dtype=np.float32), device=device)


def _neighborhood_dominant_type(obs_dict: Dict[str, Any], type_names: Tuple[str, ...]) -> int:
    """Label z_resource at a step by the type whose 5x5 field-view max is largest. Returns
    0..n_types-1 when at least one type has meaningful presence, else -1 (no discriminable
    signal). Mirrors V3-EXQ-514l / 514m / 693."""
    if not type_names:
        return -1
    field_maxes: List[float] = []
    for name in type_names:
        key = f"resource_field_view_{name}"
        if key not in obs_dict:
            return -1
        v = obs_dict[key]
        if hasattr(v, "max"):
            field_maxes.append(float(v.max().item()) if hasattr(v.max(), "item") else float(v.max()))
        else:
            return -1
    if not field_maxes or max(field_maxes) < 0.05:
        return -1
    return int(np.argmax(field_maxes))


def _identity_recovery_probe(z_samples: List[np.ndarray], targets: List[int]) -> Tuple[float, int]:
    """Train a linear classifier on (z_resource, type) and return (held-out accuracy, n_samples).
    Mirrors V3-EXQ-514l / 514m / 693 identity_recovery_probe."""
    n = len(z_samples)
    if n < 10:
        return 0.0, n
    X = np.stack(z_samples)
    y = np.array(targets)
    n_train = max(1, int(n * 0.7))
    perm = np.random.permutation(n)
    X_train, y_train = X[perm[:n_train]], y[perm[:n_train]]
    X_eval, y_eval = X[perm[n_train:]], y[perm[n_train:]]
    if len(X_eval) == 0 or len(np.unique(y_train)) < 2:
        return 0.0, n
    n_classes = int(max(y_train.max(), y_eval.max())) + 1
    z_dim = X_train.shape[1]
    probe = nn.Linear(z_dim, n_classes)
    opt = optim.Adam(probe.parameters(), lr=1e-2)
    Xt = torch.tensor(X_train, dtype=torch.float32)
    yt = torch.tensor(y_train, dtype=torch.long)
    for _ in range(200):
        opt.zero_grad()
        loss = F.cross_entropy(probe(Xt), yt)
        loss.backward()
        opt.step()
    with torch.no_grad():
        Xe = torch.tensor(X_eval, dtype=torch.float32)
        pred = probe(Xe).argmax(dim=-1).numpy()
        return float((pred == y_eval).mean()), n


def _one_way_anova_f(groups: List[np.ndarray]) -> Tuple[float, int, int]:
    """One-way ANOVA F statistic across `groups` (each a 1-D sample array). Returns (F, df1, df2).
    scipy-free; gate uses a pre-registered F-critical. Mirrors V3-EXQ-514m / 693."""
    groups = [g for g in groups if g.size > 0]
    k = len(groups)
    if k < 2:
        return 0.0, 0, 0
    grand = np.concatenate(groups)
    N = grand.size
    grand_mean = float(grand.mean())
    ss_between = float(sum(g.size * (float(g.mean()) - grand_mean) ** 2 for g in groups))
    ss_within = float(sum(((g - float(g.mean())) ** 2).sum() for g in groups))
    df1 = k - 1
    df2 = N - k
    if df2 <= 0 or ss_within <= 0.0:
        return 0.0, df1, max(df2, 0)
    f_stat = (ss_between / df1) / (ss_within / df2)
    return float(f_stat), df1, df2


def _discrimination_margin(
    z_goal_samples: List[np.ndarray], z_goal_types: List[int],
    z_res_samples: List[np.ndarray], z_res_types: List[int],
) -> float:
    """The 514 C5 FIX: a NON-SATURATING goal_resource_r as an identity-discrimination margin.
    For each goal-active sample i (z_goal_i, dominant type t_i): cos(z_goal_i, mean_zres[t_i])
    minus the mean over OTHER types t' of cos(z_goal_i, mean_zres[t']). margin = mean_i of that.
    Requires >= 2 distinct types with z_resource samples; else 0.0 (no discrimination possible --
    the correct ARM_0 negative-control value, since the legacy saturating cosine cannot apply)."""
    if not z_goal_samples or not z_res_samples:
        return 0.0
    by_type: Dict[int, List[np.ndarray]] = {}
    for z, t in zip(z_res_samples, z_res_types):
        by_type.setdefault(int(t), []).append(z)
    type_means: Dict[int, np.ndarray] = {}
    for t, zs in by_type.items():
        m = np.mean(np.stack(zs), axis=0)
        nrm = float(np.linalg.norm(m))
        type_means[t] = m / nrm if nrm > 1e-8 else m
    if len(type_means) < 2:
        return 0.0

    def _cos(a: np.ndarray, b: np.ndarray) -> float:
        na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
        if na < 1e-8 or nb < 1e-8:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    margins: List[float] = []
    for zg, t in zip(z_goal_samples, z_goal_types):
        t = int(t)
        if t not in type_means:
            continue
        same = _cos(zg, type_means[t])
        diffs = [_cos(zg, type_means[t2]) for t2 in type_means if t2 != t]
        if not diffs:
            continue
        margins.append(same - float(np.mean(diffs)))
    return float(np.mean(margins)) if margins else 0.0


def _positive_control_separation(agent, device: torch.device) -> float:
    """R3 WL readiness leg 1 (514n): build a 2-token IncentiveTokenBank from THIS run's GoalConfig
    and verify the cross-target inequality the C_WL criterion routes on CAN fire. Two deterministic
    probes: drive-favored most-wanted must differ from a designated last-consumed type. Returns the
    cross-target-inequality fraction (1.0 if the instrument works)."""
    goal_cfg = agent.config.goal
    d = int(getattr(goal_cfg, "goal_dim", WORLD_DIM))
    pc_bank = IncentiveTokenBank(goal_cfg, device)
    z_a = torch.zeros(1, d, device=device); z_a[0, 0] = 1.0   # type 1 identity embedding
    z_b = torch.zeros(1, d, device=device); z_b[0, 1] = 1.0   # type 2 identity embedding
    pc_bank.update(1, 1.0, z_a)   # bind food token (tag 1)
    pc_bank.update(2, 1.0, z_b)   # bind water token (tag 2)
    pad1 = torch.zeros(1, 2, device=device); pad1[0, 1] = 1.0  # drive favours type 2
    mw1 = pc_bank.most_wanted(per_axis_drive=pad1, scalar_drive=1.0)
    sep1 = 1.0 if (mw1 is not None and int(mw1[0]) != 1) else 0.0
    pad2 = torch.zeros(1, 2, device=device); pad2[0, 0] = 1.0  # drive favours type 1
    mw2 = pc_bank.most_wanted(per_axis_drive=pad2, scalar_drive=1.0)
    sep2 = 1.0 if (mw2 is not None and int(mw2[0]) != 2) else 0.0
    return float((sep1 + sep2) / 2.0)


def _run_arm_eval(agent, scaffold_cfg, device: torch.device, n_eps: int, n_axes: int) -> Dict[str, Any]:
    """Frozen-policy SD-049 Phase-2 behavioural eval on this arm's P2 env. Collects all four DVs:
    C_ID (neighborhood-labelled z_resource identity probe), C_GR (discrimination margin), C_WL
    (object-bound most-wanted vs the info-tag consumed/liking type -- the 514t harness port),
    C_ANOVA (per-axis drive). z_goal is refreshed at genuine contact via agent.update_z_goal
    (the substrate's own L2/L4 path)."""
    env = _build_env(scaffold_cfg, "p2")
    env.reset()
    type_names = tuple(getattr(env, "resource_type_names", ()) or ())
    bank = getattr(agent.goal_state, "incentive_bank", None)

    # identity probe + discrimination-margin sample buffers
    z_res_samples: List[np.ndarray] = []
    z_res_types: List[int] = []
    z_goal_samples: List[np.ndarray] = []
    z_goal_types: List[int] = []
    # legacy saturating cosine (diagnostic only)
    cosine_sims: List[float] = []
    # object-bound WL
    scored_wl_steps = 0
    wl_dissoc_steps = 0
    distinct_tokens_max = 0
    drive_spread_max = 0.0
    # per-axis drive ANOVA
    per_axis_drive_samples: List[np.ndarray] = []
    # consumption / contact
    contact_steps = 0
    total_steps = 0

    steps_per_ep = scaffold_cfg.scaffold_steps_per_episode
    for _ep in range(n_eps):
        _, obs_dict = env.reset()
        agent.reset()
        for _step in range(steps_per_ep):
            obs_body = obs_dict["body_state"].to(device)
            obs_world = obs_dict["world_state"].to(device)
            with torch.no_grad():
                latent = _sense_with_optional_harm(
                    agent, obs_body, obs_world, obs_dict, device,
                    scaffold_cfg.scaffold_feed_harm_stream,
                )
                ticks = agent.clock.advance()
                e1_prior = (
                    agent._e1_tick(latent) if ticks.get("e1_tick")
                    else torch.zeros(1, agent.config.latent.world_dim, device=device)
                )
                candidates = agent.generate_trajectories(latent, e1_prior, ticks)
                action = agent.select_action(candidates, ticks)

            # --- pre-step DV reads at the post-sense latent ---
            goal_state = getattr(agent, "goal_state", None)
            nb_label = _neighborhood_dominant_type(obs_dict, type_names) if type_names else -1

            if goal_state is not None and goal_state.is_active():
                with torch.no_grad():
                    z_g = goal_state.z_goal
                    if z_g.ndim == 1:
                        z_g = z_g.unsqueeze(0)
                    # legacy saturating cosine (diagnostic)
                    z_ref = latent.z_resource if latent.z_resource is not None else latent.z_world
                    if z_ref.ndim == 1:
                        z_ref = z_ref.unsqueeze(0)
                    if z_ref.shape[-1] == z_g.shape[-1]:
                        cosine_sims.append(float(F.cosine_similarity(z_g, z_ref, dim=-1).mean().item()))
                    # discrimination-margin: collect z_goal labelled by neighborhood type
                    if nb_label >= 0:
                        z_goal_samples.append(z_g.detach().cpu().numpy().squeeze())
                        z_goal_types.append(nb_label)

            # C_ID: neighborhood-labelled z_resource samples
            if latent.z_resource is not None and nb_label >= 0:
                z_res_samples.append(latent.z_resource.detach().cpu().numpy().squeeze())
                z_res_types.append(nb_label)

            # C_ANOVA: per-axis homeostatic drive (per-step vector)
            pad = _per_axis_drive_from_obs(obs_dict, device)
            if pad is not None:
                arr = np.asarray(
                    pad.detach().cpu().numpy() if hasattr(pad, "detach") else pad,
                    dtype=np.float32,
                ).reshape(-1)
                if arr.size >= n_axes >= 2:
                    per_axis_drive_samples.append(arr[:n_axes].copy())

            action_idx = int(action.argmax(dim=-1).item())
            _, harm_signal, done, info, obs_dict = env.step(action_idx)
            total_steps += 1

            # --- post-step contact: liking target from the AUTHORITATIVE info tag (514t/681-C4
            #     harness port; the obs_dict cell tag is already cleared post-step) ---
            benefit, drive = _ben_drive(obs_dict["body_state"].to(device))
            consumed_tag = _consumed_type_tag_from_info(info)  # liking target (last-consumed type)

            # R1 foraging-contact counter: benefit pulse OR a registered consumption event. ARM_0
            # (SD-049 OFF) emits no consumed tag, so the benefit leg preserves its contact rate
            # and the R1 pass/fail behaviour is NOT regressed by the WL-harness port.
            if benefit > SEED_BENEFIT_THRESHOLD or consumed_tag is not None:
                contact_steps += 1

            # WL scoring + L2 token-bind: gated on a GENUINE consumption event (the 514t harness),
            # so the per-contact scoring conjunction can co-activate (the 693 R3 vacuity fix).
            if consumed_tag is not None:
                pad2 = _per_axis_drive_from_obs(obs_dict, device)
                if pad2 is not None:
                    agent._per_axis_drive = pad2.reshape(-1)
                    flat = pad2.reshape(-1)
                    if flat.numel() >= 2:
                        drive_spread_max = max(drive_spread_max,
                                               float(flat.max().item() - flat.min().item()))
                with torch.no_grad():
                    try:
                        agent.update_z_goal(float(benefit), drive_level=float(drive),
                                            resource_type=consumed_tag)
                    except TypeError:
                        agent.update_z_goal(float(benefit), drive_level=float(drive))
                    # WANTING target = MECH-346 most-wanted pointer (the object that seeds z_goal)
                    if bank is not None and not bank.is_empty():
                        n_distinct = len(bank.wanting(
                            per_axis_drive=getattr(agent, "_per_axis_drive", None),
                            scalar_drive=float(drive),
                        ))
                        distinct_tokens_max = max(distinct_tokens_max, n_distinct)
                        mw = bank.most_wanted(
                            per_axis_drive=getattr(agent, "_per_axis_drive", None),
                            scalar_drive=float(drive),
                        )
                        # SCORE only when most-wanted is defined AND the bank holds >= 2 distinct
                        # tokens (the consumed/liking target is guaranteed non-None by the gate).
                        if mw is not None and n_distinct >= 2:
                            scored_wl_steps += 1
                            if int(mw[0]) != int(consumed_tag):
                                wl_dissoc_steps += 1

            if done:
                break

    probe_acc, n_id = _identity_recovery_probe(z_res_samples, z_res_types)
    disc_margin = _discrimination_margin(z_goal_samples, z_goal_types, z_res_samples, z_res_types)
    wl_fraction = (float(wl_dissoc_steps) / float(scored_wl_steps)) if scored_wl_steps > 0 else 0.0
    legacy_goal_resource_r = float(np.mean(cosine_sims)) if cosine_sims else 0.0
    behav_contact_rate = (float(contact_steps) / float(total_steps)) if total_steps > 0 else 0.0
    run_populated = bool(
        distinct_tokens_max >= 2
        and scored_wl_steps >= MIN_SCORED_STEPS
        and drive_spread_max > DRIVE_SPREAD_FLOOR
    )

    per_axis_groups: List[np.ndarray] = []
    if per_axis_drive_samples:
        mat = np.stack(per_axis_drive_samples)
        for a in range(mat.shape[1]):
            per_axis_groups.append(mat[:, a].astype(np.float64))
    anova_f, anova_df1, anova_df2 = _one_way_anova_f(per_axis_groups)

    return {
        "probe_acc_identity": probe_acc,
        "n_identity_samples": n_id,
        "discrimination_margin": disc_margin,
        "object_bound_wl_dissoc_fraction": wl_fraction,
        "n_scored_wl_steps": scored_wl_steps,
        "n_wl_dissoc_steps": wl_dissoc_steps,
        "distinct_tokens_max": distinct_tokens_max,
        "drive_spread_max": drive_spread_max,
        "run_bank_populated": run_populated,
        "per_axis_drive_anova_f": anova_f,
        "per_axis_drive_anova_df1": anova_df1,
        "per_axis_drive_anova_df2": anova_df2,
        "n_per_axis_drive_samples": len(per_axis_drive_samples),
        "legacy_goal_resource_r_diagnostic": legacy_goal_resource_r,
        "n_cosine_samples": len(cosine_sims),
        "behav_contact_rate": behav_contact_rate,
        "n_z_goal_samples": len(z_goal_samples),
    }


def _aborted_record(seed: int, arm_id: str, stage: str, reason: str) -> Dict[str, Any]:
    return {
        "seed": seed, "arm": arm_id, "aborted_at": stage, "abort_reason": reason,
        "guard_pass": False, "contributes": False,
        "p2_contact_rate": 0.0, "p2_z_goal_norm_at_contact_peak": 0.0, "p2_num_contact_events": 0,
        "probe_acc_identity": 0.0, "n_identity_samples": 0, "discrimination_margin": 0.0,
        "object_bound_wl_dissoc_fraction": 0.0, "n_scored_wl_steps": 0, "n_wl_dissoc_steps": 0,
        "distinct_tokens_max": 0, "drive_spread_max": 0.0, "run_bank_populated": False,
        "per_axis_drive_anova_f": 0.0, "n_per_axis_drive_samples": 0,
        "legacy_goal_resource_r_diagnostic": 0.0, "behav_contact_rate": 0.0, "n_z_goal_samples": 0,
        "pc_separation": 0.0,
    }


def _run_seed_arm(seed: int, arm_id: str, label: str, sd049_on: bool, n_types: int,
                  dry_run: bool, total_eps: int) -> Dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)  # deterministic identity-probe train/eval split
    scaffold_cfg = _make_scaffold_cfg(dry_run, sd049_on, n_types)
    device = torch.device("cpu")
    n_axes = int(n_types) if sd049_on else 1

    probe_env = _build_env(scaffold_cfg, "p2")
    probe_env.reset()
    agent = REEAgent(_make_config(probe_env)).to(device)
    scheduler = ScaffoldedSD054OnboardingScheduler(scaffold_cfg)

    # Canonical seed/condition boundary line (runner resets episodes_in_run on this).
    print(f"Seed {seed} Condition {arm_id}_{label}", flush=True)
    print(f"[{arm_id}/{label}] seed {seed} sd049={sd049_on} n_types={n_types}", flush=True)
    done = 0

    s0 = scheduler.run_stage0_nursery(agent, device)
    done += s0.n_episodes
    if s0.aborted:
        print(f"verdict: FAIL arm={arm_id} seed={seed} aborted_at=stage0 reason={s0.abort_reason}", flush=True)
        return _aborted_record(seed, arm_id, "stage0", s0.abort_reason)

    s0b = scheduler.run_stage0b_consolidation(agent, device, stage0_baseline_norm=s0.z_goal_norm_peak)
    done += s0b.n_episodes
    if s0b.aborted:
        print(f"verdict: FAIL arm={arm_id} seed={seed} aborted_at=stage0b reason={s0b.abort_reason}", flush=True)
        return _aborted_record(seed, arm_id, "stage0b", s0b.abort_reason)

    p0 = scheduler.run_p0(agent, device)
    done += p0.n_episodes
    if p0.aborted:
        print(f"verdict: FAIL arm={arm_id} seed={seed} aborted_at=p0 reason={p0.abort_reason}", flush=True)
        return _aborted_record(seed, arm_id, "p0", p0.abort_reason)

    hz = scheduler.run_hazard_avoidance(agent, device)
    done += hz.n_episodes
    if hz.aborted:
        print(f"verdict: FAIL arm={arm_id} seed={seed} aborted_at=hazard reason={hz.abort_reason}", flush=True)
        return _aborted_record(seed, arm_id, "hazard", hz.abort_reason)

    p1 = scheduler.run_p1(agent, device)
    done += p1.n_episodes

    # --- 603n-canonical contact guard via run_p2 (consumption-event-gated readout) ---
    p2 = scheduler.run_p2(agent, device)
    done += p2.n_episodes
    print(f"  [train] p2_guard arm={arm_id} seed={seed} ep {done}/{total_eps}"
          f" contact_rate={p2.contact_rate:.4f} events={p2.num_contact_events}"
          f" z_goal_at_contact={p2.z_goal_norm_at_contact_peak:.4f}", flush=True)

    guard_pass = bool(
        p2.contact_rate > CONTACT_GATE
        and p2.z_goal_norm_at_contact_peak > P2_ZGOAL_GATE
    )

    # --- SD-049 Phase-2 behavioural DVs (always measured; gated at aggregation) ---
    behav = _run_arm_eval(agent, scaffold_cfg, device, BEHAV_EVAL_EPISODES, n_axes)
    done += BEHAV_EVAL_EPISODES
    pc_sep = _positive_control_separation(agent, device)

    # R1 consumption: a seed CONTRIBUTES to its arm only if it forages competently AND consumes.
    contributes = bool(guard_pass and behav["behav_contact_rate"] > CONSUMPTION_FLOOR)

    print(f"  [eval] arm={arm_id} seed={seed} contributes={contributes}"
          f" id_acc={behav['probe_acc_identity']:.3f} (n={behav['n_identity_samples']})"
          f" margin={behav['discrimination_margin']:.3f}"
          f" wl_frac={behav['object_bound_wl_dissoc_fraction']:.3f} (scored={behav['n_scored_wl_steps']})"
          f" anova_f={behav['per_axis_drive_anova_f']:.3f}"
          f" contact={behav['behav_contact_rate']:.4f} pc_sep={pc_sep:.2f}", flush=True)

    # Per-cell verdict line (one per seed x arm; increments runner runs_done). A cell PASSES
    # if it contributes (forages + consumes); the overall PASS/FAIL is decided in run_experiment.
    print(f"verdict: {'PASS' if contributes else 'FAIL'} arm={arm_id} seed={seed}"
          f" contributes={contributes} guard_pass={guard_pass}", flush=True)

    rec: Dict[str, Any] = {
        "seed": seed, "arm": arm_id, "label": label, "sd049_on": sd049_on, "n_resource_types": n_types,
        "aborted_at": None, "abort_reason": "",
        "guard_pass": guard_pass, "contributes": contributes,
        "stage0_z_goal_norm_peak": float(s0.z_goal_norm_peak),
        "p1_survival_pass": bool(p1.survival_gate_passed),
        "hazard_stage_survival_pass": bool(hz.survival_gate_passed),
        "p2_contact_rate": float(p2.contact_rate),
        "p2_z_goal_norm_at_contact_peak": float(p2.z_goal_norm_at_contact_peak),
        "p2_num_contact_events": int(p2.num_contact_events),
        "pc_separation": float(pc_sep),
    }
    rec.update(behav)
    return rec


def _frac(flags: List[bool]) -> float:
    return float(sum(1 for f in flags if f)) / float(len(flags)) if flags else 0.0


def _arm_records(per_run: List[Dict[str, Any]], arm_id: str) -> List[Dict[str, Any]]:
    return [r for r in per_run if r.get("arm") == arm_id]


def _arm_contributing(per_run: List[Dict[str, Any]], arm_id: str) -> List[Dict[str, Any]]:
    return [r for r in _arm_records(per_run, arm_id) if r.get("contributes")]


def _mean_over(records: List[Dict[str, Any]], key: str) -> float:
    vals = [r[key] for r in records if key in r]
    return float(np.mean(vals)) if vals else 0.0


def run_experiment(dry_run: bool = False) -> Dict[str, Any]:
    print(f"[{EXPERIMENT_TYPE}] starting (dry_run={dry_run})", flush=True)
    seeds = SEEDS[:1] if dry_run else SEEDS
    arms = ARM_SPECS[:2] if dry_run else ARM_SPECS  # dry-run exercises ARM_0 + ARM_1 only (cheap)
    if dry_run:
        per_seed_eps = 2 + 2 + 5 + 5 + 5 + 2 + BEHAV_EVAL_EPISODES
    else:
        per_seed_eps = (STAGE0_BUDGET + STAGE0B_BUDGET + P0_BUDGET + HAZARD_STAGE_BUDGET
                        + P1_BUDGET + P2_BUDGET + BEHAV_EVAL_EPISODES)
    total_eps = per_seed_eps * len(seeds) * len(arms)

    per_run: List[Dict[str, Any]] = []
    for arm_id, label, sd049_on, n_types in arms:
        for s in seeds:
            per_run.append(_run_seed_arm(s, arm_id, label, sd049_on, n_types, dry_run, per_seed_eps))

    # ---- aggregate per arm over contributing seeds ----
    arm_summary: Dict[str, Any] = {}
    for arm_id, label, sd049_on, n_types in arms:
        contrib = _arm_contributing(per_run, arm_id)
        allrec = _arm_records(per_run, arm_id)
        arm_summary[arm_id] = {
            "label": label, "sd049_on": sd049_on, "n_resource_types": n_types,
            "n_contributing": len(contrib),
            "contribute_frac": _frac([r["contributes"] for r in allrec]),
            "mean_probe_acc_identity": _mean_over(contrib, "probe_acc_identity"),
            "pooled_n_identity_samples": int(sum(r["n_identity_samples"] for r in contrib)),
            "mean_discrimination_margin": _mean_over(contrib, "discrimination_margin"),
            "mean_object_bound_wl_dissoc_fraction": _mean_over(contrib, "object_bound_wl_dissoc_fraction"),
            "n_scored_wl_steps_total": int(sum(r["n_scored_wl_steps"] for r in contrib)),
            "run_bank_populated_frac": _frac([r["run_bank_populated"] for r in contrib]),
            "per_axis_drive_anova_f_max": max((r["per_axis_drive_anova_f"] for r in contrib), default=0.0),
            "mean_behav_contact_rate": _mean_over(contrib, "behav_contact_rate"),
            "mean_legacy_goal_resource_r_diagnostic": _mean_over(contrib, "legacy_goal_resource_r_diagnostic"),
        }

    primary = arm_summary.get(PRIMARY_ARM, {})
    control = arm_summary.get(CONTROL_ARM, {})

    # ---- NON-VACUITY GUARDS ----
    # R1 consumption: control + primary each have >= MIN_FRACTION contributing seeds.
    r1_consumption = bool(
        control.get("contribute_frac", 0.0) >= MIN_FRACTION
        and primary.get("contribute_frac", 0.0) >= MIN_FRACTION
    )
    # R2 identity-probe-fireable: primary pooled identity samples >= floor.
    r2_identity_fireable = bool(primary.get("pooled_n_identity_samples", 0) >= N_ID_SAMPLES_FLOOR)
    # R3 WL-channel-fireable: leg-1 pc_separation (all run seeds; instrument is config-derived,
    # identical per agent) AND leg-2 primary run_populated_frac.
    pc_seps = [r["pc_separation"] for r in per_run if "pc_separation" in r]
    pc_separation = float(np.mean(pc_seps)) if pc_seps else 0.0
    r3_wl_fireable = bool(
        pc_separation >= PC_SEPARATION_FLOOR
        and primary.get("run_bank_populated_frac", 0.0) >= MIN_FRACTION
    )
    non_vacuity_met = bool(r1_consumption and r2_identity_fireable and r3_wl_fireable)

    # ---- DISCRIMINATION CRITERIA (primary arm; lift over control) ----
    c_id = bool(primary.get("mean_probe_acc_identity", 0.0) >= ID_PROBE_FLOOR
                and primary.get("pooled_n_identity_samples", 0) >= N_ID_SAMPLES_FLOOR)
    gr_lift = float(primary.get("mean_discrimination_margin", 0.0)
                    - control.get("mean_discrimination_margin", 0.0))
    c_gr = bool(gr_lift >= GR_LIFT)
    c_wl = bool(primary.get("mean_object_bound_wl_dissoc_fraction", 0.0) >= WL_FRACTION)
    c_anova = bool(primary.get("per_axis_drive_anova_f_max", 0.0) > F_CRIT_P01)

    # ---- OUTCOME + PER-CLAIM DIRECTION ----
    if not non_vacuity_met:
        outcome = "FAIL"
        readiness_route = "substrate_not_ready_requeue"
        dir_sd049 = "non_contributory"
        dir_sd015 = "non_contributory"
        overall_direction = "non_contributory"
        route_reason = "non_vacuity_unmet"
    else:
        overall_pass = bool(c_id and c_gr and c_wl and c_anova)
        outcome = "PASS" if overall_pass else "FAIL"
        # SD-015: encoder identity (C_ID) + z_resource-driven navigation (C_GR)
        sd015_hits = int(c_id) + int(c_gr)
        dir_sd015 = "supports" if sd015_hits == 2 else ("mixed" if sd015_hits == 1 else "weakens")
        # SD-049: full 4-criterion substrate validation
        sd049_hits = int(c_id) + int(c_anova) + int(c_gr) + int(c_wl)
        if sd049_hits == 4:
            dir_sd049 = "supports"
        elif sd049_hits == 0:
            dir_sd049 = "weakens"
        else:
            dir_sd049 = "mixed"
        if dir_sd049 == "supports" and dir_sd015 == "supports":
            overall_direction = "supports"
        elif dir_sd049 == "weakens" and dir_sd015 == "weakens":
            overall_direction = "weakens"
        else:
            overall_direction = "mixed"
        readiness_route = "closes_goal_pipeline_GAP2_StageB" if overall_pass else "residual_dv_open"
        route_reason = "all_criteria_met" if overall_pass else "partial_criteria"

    # Non-degeneracy scoring net (2026-06-11): the discrimination criteria get a fair,
    # non-vacuous test only when ALL three non-vacuity guards (R1/R2/R3) are met. A
    # non_contributory verdict before that is the V3-EXQ-514m / 693 class (the WL channel
    # was never written) -> mark non_degenerate=false so the indexer excludes the re-fail
    # from confidence/conflict scoring rather than scoring it as a genuine weakens.
    non_degenerate = bool(non_vacuity_met)
    degeneracy_reason = ("" if non_degenerate
                         else "discrimination criteria not non-vacuously testable: "
                              f"{route_reason} (R1={r1_consumption}, R2={r2_identity_fireable}, "
                              f"R3={r3_wl_fireable}); WL channel not fireable -> the V3-EXQ-693 "
                              "stale-harness signature; substrate_not_ready_requeue, never a weakens.")

    print(f"[{EXPERIMENT_TYPE}] non_vacuity_met={non_vacuity_met}"
          f" (R1={r1_consumption} R2={r2_identity_fireable} R3={r3_wl_fireable})"
          f" C_ID={c_id} C_GR={c_gr}(lift={gr_lift:.3f}) C_WL={c_wl} C_ANOVA={c_anova}"
          f" -> outcome={outcome} route={readiness_route}", flush=True)
    print(f"[{EXPERIMENT_TYPE}] per_claim SD-049={dir_sd049} SD-015={dir_sd015}"
          f" non_degenerate={non_degenerate}", flush=True)

    acceptance = {
        "non_vacuity_met": non_vacuity_met,
        "R1_consumption": r1_consumption,
        "R2_identity_probe_fireable": r2_identity_fireable,
        "R3_wl_channel_fireable": r3_wl_fireable,
        "pc_separation": pc_separation,
        "C_ID_identity_recovery": c_id,
        "C_GR_goal_resource_discrimination_margin": c_gr,
        "C_GR_lift_arm2_minus_arm0": gr_lift,
        "C_WL_object_bound_wanting_neq_liking": c_wl,
        "C_ANOVA_per_axis_drive": c_anova,
        "primary_arm": PRIMARY_ARM,
        "control_arm": CONTROL_ARM,
        "overall_pass": bool(non_vacuity_met and c_id and c_gr and c_wl and c_anova),
        "thresholds": {
            "p2_zgoal_gate": P2_ZGOAL_GATE, "consumption_floor": CONSUMPTION_FLOOR,
            "min_fraction": MIN_FRACTION, "id_probe_floor": ID_PROBE_FLOOR,
            "n_id_samples_floor": N_ID_SAMPLES_FLOOR, "gr_lift": GR_LIFT,
            "wl_fraction": WL_FRACTION, "anova_f_crit_p01": F_CRIT_P01,
            "pc_separation_floor": PC_SEPARATION_FLOOR, "min_scored_wl_steps": MIN_SCORED_STEPS,
            "drive_spread_floor": DRIVE_SPREAD_FLOOR,
        },
    }

    return {
        "outcome": outcome,
        "evidence_direction": overall_direction,
        "evidence_direction_per_claim": {"SD-049": dir_sd049, "SD-015": dir_sd015},
        "non_degenerate": non_degenerate,
        "degeneracy_reason": degeneracy_reason,
        "acceptance": acceptance,
        "arm_summary": arm_summary,
        "interpretation": {
            "readiness_route": readiness_route,
            "route_reason": route_reason,
            "wl_harness_port_note": "693a ports the 514r/s/t WL-scoring harness into the 693 fork: "
                                    "the liking target + L2 token-bind type are read from the "
                                    "AUTHORITATIVE info tag (info['sd049_consumed_type_tag_this_tick'], "
                                    "the 681-C4 fix captured pre-clear), and the WL scoring + bind "
                                    "block is gated on a genuine consumption event (consumed_tag is "
                                    "not None) so R3 can fire. 693 read obs_dict POST-CLEAR -> rtype "
                                    "None at the consumption tick -> scored_wl_steps=0 -> R3 vacuous. "
                                    "The R1 contact counter keeps the benefit-OR-consumed gate so "
                                    "ARM_0 (SD-049 OFF) preserves its contact rate.",
            "c5_fix_note": "goal_resource_r is measured as a NON-SATURATING identity-discrimination "
                           "margin (same-type minus different-type cosine of z_goal vs per-type "
                           "mean z_resource), so ARM_0 ~0 by absence of type structure -- the 514l "
                           "C5 defect (legacy cosine(z_goal,z_world) saturating ARM_0 ~1.0) is fixed. "
                           "The legacy saturating cosine is reported per arm as a diagnostic only.",
            "non_vacuity_note": "R1 consumption + R2 identity-probe-fireable + R3 WL-channel-fireable "
                                "(same-statistic 514n gate). Any unmet -> substrate_not_ready_requeue "
                                "(FAIL; both claims non_contributory), NEVER a false weakens.",
            "c_gr_watch_item": "PRE-REGISTERED (autopsy_V3-EXQ-693 Section 5 secondary flag, gated "
                               "behind R3 in 693 so never adjudicated): the C_GR discrimination-margin "
                               "lift was near-zero (ARM_2-ARM_0=0.0067; -0.0087 ARM_3). Now that the "
                               "WL channel scores, a margin-near-zero outcome (non_vacuity met, C_GR "
                               "fails while other DVs fire) is the LIKELY REAL SD-049 Phase-2 "
                               "conversion question -- read as a conversion-side substrate_ceiling / "
                               "substrate_conditional re-route candidate (route via /failure-autopsy), "
                               "NOT a standalone falsification. A WL-flat-only outcome with "
                               "non-vacuity met is the same SD-049 falsifier-branch re-route candidate.",
        },
        "per_run": per_run,
    }


def main(dry_run: bool = False) -> Dict[str, Any]:
    result = run_experiment(dry_run=dry_run)
    if dry_run:
        print(f"[{EXPERIMENT_TYPE}] dry-run complete; manifest not written.", flush=True)
        return {"outcome": result["outcome"], "manifest_path": None}

    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{timestamp}_v3"
    out_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments" / EXPERIMENT_TYPE
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": timestamp,
        "supersedes": SUPERSEDES,
        "outcome": result["outcome"],
        "evidence_direction": result["evidence_direction"],
        "evidence_direction_per_claim": result["evidence_direction_per_claim"],
        "sleep_driver_pattern": "N/A (waking goal-pipeline onboarding scheduler; no sleep loop)",
        "substrate": "scaffolded_sd054_onboarding full curriculum (Stage-0 -> Stage-0b -> P0 -> "
                     "Stage-H -> P1 -> P2; harm-pathway training ON; 603n config) + SD-049 Phase-2 "
                     "hybrid encoder (SD-015 z_resource) + SD-057 incentive token bank; 4-arm "
                     "substrate gradient OFF / 2-type / 3-type+novelty / 5-type. WL-scoring harness "
                     "ported from V3-EXQ-514r/s/t (info-tag consumed-type liking read).",
        "arms": [{"arm": a, "label": l, "sd049_on": s, "n_resource_types": n} for a, l, s, n in ARM_SPECS],
        "closes": "goal_pipeline:GAP-2 Stage B (on PASS); governance reconciles "
                  "pending_retest_after_substrate on SD-049 + SD-015 (and the dependent cohort) later.",
        "method_note": "Measurement re-issue of V3-EXQ-693 (NEW letter; same scientific question). "
                       "Per arm + seed: build an agent through the full onboarding curriculum on that "
                       "arm's SD-049-enabled (per arm) env, then frozen-policy measure the four SD-049 "
                       "DVs. goal_resource_r is the NON-SATURATING discrimination margin (514 C5 fix); "
                       "wanting!=liking is the SD-057 object-bound most-wanted-vs-info-tag-consumed "
                       "read (the 514t harness port that fixes the 693 R3 vacuity). Three non-vacuity "
                       "guards self-route substrate_not_ready_requeue, never a false weakens.",
        "anova_note": "scipy unavailable on runners; C_ANOVA gates on a pre-registered F-critical for a "
                      "one-way ANOVA across the per-axis homeostatic drives (df1 = n_axes-1; "
                      "df2 ~ inf over pooled per-step samples; F(0.01,2,inf)=4.605). F + dfs reported.",
        "scaffold_curriculum": {
            "stage0_budget": STAGE0_BUDGET, "stage0b_budget": STAGE0B_BUDGET, "p0_budget": P0_BUDGET,
            "hazard_stage_budget": HAZARD_STAGE_BUDGET, "p1_budget": P1_BUDGET, "p2_budget": P2_BUDGET,
            "behav_eval_episodes": BEHAV_EVAL_EPISODES, "train_steps": TRAIN_STEPS,
            "scaffold_train_harm_pathway": True, "scaffold_feed_harm_stream": True,
            "cue_recall_bridge_enabled_per_arm": {a: s for a, l, s, n in ARM_SPECS},
            "z_goal_enabled": True, "drive_weight": DRIVE_WEIGHT,
            "config_basis": "V3-EXQ-603n (the substrate-readiness run that flipped "
                            "scaffolded_sd054_onboarding ready=true 2026-06-11)",
        },
        "stage_plan": stage_plan(),
        "predecessor": "V3-EXQ-693 (SD-049 Phase-2 4-arm; FAIL/non_contributory -- stale 514l "
                       "WL-scoring harness -> R3 vacuous; this is the harness-port measurement "
                       "re-issue) / V3-EXQ-514t (working WL-scoring harness ported here).",
    }
    manifest.update(result)
    out_path = out_dir / f"{run_id}.json"
    out_path.write_text(json.dumps(manifest, indent=2))
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
        )
