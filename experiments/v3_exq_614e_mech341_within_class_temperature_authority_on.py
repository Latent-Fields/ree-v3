#!/opt/local/bin/python3
"""
V3-EXQ-614e -- MECH-341 within-class temperature CLEAN RETEST with the
modulatory-bias-selection-authority substrate ENABLED.

Supersedes V3-EXQ-614d (FAIL 2026-06-03, manifest
v3_exq_614d_mech341_within_class_temperature_committed_class_20260603T120121Z_v3.json).
614d FAILed for a SUBSTRATE-BUG reason, not a scientific one: committed-class
entropy was byte-identical across within-class temperature T=0.5/1.0/2.0
(1.056572 every arm), the within-class branch fired 3/3 seeds, paired-lift seed
count 0/3. Root cause: the scoring-layer diversity lever never reached
committed-action selection because the score-side modulatory contribution was
zeroed at E3.select -- the float32 catastrophic-cancellation bug in the
modulatory-bias-selection-authority gate (the gate reconstructed the modulatory
contribution as `scores - scores_raw`, which annihilates at large primary-score
magnitude). That bug is FIXED (V3-EXQ-643a PASS 2026-06-06; e3_selector.py now
tracks the combined modulatory contribution explicitly via `_modulatory_accum`),
and the substrate `modulatory-bias-selection-authority` is `status: implemented,
ready: true` in REE_assembly/evidence/planning/substrate_queue.json.

What 614e changes vs 614d (the ONLY difference is the substrate fix being armed)
-------------------------------------------------------------------------------
614e sets use_modulatory_selection_authority=True (gain=0.5) on every arm. This
arms BOTH authority sites in the now-fixed substrate:
  Site 1 (e3_selector.select): the combined modulatory contribution is rescaled
    to gain * raw_score_range and re-added (immune to the 614d cancellation bug).
  Site 2 (e3_score_diversity.stratified_select): class-representative scores are
    normalized to UNIT range before the stratified across-class softmax. This is
    the load-bearing fix for THIS experiment -- the within-class temperature
    changes the committed within-class representative, which changes its
    representative score, but under the legacy absolute-scale across-class softmax
    a large class-rep gap collapsed the committed-class distribution regardless of
    within-class T (the exact 614d C2 byte-identical signature). With Site-2
    unit-range normalization, the within-class representative shift now reaches
    the committed-action selection.
All other levers, env, seeds, phases, and the committed-class measurement are
IDENTICAL to 614d. 614e is the clean retest of the within-class temperature
lineage on a working substrate.

experiment_purpose = EVIDENCE. The substrate is validated, so this is a genuine
evidence test of MECH-341's within-class sub-axis (the diagnostic-measurability
question 614c/614d asked is now settled by the substrate fix). MECH-341 is a
v3_pending candidate (e3_scoring_preserves_trajectory_class_diversity); a PASS
here is supports-evidence that the Layer-B scoring diversity lever is load-bearing
on committed-action diversity and reaches committed selection.

Arms (4, on the SD-056-amended baseline + modulatory-authority ON)
-----------------------------------------------------------------
  ARM_0_LEGACY:   stratified_within_class_temperature = None   (legacy argmin within-class)
  ARM_1_T_0_5:    stratified_within_class_temperature = 0.5    (sharpened)
  ARM_2_T_1_0:    stratified_within_class_temperature = 1.0    (mid-T)
  ARM_3_T_2_0:    stratified_within_class_temperature = 2.0    (flatter)

All four arms run with use_modulatory_selection_authority=True (gain=0.5). The
within-class temperature is the ONLY swept axis; ARM_0_LEGACY is the
within-class-legacy baseline WITH authority on (the paired control for the C2
within-seed lift comparison).

All other levers held at the V3-EXQ-614b ARM_2 ALL_ON config:
  Layer A: SP-CEM ON (use_support_preserving_cem=True)
  Layer B: MECH-341 ON (use_e3_score_diversity=True; both sub-flavours ON;
           entropy_bias_scale=2.0)
  Layer C: MECH-313 ON (use_noise_floor=True; noise_floor_alpha=0.1)
  Layer D: V_s minimal stack ON (use_per_stream_vs=True + use_vs_rollout_gating=True)
  SD-056 amend: all 5 lever flags ON (multi-step contrastive h=5 + per-step
                output norm clamp ratio=2.0 + t=1 contrastive). The rollout
                output-norm clamp keeps the primary E3 scores bounded so the
                modulatory-authority gate does not see a 643-style 1e32 score
                magnitude. NO online SD-056 TRAINING happens here (agent in eval),
                so the 643 instability is not a concern, but the clamp is retained
                for lineage parity.
  use_differentiable_cem: NOT FLIPPED (default False; SD-055 safety note preserved)

Pre-registered acceptance criteria
----------------------------------
  C1 (substrate-operative non-vacuity -- the direct 614d-root-cause guard):
     across the positive-temperature arms, the within-class branch fires
     (mech341_n_within_class_sampled >= WITHIN_CLASS_FIRE_FLOOR) on a majority
     (>= 2/3) of seeds, AND the Site-2 stratified across-class authority
     normalization fires (mech341_n_authority_normalized > 0) on a majority of
     seeds, AND ARM_0_LEGACY committed_class_entropy is non-degenerate
     (cross-seed mean > C1_NONDEGEN_FLOOR). This confirms the substrate is
     operative and the 614d zeroing is fixed -- i.e. the test is non-vacuous.
  C2 (PRIMARY -- the substrate target): committed_class_entropy RISES with
     within-class temperature. Each positive-temperature arm {0.5, 1.0, 2.0}
     produces at least C2_MIN_LIFT_SEEDS_PER_ARM (= 1) per-seed PAIRED lift in
     committed_class_entropy_nats over ARM_0_LEGACY of at least C2_LIFT_MARGIN_NATS.
     Paired by seed index (same seed -> same env -> fair within-seed comparison).
     This is the substrate-target acceptance: the scoring-layer diversity lever
     must reach committed-action selection.
  C3 (substrate-readiness): all 4 arms produce frac_pre_ge2 > 0.3 on a majority
     (>= 2/3) of seeds. SP-CEM-layer candidate-pool diversity check.

Overall outcome
---------------
  PASS = C1 (substrate operative, test non-vacuous) AND C2 (per-arm within-class
         committed-class lift). C3 reported as context.
  FAIL = C1 holds but C2 fails (lever operative but adds no committed-class
         diversity -> within-class sub-axis not load-bearing -- a genuine
         negative for MECH-341), OR C1 fails (substrate not operative -> the test
         could not be run; route to /diagnose-errors, not an MECH-341 verdict).

Evidence direction (MECH-341)
-----------------------------
  PASS (C1 + C2): supports -- within-class proportional sampling lever reaches
    committed-action selection and lifts committed-class diversity. MECH-341
    within-class sub-axis is load-bearing under the working substrate.
  C1 holds, C2 fails: weakens -- lever operative + authority active but no
    committed-class lift; the within-class sub-axis adds no marginal committed
    diversity over legacy argmin.
  C1 fails: non_contributory -- substrate not operative in this run; the within-
    class lever could not express itself (route to /diagnose-errors on the
    authority / stratified-select wiring, NOT an MECH-341 falsification).

Claims: [MECH-341] (single claim; the within-class Layer-B sub-axis is the only
varied lever). experiment_purpose=evidence.

Phases
------
P0 (30 ep, instrumentation OFF): encoder warmup.
P1 (60 ep, instrumentation ON): behavioural measurement window. Matches the
V3-EXQ-614b / 614c / 614d P1 budget for direct manifest comparability.

Budget: 4 arms x 3 seeds x 90 ep x 200 steps = 216k steps total.
Estimated ~3-4 h. The C2 paired-lift comparison is within-run (same seed, same
machine), so no cross-machine reference is required -> machine_affinity "any".

See REE_assembly/docs/architecture/modulatory_bias_selection_authority.md
(V3-EXQ-643a fix section), REE_assembly/docs/architecture/
mech_341_e3_score_diversity_preservation.md ("Amend 2026-06-01" section),
REE_assembly/evidence/planning/substrate_queue.json
(modulatory-bias-selection-authority + MECH-341 entries),
REE_assembly/evidence/planning/failure_autopsy_V3-EXQ-643_2026-06-06.{md,json},
REE_assembly/evidence/planning/behavioral_diversity_isolation_plan.md GAP-B.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from experiment_protocol import emit_outcome
from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig
from experiments.pack_writer import write_flat_manifest  # noqa: E402


EXPERIMENT_TYPE = "v3_exq_614e_mech341_within_class_temperature_authority_on"
QUEUE_ID = "V3-EXQ-614e"
SUPERSEDES = "V3-EXQ-614d"
CLAIM_IDS: List[str] = ["MECH-341"]
EXPERIMENT_PURPOSE = "evidence"

# V3-EXQ-614e sweep axis: within-class temperature in {None, 0.5, 1.0, 2.0}.
# None = legacy argmin within-class (the paired control arm). Set on
# E3ScoreDiversityConfig.stratified_within_class_temperature via
# REEConfig.e3_diversity_stratified_within_class_temperature. See MECH-341
# amend 2026-06-01.
WITHIN_CLASS_T_BY_ARM: Dict[str, Optional[float]] = {
    "ARM_0_LEGACY": None,
    "ARM_1_T_0_5": 0.5,
    "ARM_2_T_1_0": 1.0,
    "ARM_3_T_2_0": 2.0,
}

# modulatory-bias-selection-authority (V3-EXQ-643a-fixed substrate). ON for ALL
# arms in 614e -- this is the difference vs 614d. gain=0.5 keeps modulatory
# signals competitive in near-ties but subdominant when the primary harm/goal
# gap exceeds gain * raw_score_range.
USE_MODULATORY_SELECTION_AUTHORITY = True
MODULATORY_AUTHORITY_GAIN = 0.5

# C2 within-class committed-class lift: each positive-temperature arm must
# produce at least C2_MIN_LIFT_SEEDS_PER_ARM per-seed PAIRED lift in
# committed_class_entropy_nats over ARM_0_LEGACY of at least C2_LIFT_MARGIN_NATS.
# Margin chosen small relative to committed-class entropy magnitudes (~0.5-1.2
# nats) but above per-seed measurement noise (matches the 614d margin).
C2_LIFT_MARGIN_NATS = 0.05
C2_MIN_LIFT_SEEDS_PER_ARM = 1   # >= 1 paired-lift seed PER positive-T arm

# C1 substrate-operative non-vacuity: ARM_0_LEGACY committed_class_entropy
# cross-seed mean must clear this floor (committed selection not collapsed to a
# single class).
C1_NONDEGEN_FLOOR = 0.10

# Within-class firing gate (C1 input). A positive-temperature arm whose
# within-class branch accumulated at least this many samples on a majority of
# seeds is "branch-active".
WITHIN_CLASS_FIRE_FLOOR = 10

# SD-056 amend lever defaults applied uniformly across all 4 arms.
SD056_MULTISTEP_CONTRASTIVE = True
SD056_CONTRASTIVE_HORIZON = 5
SD056_OUTPUT_NORM_CLAMP = True
SD056_OUTPUT_NORM_CLAMP_RATIO = 2.0
SD056_T1_CONTRASTIVE_ENABLED = True
SD056_T1_CONTRASTIVE_WEIGHT = 0.01

SEEDS = [42, 43, 44]
P0_WARMUP_EPISODES = 30
P1_MEASUREMENT_EPISODES = 60
STEPS_PER_EPISODE = 200

DRY_RUN_SEEDS = [42]
DRY_RUN_P0 = 2
DRY_RUN_P1 = 2
DRY_RUN_STEPS = 30

# Pre-registered behavioural thresholds (preserved from the 614 lineage so the
# P1 per-tick semantics remain comparable across the cluster).
RUNG1_ENTROPY_THRESHOLD = 0.3
RUNG1_MIN_CLASSES = 2
MIN_SEEDS_PER_ARM_FOR_PASS = 2  # of 3 (used for C1 / C3 majority checks)

PRE_GE2_FRAC_GATE = 0.5

# MECH-341 sub-flavour scale used in the entropy-ON arms (matches 614d / 614b).
MECH341_ENTROPY_BIAS_SCALE = 2.0

# V_s (D) thresholds (minimal stack).
VS_SNAPSHOT_REFRESH_THRESHOLD = 0.5
VS_E1_THRESHOLD = 0.4


# IDENTICAL to V3-EXQ-611 / 614c / 614d for direct manifest comparability.
ENV_KWARGS = dict(
    size=12,
    num_hazards=4,
    num_resources=5,
    hazard_harm=0.05,
    env_drift_interval=5,
    env_drift_prob=0.1,
    proximity_harm_scale=0.1,
    proximity_benefit_scale=0.05,
    proximity_approach_threshold=0.2,
    hazard_field_decay=0.5,
    resource_respawn_on_consume=True,
    toroidal=False,
    harm_history_len=10,
    reef_enabled=True,
    n_reef_patches=3,
    reef_patch_radius=2,
    hazard_food_attraction=0.7,
    reef_bipartite_layout=True,
    reef_bipartite_axis="horizontal",
    reef_bipartite_agent_band_radius=1,
)


ARMS: List[Dict[str, Any]] = [
    {
        "arm_id": "ARM_0_LEGACY",
        "label": "within_class_legacy_argmin",
        "within_class_temperature": None,
        "substrate_axes": {
            "A_sp_cem": True,
            "B_mech341": True,
            "C_noise_floor": True,
            "D_vs": True,
        },
    },
    {
        "arm_id": "ARM_1_T_0_5",
        "label": "within_class_T_0_5_sharpened",
        "within_class_temperature": 0.5,
        "substrate_axes": {
            "A_sp_cem": True,
            "B_mech341": True,
            "C_noise_floor": True,
            "D_vs": True,
        },
    },
    {
        "arm_id": "ARM_2_T_1_0",
        "label": "within_class_T_1_0_mid",
        "within_class_temperature": 1.0,
        "substrate_axes": {
            "A_sp_cem": True,
            "B_mech341": True,
            "C_noise_floor": True,
            "D_vs": True,
        },
    },
    {
        "arm_id": "ARM_3_T_2_0",
        "label": "within_class_T_2_0_flatter",
        "within_class_temperature": 2.0,
        "substrate_axes": {
            "A_sp_cem": True,
            "B_mech341": True,
            "C_noise_floor": True,
            "D_vs": True,
        },
    },
]


def _make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(seed=seed, **ENV_KWARGS)


def _make_agent(
    env: CausalGridWorldV2,
    axes: Dict[str, bool],
    within_class_temperature: Optional[float] = None,
) -> REEAgent:
    """Build a REEAgent with the SD-056-amended 4-substrate baseline AND the
    modulatory-bias-selection-authority substrate ON.

    Config is bit-identical to V3-EXQ-614d _make_agent EXCEPT
    use_modulatory_selection_authority=True (+ gain) -- the V3-EXQ-643a-fixed
    substrate that lets the within-class temperature lever reach committed-action
    selection (Site-2 stratified across-class unit-range normalization).
    """
    a_on = bool(axes["A_sp_cem"])
    b_on = bool(axes["B_mech341"])
    c_on = bool(axes["C_noise_floor"])
    d_on = bool(axes["D_vs"])

    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=32,
        world_dim=32,
        alpha_world=0.9,
        alpha_self=0.3,
        use_harm_stream=True,
        z_harm_dim=32,
        use_affective_harm_stream=True,
        z_harm_a_dim=16,
        harm_history_len=10,
        z_goal_enabled=True,
        goal_weight=0.5,
        drive_weight=2.0,
        e1_goal_conditioned=True,
        use_resource_proximity_head=True,
        resource_proximity_weight=0.5,
        benefit_eval_enabled=True,
        benefit_weight=1.0,
        # A (SP-CEM)
        use_support_preserving_cem=a_on,
        support_preserving_stratified_elites=a_on,
        support_preserving_ao_std_floor=(0.2 if a_on else 0.0),
        support_preserving_min_first_action_classes=2,
        # B (MECH-341)
        use_e3_score_diversity=b_on,
        use_e3_diversity_entropy_bonus=True,    # consulted only when master ON
        use_e3_diversity_stratified_select=True,
        e3_diversity_entropy_bias_scale=MECH341_ENTROPY_BIAS_SCALE,
        # MECH-341 amend 2026-06-01 sweep axis: within-class proportional
        # sampling temperature. None = legacy argmin within-class.
        e3_diversity_stratified_within_class_temperature=within_class_temperature,
        # modulatory-bias-selection-authority (V3-EXQ-643a-fixed). ON for all
        # arms in 614e -- arms Site 1 (additive authority) AND Site 2 (stratified
        # across-class unit-range normalization). Site 2 is what lets the
        # within-class temperature reach committed-action selection (the 614d
        # C2 fix).
        use_modulatory_selection_authority=USE_MODULATORY_SELECTION_AUTHORITY,
        modulatory_authority_gain=MODULATORY_AUTHORITY_GAIN,
        # C (MECH-313)
        use_noise_floor=c_on,
        noise_floor_alpha=0.1,
        # D (V_s minimal)
        use_per_stream_vs=d_on,
        use_vs_rollout_gating=d_on,
        vs_gate_snapshot_refresh_threshold=VS_SNAPSHOT_REFRESH_THRESHOLD,
        vs_gate_e1_threshold=VS_E1_THRESHOLD,
        # SD-056 amend (uniform across ALL arms; not an isolation axis).
        e2_action_contrastive_enabled=SD056_T1_CONTRASTIVE_ENABLED,
        e2_action_contrastive_weight=SD056_T1_CONTRASTIVE_WEIGHT,
        e2_action_contrastive_multistep_enabled=SD056_MULTISTEP_CONTRASTIVE,
        e2_action_contrastive_horizon=SD056_CONTRASTIVE_HORIZON,
        e2_rollout_output_norm_clamp_enabled=SD056_OUTPUT_NORM_CLAMP,
        e2_rollout_output_norm_clamp_ratio=SD056_OUTPUT_NORM_CLAMP_RATIO,
    )
    return REEAgent(cfg)


# ---------------------------------------------------------------------------
# Per-tick measurement helpers (mirror EXQ-608 / EXQ-611 / 614c / 614d)
# ---------------------------------------------------------------------------


def _trajectory_first_action_class(traj) -> int:
    return int(
        traj.actions[:, 0, :].argmax(dim=-1).detach().reshape(-1)[0].item()
    )


def _per_class_score_stats(
    candidates: List, last_scores: Optional[torch.Tensor]
) -> Tuple[Dict[int, Dict[str, float]], Optional[int], Optional[float], bool]:
    """Score-layer per-class stats. The SELECTED class here is the score-layer
    argmin of last_scores (which, with authority ON, is the POST-authority
    score). Retained as informational context; the temperature-sensitive C2
    signal is the COMMITTED-action class distribution measured separately below.
    """
    if (
        not candidates
        or len(candidates) < 2
        or last_scores is None
        or last_scores.numel() != len(candidates)
    ):
        return {}, None, None, False

    scores_t = last_scores.detach().reshape(-1).float()
    per_class_scores: Dict[int, List[float]] = {}
    classes_per_cand: List[int] = []
    for i, traj in enumerate(candidates):
        cls = _trajectory_first_action_class(traj)
        classes_per_cand.append(cls)
        per_class_scores.setdefault(cls, []).append(float(scores_t[i].item()))

    per_class: Dict[int, Dict[str, float]] = {}
    for cls, vals in per_class_scores.items():
        n = len(vals)
        mean_v = sum(vals) / n
        if n == 1:
            std_v = 0.0
        else:
            var_v = sum((v - mean_v) ** 2 for v in vals) / n
            std_v = math.sqrt(var_v)
        per_class[cls] = {
            "n": int(n),
            "score_mean": float(mean_v),
            "score_std": float(std_v),
        }

    sel_idx = int(scores_t.argmin().item())
    selected_class = int(classes_per_cand[sel_idx])

    sorted_means = sorted(m["score_mean"] for m in per_class.values())
    if len(sorted_means) >= 2:
        top2_gap = float(sorted_means[1] - sorted_means[0])
    else:
        top2_gap = None

    return per_class, selected_class, top2_gap, True


def _obs_harm(obs_dict):
    h = obs_dict.get("harm_obs")
    return h.float().unsqueeze(0) if h is not None else None


def _obs_harm_a(obs_dict):
    h = obs_dict.get("harm_obs_a")
    return h.float().unsqueeze(0) if h is not None else None


def _obs_harm_history(obs_dict):
    h = obs_dict.get("harm_history")
    return h.float().unsqueeze(0) if h is not None else None


def _entropy_from_counts(counts: Dict[int, int]) -> float:
    n = sum(counts.values())
    if n <= 0:
        return 0.0
    h = 0.0
    for c in counts.values():
        if c <= 0:
            continue
        p = c / n
        h -= p * math.log(p)
    return float(h)


def _run_seed_arm(
    arm: Dict[str, Any],
    seed: int,
    p0_episodes: int,
    p1_episodes: int,
    steps_per_episode: int,
) -> Dict[str, Any]:
    torch.manual_seed(seed)
    env = _make_env(seed)
    agent = _make_agent(
        env,
        arm["substrate_axes"],
        within_class_temperature=arm.get("within_class_temperature"),
    )
    agent.eval()

    total_train_eps = p0_episodes + p1_episodes
    error_note: Optional[str] = None
    n_p0_ticks = 0
    n_p1_ticks = 0
    n_p1_logged = 0
    n_p1_pre_ge2 = 0
    n_p1_pre_eq1 = 0
    selected_classes_p1: Dict[int, int] = {}
    committed_classes_p1: Dict[int, int] = {}
    top2_gaps_pre_ge2: List[float] = []

    # MECH-341 within-class firing diagnostics, accumulated over the P1 window.
    n_within_class_sampled_total = 0
    n_stratified_fired_total = 0
    last_within_class_temperature = 0.0

    # Site-2 stratified across-class authority normalization firing (the V3-EXQ-
    # 643a fix that makes the within-class temperature reach committed selection).
    n_authority_normalized_total = 0
    last_rep_score_range = 0.0

    # Site-1 additive authority diagnostics (informational; read from E3 each
    # P1 tick from agent.e3.last_score_diagnostics).
    n_p1_authority_active = 0
    authority_scale_factors: List[float] = []
    authority_ranges: List[float] = []
    raw_score_ranges: List[float] = []

    for ep in range(total_train_eps):
        is_p1 = ep >= p0_episodes
        phase_label = "P1" if is_p1 else "P0"

        _, obs_dict = env.reset()
        agent.reset()

        z_self_prev: Optional[torch.Tensor] = None
        action_prev: Optional[torch.Tensor] = None

        for _step in range(steps_per_episode):
            body = obs_dict["body_state"].float()
            world = obs_dict["world_state"].float()
            if body.dim() == 1:
                body = body.unsqueeze(0)
            if world.dim() == 1:
                world = world.unsqueeze(0)

            latent = agent.sense(
                obs_body=body, obs_world=world,
                obs_harm=_obs_harm(obs_dict),
                obs_harm_a=_obs_harm_a(obs_dict),
                obs_harm_history=_obs_harm_history(obs_dict),
            )

            if z_self_prev is not None and action_prev is not None:
                agent.record_transition(
                    z_self_prev, action_prev, latent.z_self.detach()
                )

            ticks = agent.clock.advance()
            wdim = latent.z_world.shape[-1]
            e1_prior = (
                agent._e1_tick(latent) if ticks.get("e1_tick", False)
                else torch.zeros(1, wdim, device=agent.device)
            )
            candidates = agent.generate_trajectories(latent, e1_prior, ticks)

            pre_e3_classes: List[int] = []
            if is_p1 and candidates:
                pre_e3_classes = sorted({
                    _trajectory_first_action_class(t) for t in candidates
                })

            action = agent.select_action(candidates, ticks)
            if not torch.isfinite(action).all():
                if error_note is None:
                    error_note = (
                        f"non-finite action at arm={arm['arm_id']} seed={seed} "
                        f"phase={phase_label} ep={ep} step={_step}"
                    )
                break

            if is_p1:
                last_scores = getattr(agent.e3, "last_scores", None)
                per_class, sel_class, top2_gap, logged = _per_class_score_stats(
                    candidates, last_scores
                )
                committed_class = int(action[0].argmax().item())

                pre_count = len(pre_e3_classes)
                if pre_count >= 2:
                    n_p1_pre_ge2 += 1
                elif pre_count == 1:
                    n_p1_pre_eq1 += 1

                if logged:
                    n_p1_logged += 1
                    if sel_class is not None:
                        selected_classes_p1[sel_class] = (
                            selected_classes_p1.get(sel_class, 0) + 1
                        )
                    if pre_count >= 2 and top2_gap is not None:
                        top2_gaps_pre_ge2.append(top2_gap)

                committed_classes_p1[committed_class] = (
                    committed_classes_p1.get(committed_class, 0) + 1
                )
                n_p1_ticks += 1

                # Site-1 additive-authority diagnostics (per-tick).
                diag = getattr(agent.e3, "last_score_diagnostics", None)
                if isinstance(diag, dict):
                    if bool(diag.get("modulatory_authority_active", False)):
                        n_p1_authority_active += 1
                        sf = float(diag.get("modulatory_authority_scale_factor", 0.0))
                        authority_scale_factors.append(sf)
                    authority_ranges.append(
                        float(diag.get("modulatory_authority_range", 0.0))
                    )
                    raw_score_ranges.append(
                        float(diag.get("e3_raw_score_range_mean", 0.0))
                    )
            else:
                n_p0_ticks += 1

            _, _harm_signal, done, info, obs_dict = env.step(action)

            if agent.goal_state is not None:
                benefit_exposure = float(info.get("benefit_exposure", 0.0))
                energy = float(body[0, 3].item())
                drive_level = max(0.0, 1.0 - energy)
                agent.update_z_goal(
                    benefit_exposure=benefit_exposure,
                    drive_level=drive_level,
                )

            z_self_prev = latent.z_self.detach()
            action_prev = action.detach()

            if done:
                break

        # MECH-341 within-class + Site-2 authority diagnostics: read AFTER the
        # inner step loop (per-episode E3ScoreDiversity counters reflect this
        # episode's cumulative firing) and BEFORE the next agent.reset() clears
        # them. Accumulate only over the P1 measurement window.
        if is_p1 and getattr(agent, "score_diversity", None) is not None:
            sd_state = agent.score_diversity.get_state()
            n_within_class_sampled_total += int(
                sd_state.get("mech341_n_within_class_sampled", 0)
            )
            n_stratified_fired_total += int(
                sd_state.get("mech341_n_stratified_fired", 0)
            )
            n_authority_normalized_total += int(
                sd_state.get("mech341_n_authority_normalized", 0)
            )
            ep_within_temp = float(
                sd_state.get("mech341_last_within_class_temperature", 0.0)
            )
            if ep_within_temp != 0.0:
                last_within_class_temperature = ep_within_temp
            ep_rep_range = float(sd_state.get("mech341_last_rep_score_range", 0.0))
            if ep_rep_range != 0.0:
                last_rep_score_range = ep_rep_range

        if (ep + 1) % 10 == 0 or (ep + 1) == total_train_eps:
            print(
                f"  [train] arm={arm['arm_id']} seed={seed} phase={phase_label} "
                f"ep {ep + 1}/{total_train_eps}",
                flush=True,
            )

        if error_note is not None:
            break

    if n_p1_ticks > 0:
        frac_pre_ge2 = float(n_p1_pre_ge2 / n_p1_ticks)
        authority_active_frac = float(n_p1_authority_active / n_p1_ticks)
    else:
        frac_pre_ge2 = 0.0
        authority_active_frac = 0.0

    if top2_gaps_pre_ge2:
        mean_top2_gap = float(sum(top2_gaps_pre_ge2) / len(top2_gaps_pre_ge2))
    else:
        mean_top2_gap = None

    mean_authority_scale = (
        float(sum(authority_scale_factors) / len(authority_scale_factors))
        if authority_scale_factors else 0.0
    )
    mean_authority_range = (
        float(sum(authority_ranges) / len(authority_ranges))
        if authority_ranges else 0.0
    )
    mean_raw_score_range = (
        float(sum(raw_score_ranges) / len(raw_score_ranges))
        if raw_score_ranges else 0.0
    )

    n_selected_classes = len(selected_classes_p1)
    n_committed_classes = len(committed_classes_p1)
    selected_class_entropy = _entropy_from_counts(selected_classes_p1)
    committed_class_entropy = _entropy_from_counts(committed_classes_p1)

    # Per-seed substrate-readiness flag (C3 input): pre-E3 pool diversity gate.
    seed_substrate_ready = bool(frac_pre_ge2 > 0.3)

    # Per-seed within-class branch activity (C1 input).
    within_class_branch_active = bool(
        n_within_class_sampled_total >= WITHIN_CLASS_FIRE_FLOOR
    )
    # Per-seed Site-2 stratified authority normalization firing (C1 input -- the
    # direct evidence the 614d zeroing is fixed and the lever reaches committed
    # selection).
    authority_normalization_fired = bool(n_authority_normalized_total > 0)

    return {
        "arm_id": arm["arm_id"],
        "seed": int(seed),
        "within_class_temperature": arm.get("within_class_temperature"),
        "p0_episodes_run": int(min(p0_episodes, total_train_eps)),
        "p1_episodes_run": int(max(0, total_train_eps - p0_episodes)),
        "n_p0_ticks": int(n_p0_ticks),
        "n_p1_ticks": int(n_p1_ticks),
        "n_p1_logged": int(n_p1_logged),
        "n_p1_pre_ge2": int(n_p1_pre_ge2),
        "n_p1_pre_eq1": int(n_p1_pre_eq1),
        "frac_pre_ge2": round(frac_pre_ge2, 6),
        "mean_top2_class_gap": (
            None if mean_top2_gap is None else round(mean_top2_gap, 6)
        ),
        "selected_classes_p1_counts": {
            str(k): int(v) for k, v in sorted(selected_classes_p1.items())
        },
        "committed_classes_p1_counts": {
            str(k): int(v) for k, v in sorted(committed_classes_p1.items())
        },
        "n_unique_selected_classes": int(n_selected_classes),
        "n_unique_committed_classes": int(n_committed_classes),
        # Score-layer argmin entropy (post-authority; informational):
        "selected_class_entropy_nats": round(selected_class_entropy, 6),
        # C2 metric (committed action; temperature-sensitive; the lift signal):
        "committed_class_entropy_nats": round(committed_class_entropy, 6),
        # MECH-341 within-class firing diagnostics (per-seed accumulated):
        "mech341_n_within_class_sampled": int(n_within_class_sampled_total),
        "mech341_n_stratified_fired": int(n_stratified_fired_total),
        "mech341_last_within_class_temperature": round(
            last_within_class_temperature, 6
        ),
        # Site-2 stratified-authority normalization diagnostics (per-seed):
        "mech341_n_authority_normalized": int(n_authority_normalized_total),
        "mech341_last_rep_score_range": round(last_rep_score_range, 6),
        # Site-1 additive-authority diagnostics (per-seed; informational):
        "authority_active_frac": round(authority_active_frac, 6),
        "mean_authority_scale_factor": round(mean_authority_scale, 6),
        "mean_authority_range": round(mean_authority_range, 6),
        "mean_raw_score_range": round(mean_raw_score_range, 6),
        # Per-seed flags:
        "within_class_branch_active": within_class_branch_active,
        "authority_normalization_fired": authority_normalization_fired,
        "seed_substrate_ready": seed_substrate_ready,
        "error_note": error_note,
    }


def _interpret_arm(seed_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    completed = [r for r in seed_rows if r["error_note"] is None]
    n_seeds_completed = len(completed)

    sel_entropies = [r["selected_class_entropy_nats"] for r in completed]
    com_entropies = [r["committed_class_entropy_nats"] for r in completed]
    mean_selected_entropy = (
        sum(sel_entropies) / len(sel_entropies) if sel_entropies else 0.0
    )
    mean_committed_entropy = (
        sum(com_entropies) / len(com_entropies) if com_entropies else 0.0
    )

    n_substrate_ready = sum(1 for r in completed if r["seed_substrate_ready"])
    n_within_class_active = sum(
        1 for r in completed if r["within_class_branch_active"]
    )
    n_authority_norm_fired = sum(
        1 for r in completed if r["authority_normalization_fired"]
    )
    within_class_samples = [
        r["mech341_n_within_class_sampled"] for r in completed
    ]

    return {
        "n_seeds_completed": int(n_seeds_completed),
        "mean_selected_class_entropy_nats": round(mean_selected_entropy, 6),
        "mean_committed_class_entropy_nats": round(mean_committed_entropy, 6),
        "n_seeds_substrate_ready": int(n_substrate_ready),
        "majority_substrate_ready": bool(
            n_substrate_ready >= MIN_SEEDS_PER_ARM_FOR_PASS
        ),
        "n_seeds_within_class_active": int(n_within_class_active),
        "majority_within_class_active": bool(
            n_within_class_active >= MIN_SEEDS_PER_ARM_FOR_PASS
        ),
        "n_seeds_authority_norm_fired": int(n_authority_norm_fired),
        "majority_authority_norm_fired": bool(
            n_authority_norm_fired >= MIN_SEEDS_PER_ARM_FOR_PASS
        ),
        "within_class_samples_per_seed": [int(s) for s in within_class_samples],
    }


def _classify_outcome(
    c1: bool, c2: bool, c3: bool
) -> Tuple[str, str, str]:
    """V3-EXQ-614e outcome map.

    C1: substrate-operative non-vacuity (within-class branch fires + Site-2
        authority normalization fires on a majority of seeds in the positive
        arms AND ARM_0 committed entropy non-degenerate).
    C2: per-arm within-class committed-class lift (each positive-T arm has
        >= C2_MIN_LIFT_SEEDS_PER_ARM paired-lift seeds over ARM_0_LEGACY).
    C3: substrate-readiness (frac_pre_ge2 > 0.3 majority, all arms).

    PASS = C1 AND C2.
    """
    if c1 and c2:
        return (
            "PASS", "supports",
            "PASS_C1_C2_within_class_lever_reaches_committed_selection_and_lifts_diversity",
        )
    if c1 and (not c2):
        return (
            "FAIL", "weakens",
            "FAIL_C1_holds_C2_fails_lever_operative_but_no_committed_class_lift",
        )
    # not c1
    return (
        "FAIL", "non_contributory",
        "FAIL_C1_substrate_not_operative_route_to_diagnose_errors",
    )


def run_experiment(
    seeds: List[int],
    p0_episodes: int,
    p1_episodes: int,
    steps_per_episode: int,
    dry_run: bool,
) -> Dict[str, Any]:
    arms_out: List[Dict[str, Any]] = []
    for arm in ARMS:
        print(
            f"Arm {arm['arm_id']} ({arm['label']}) "
            f"(P0={p0_episodes} ep, P1={p1_episodes} ep, "
            f"steps_per_episode={steps_per_episode}, "
            f"authority=ON gain={MODULATORY_AUTHORITY_GAIN}, dry_run={dry_run})",
            flush=True,
        )
        seed_rows: List[Dict[str, Any]] = []
        for s in seeds:
            print(f"Seed {s} Condition {arm['label']}", flush=True)
            row = _run_seed_arm(arm, s, p0_episodes, p1_episodes, steps_per_episode)
            seed_rows.append(row)
            verdict = "PASS" if row["error_note"] is None else "FAIL"
            print(f"verdict: {verdict}", flush=True)
        cross = _interpret_arm(seed_rows)
        arms_out.append({
            "arm_id": arm["arm_id"],
            "label": arm["label"],
            "within_class_temperature": arm.get("within_class_temperature"),
            "substrate_axes": arm["substrate_axes"],
            "per_seed_results": seed_rows,
            "cross_seed_interpretation": cross,
        })

    # ----- Cross-arm acceptance criteria (V3-EXQ-614e) -----
    by_id = {a["arm_id"]: a for a in arms_out}
    arm_legacy = by_id["ARM_0_LEGACY"]
    swept_arms = [by_id["ARM_1_T_0_5"], by_id["ARM_2_T_1_0"], by_id["ARM_3_T_2_0"]]

    # --- C2 (PRIMARY): per-seed PAIRED committed-class lift over ARM_0_LEGACY,
    #     >= C2_MIN_LIFT_SEEDS_PER_ARM seeds for EACH positive-temperature arm.
    legacy_committed_by_seed: Dict[int, float] = {
        int(r["seed"]): r["committed_class_entropy_nats"]
        for r in arm_legacy["per_seed_results"]
        if r["error_note"] is None
    }

    def _arm_paired_lift_count(arm) -> int:
        n = 0
        for r in arm["per_seed_results"]:
            if r["error_note"] is not None:
                continue
            seed = int(r["seed"])
            if seed not in legacy_committed_by_seed:
                continue
            lift = r["committed_class_entropy_nats"] - legacy_committed_by_seed[seed]
            if lift >= C2_LIFT_MARGIN_NATS:
                n += 1
        return n

    c2_per_arm_lift = {a["arm_id"]: _arm_paired_lift_count(a) for a in swept_arms}
    # Per-arm requirement: EACH positive-T arm must clear the per-arm seed floor.
    c2_holds = all(
        n >= C2_MIN_LIFT_SEEDS_PER_ARM for n in c2_per_arm_lift.values()
    )

    c2_per_arm_committed_mean = {
        a["arm_id"]: a["cross_seed_interpretation"]["mean_committed_class_entropy_nats"]
        for a in swept_arms
    }
    legacy_committed_mean = arm_legacy["cross_seed_interpretation"][
        "mean_committed_class_entropy_nats"
    ]

    # --- C1 (substrate-operative non-vacuity): within-class branch fires AND
    #     Site-2 authority normalization fires on a majority of seeds in the
    #     positive arms, AND ARM_0_LEGACY committed entropy non-degenerate.
    swept_branch_active_majority = all(
        a["cross_seed_interpretation"]["majority_within_class_active"]
        for a in swept_arms
    )
    swept_authority_norm_majority = all(
        a["cross_seed_interpretation"]["majority_authority_norm_fired"]
        for a in swept_arms
    )
    legacy_nondegen = bool(legacy_committed_mean > C1_NONDEGEN_FLOOR)
    c1_holds = bool(
        swept_branch_active_majority
        and swept_authority_norm_majority
        and legacy_nondegen
    )

    # --- C3 (substrate-readiness): all arms frac_pre_ge2 > 0.3 majority.
    c3_per_arm_ready = {
        a["arm_id"]: a["cross_seed_interpretation"]["n_seeds_substrate_ready"]
        for a in arms_out
    }
    c3_holds = all(
        n >= MIN_SEEDS_PER_ARM_FOR_PASS for n in c3_per_arm_ready.values()
    )

    # Authority-active fraction context (Site-1 additive authority, all arms).
    authority_active_frac_by_arm = {
        a["arm_id"]: round(
            sum(
                r["authority_active_frac"]
                for r in a["per_seed_results"] if r["error_note"] is None
            ) / max(1, sum(1 for r in a["per_seed_results"] if r["error_note"] is None)),
            6,
        )
        for a in arms_out
    }
    authority_norm_fired_per_arm = {
        a["arm_id"]: a["cross_seed_interpretation"]["n_seeds_authority_norm_fired"]
        for a in arms_out
    }
    within_class_active_per_arm = {
        a["arm_id"]: a["cross_seed_interpretation"]["n_seeds_within_class_active"]
        for a in swept_arms
    }

    (
        outcome_label, mech341_direction, interpretation_label,
    ) = _classify_outcome(c1_holds, c2_holds, c3_holds)
    overall_direction = mech341_direction

    total_seeds = len(ARMS) * len(seeds)
    total_completed = sum(
        a["cross_seed_interpretation"]["n_seeds_completed"] for a in arms_out
    )

    return {
        "outcome": outcome_label,
        "overall_direction": overall_direction,
        "evidence_direction_per_claim": {
            "MECH-341": mech341_direction,
        },
        "interpretation_label": interpretation_label,
        "seeds": seeds,
        "n_arms": len(arms_out),
        "total_seeds_attempted": int(total_seeds),
        "total_seeds_completed": int(total_completed),
        "p0_episodes": int(p0_episodes),
        "p1_episodes": int(p1_episodes),
        "steps_per_episode": int(steps_per_episode),
        "decision_rule_thresholds": {
            "rung1_entropy_threshold": float(RUNG1_ENTROPY_THRESHOLD),
            "rung1_min_classes": int(RUNG1_MIN_CLASSES),
            "min_seeds_per_arm_for_pass": int(MIN_SEEDS_PER_ARM_FOR_PASS),
            "pre_ge2_frac_gate": float(PRE_GE2_FRAC_GATE),
            "mech341_entropy_bias_scale": float(MECH341_ENTROPY_BIAS_SCALE),
            "vs_snapshot_refresh_threshold": float(VS_SNAPSHOT_REFRESH_THRESHOLD),
            "vs_e1_threshold": float(VS_E1_THRESHOLD),
            "c2_lift_margin_nats": float(C2_LIFT_MARGIN_NATS),
            "c2_min_lift_seeds_per_arm": int(C2_MIN_LIFT_SEEDS_PER_ARM),
            "c1_nondegen_floor": float(C1_NONDEGEN_FLOOR),
            "within_class_fire_floor": int(WITHIN_CLASS_FIRE_FLOOR),
            "use_modulatory_selection_authority": bool(
                USE_MODULATORY_SELECTION_AUTHORITY
            ),
            "modulatory_authority_gain": float(MODULATORY_AUTHORITY_GAIN),
        },
        "acceptance_criteria": {
            "C1_substrate_operative_non_vacuity": c1_holds,
            "C1_swept_within_class_branch_active_majority": swept_branch_active_majority,
            "C1_swept_authority_normalization_fired_majority": swept_authority_norm_majority,
            "C1_legacy_committed_entropy_mean": round(legacy_committed_mean, 6),
            "C1_legacy_committed_entropy_nondegenerate": legacy_nondegen,
            "C2_within_class_committed_class_lift_per_arm": c2_holds,
            "C2_per_arm_paired_lift_seed_counts": {
                k: int(v) for k, v in c2_per_arm_lift.items()
            },
            "C2_legacy_committed_class_entropy_mean": round(
                legacy_committed_mean, 6
            ),
            "C2_per_arm_committed_class_entropy_mean": {
                k: round(float(v), 6) for k, v in c2_per_arm_committed_mean.items()
            },
            "C3_substrate_readiness": c3_holds,
            "C3_per_arm_ready_seed_counts": {
                k: int(v) for k, v in c3_per_arm_ready.items()
            },
        },
        "authority_firing_diagnostics": {
            "authority_active_frac_by_arm": authority_active_frac_by_arm,
            "authority_normalization_fired_seed_counts_by_arm": {
                k: int(v) for k, v in authority_norm_fired_per_arm.items()
            },
            "within_class_active_seed_counts_per_arm": {
                k: int(v) for k, v in within_class_active_per_arm.items()
            },
            "note": (
                "Site-1 = additive authority in e3_selector.select "
                "(modulatory_authority_active). Site-2 = stratified across-class "
                "unit-range normalization in e3_score_diversity.stratified_select "
                "(mech341_n_authority_normalized). Site-2 is the load-bearing path "
                "for the within-class temperature lever -- it is the V3-EXQ-643a "
                "fix that lets the within-class representative shift reach "
                "committed-action selection (the 614d C2 byte-identical fix)."
            ),
        },
        "interpretation_grid": {
            "PASS_C1_C2": (
                "Within-class proportional sampling lever VALIDATED at the "
                "committed-action layer on the V3-EXQ-643a-fixed substrate. "
                "MECH-341 within-class sub-axis is load-bearing on committed-class "
                "diversity. Route to /governance: MECH-341 v3_pending clearance "
                "candidate (supports); record the winning within-class temperature "
                "for the Q-054 re-issue + arc_062 GAP-B wiring. Do NOT auto-flip "
                "the within-class default (governance ratification gate)."
            ),
            "FAIL_C1_holds_C2_fails": (
                "Lever operative (within-class branch fires + Site-2 authority "
                "normalization fires) but adds no marginal committed-class "
                "diversity over legacy argmin under the working substrate. "
                "MECH-341 within-class sub-axis = weakens. Route to /governance "
                "with the within-class sub-axis as not load-bearing; propagate to "
                "Q-054 and the arc_062 GAP-B wiring decision (the within-class "
                "lever is not the lift source)."
            ),
            "FAIL_C1_substrate_not_operative": (
                "The within-class branch did not fire and/or Site-2 authority "
                "normalization did not fire and/or ARM_0 committed entropy was "
                "degenerate -- the substrate was not operative in this run, so "
                "the within-class lever could not express itself. This is NOT an "
                "MECH-341 falsification. Route to /diagnose-errors on the "
                "modulatory-authority / stratified-select wiring (verify "
                "use_modulatory_selection_authority armed both sites and the "
                "primary scores stayed bounded)."
            ),
        },
        "arms": arms_out,
    }


def _build_manifest(
    result: Dict[str, Any],
    timestamp_utc: str,
    dry_run: bool,
) -> Dict[str, Any]:
    run_id = f"{EXPERIMENT_TYPE}_{timestamp_utc}_v3"
    return {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "supersedes": SUPERSEDES,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": timestamp_utc,
        "outcome": result["outcome"],
        "evidence_direction": result["overall_direction"],
        "evidence_direction_per_claim": result["evidence_direction_per_claim"],
        "evidence_direction_note": (
            f"V3-EXQ-614e MECH-341 within-class temperature CLEAN RETEST with "
            f"modulatory-bias-selection-authority ON (supersedes V3-EXQ-614d, "
            f"which FAILed only because the score-side lever was zeroed at "
            f"E3.select -- the float32 catastrophic-cancellation bug fixed by "
            f"V3-EXQ-643a). C2 (PRIMARY) tests whether committed_class_entropy "
            f"RISES with within-class temperature (per-arm >= "
            f"{C2_MIN_LIFT_SEEDS_PER_ARM} paired-lift seed); C1 is the "
            f"substrate-operative non-vacuity guard (within-class branch fires + "
            f"Site-2 authority normalization fires + ARM_0 committed entropy "
            f"non-degenerate). interpretation_label="
            f"{result['interpretation_label']}. "
            f"C1 (substrate operative)="
            f"{result['acceptance_criteria']['C1_substrate_operative_non_vacuity']}, "
            f"C2 (per-arm within-class lift)="
            f"{result['acceptance_criteria']['C2_within_class_committed_class_lift_per_arm']}, "
            f"C3 (substrate-readiness)={result['acceptance_criteria']['C3_substrate_readiness']}. "
            f"experiment_purpose=evidence; MECH-341 is a v3_pending candidate "
            f"(e3_scoring_preserves_trajectory_class_diversity)."
        ),
        "dry_run": bool(dry_run),
        "env_kwargs": dict(ENV_KWARGS),
        "config_summary": {
            "vs_stack": "minimal (use_per_stream_vs + use_vs_rollout_gating)",
            "mech341_sub_flavours": "both (entropy_bonus + stratified_select)",
            "mech341_entropy_bias_scale": MECH341_ENTROPY_BIAS_SCALE,
            "within_class_temperature_by_arm": {
                k: v for k, v in WITHIN_CLASS_T_BY_ARM.items()
            },
            "use_modulatory_selection_authority": USE_MODULATORY_SELECTION_AUTHORITY,
            "modulatory_authority_gain": MODULATORY_AUTHORITY_GAIN,
            "z_goal_enabled": True,
            "drive_weight": 2.0,
            "alpha_world": 0.9,
            "reef_enabled": True,
            "reef_bipartite_layout": True,
            "sd056_amend_active": True,
            "sd056_t1_contrastive_enabled": SD056_T1_CONTRASTIVE_ENABLED,
            "sd056_t1_contrastive_weight": SD056_T1_CONTRASTIVE_WEIGHT,
            "sd056_multistep_contrastive": SD056_MULTISTEP_CONTRASTIVE,
            "sd056_contrastive_horizon": SD056_CONTRASTIVE_HORIZON,
            "sd056_output_norm_clamp": SD056_OUTPUT_NORM_CLAMP,
            "sd056_output_norm_clamp_ratio": SD056_OUTPUT_NORM_CLAMP_RATIO,
            "use_differentiable_cem": "NOT FLIPPED (default False; SD-055 safety note)",
        },
        "result": result,
    }


def main() -> Tuple[Optional[str], Optional[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()

    if args.dry_run:
        seeds = list(DRY_RUN_SEEDS)
        p0 = DRY_RUN_P0
        p1 = DRY_RUN_P1
        steps = DRY_RUN_STEPS
    else:
        seeds = list(SEEDS)
        p0 = P0_WARMUP_EPISODES
        p1 = P1_MEASUREMENT_EPISODES
        steps = STEPS_PER_EPISODE

    result = run_experiment(
        seeds=seeds,
        p0_episodes=p0,
        p1_episodes=p1,
        steps_per_episode=steps,
        dry_run=bool(args.dry_run),
    )

    timestamp_utc = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    manifest = _build_manifest(result, timestamp_utc, dry_run=bool(args.dry_run))

    if args.out_dir is not None:
        out_dir = Path(args.out_dir)
    else:
        out_dir = (
            Path(__file__).resolve().parents[2]
            / "REE_assembly" / "evidence" / "experiments"
        )
    out_path = write_flat_manifest(
        manifest,
        out_dir,
        dry_run=args.dry_run,
        config=manifest.get("config") or manifest.get("config_summary"),
        seeds=SEEDS,
        script_path=Path(__file__),
    )

    print(f"manifest: {out_path}", flush=True)
    if not args.dry_run:
        print(f"Result written to: {out_path}", flush=True)
    print(
        f"outcome: {result['outcome']} "
        f"completed={result['total_seeds_completed']}/{result['total_seeds_attempted']} "
        f"C1={result['acceptance_criteria']['C1_substrate_operative_non_vacuity']} "
        f"C2={result['acceptance_criteria']['C2_within_class_committed_class_lift_per_arm']} "
        f"C3={result['acceptance_criteria']['C3_substrate_readiness']} "
        f"label={result['interpretation_label']}",
        flush=True,
    )

    if args.dry_run:
        try:
            out_path.unlink()
        except FileNotFoundError:
            pass

    outcome_norm = result["outcome"].upper()
    outcome_emit = outcome_norm if outcome_norm in ("PASS", "FAIL") else "FAIL"
    manifest_for_sentinel = str(out_path) if not args.dry_run else None
    return outcome_emit, manifest_for_sentinel


if __name__ == "__main__":
    _outcome, _manifest_path = main()
    if _outcome is not None:
        emit_outcome(outcome=_outcome, manifest_path=_manifest_path)
    sys.exit(0)
