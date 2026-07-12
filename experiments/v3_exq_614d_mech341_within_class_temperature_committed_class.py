#!/opt/local/bin/python3
"""
V3-EXQ-614d -- MECH-341 stratified_within_class_temperature 4-arm sweep,
CORRECTED HARNESS (committed-action class distribution + within-class
firing diagnostics + cross-seed-mean regression guard).

Successor to V3-EXQ-614c (FAIL_C1_regression 2026-06-01, manifest
v3_exq_614c_mech341_stratified_within_class_temperature_sweep_20260601T124509Z_v3.json),
which RAN to completion (12/12 cells, no crash) but was reclassified
non_contributory for MECH-341 because BOTH failing criteria were
test-harness defects, NOT substrate falsification (see
REE_assembly/evidence/planning/failure_autopsy_V3-EXQ-614c_2026-06-01.md).
This re-run keeps the substrate config bit-identical to 614c and fixes
ONLY the measurement harness.

EXPERIMENT_PURPOSE = diagnostic. This is a corrected-harness substrate-
readiness / measurability re-run: it confirms whether the 2026-06-01
MECH-341 within-class proportional-sampling amend produces a MEASURABLE
effect under the SD-056-amended baseline when the temperature-sensitive
signal is read at the right layer. It is excluded from governance
confidence/conflict scoring (Phase-3 rule); it gates MECH-341 v3_pending
clearance, the Q-054 re-issue (V3-EXQ-616a), and arc_062_rule_apprehension:
GAP-B (V3-EXQ-543l cohort) at the governance-decision level.

What the two 614c defects were, and how 614d fixes them
-------------------------------------------------------
DEFECT 1 (C2 structurally vacuous). 614c measured
`selected_class_entropy_nats` from `_per_class_score_stats` using
`sel_idx = scores_t.argmin()` -- the SCORE-LAYER argmin class. That
quantity is upstream of, and TEMPERATURE-INVARIANT to, the within-class
temperature lever (the lever only changes WHICH within-class candidate
becomes the committed representative inside stratified_select, and only
when a first-action class holds >= 2 candidates). Result: ARM_1/2/3
(T=0.5/1.0/2.0) were bit-identical per seed -- zero discriminative signal.
  FIX: 614d's C2 reads the COMMITTED-ACTION class distribution
  (`committed_classes_p1_counts` -> committed_class_entropy_nats). The
  within-class temperature changes the committed within-class
  representative, which changes its representative score, which changes
  the across-class softmax, which changes the committed-class
  distribution over P1 ticks. That is the temperature-sensitive signal.
  614d ALSO records the lever's own firing diagnostics per seed
  (mech341_n_within_class_sampled, mech341_n_stratified_fired,
  mech341_last_within_class_temperature) from agent.score_diversity.
  get_state(), so a null result can be read as "lever never fired (env
  starves multi-candidate-per-class events)" vs "lever fired but no
  committed-class lift" -- the distinction 614c could not make.

DEFECT 2 (C1 regression predicate mis-specified). 614c's C1 checked each
ARM_0 seed's entropy against the band [0.720, 0.880] (614b ARM_2 ALL_ON
cross-seed mean 0.800 +/- 10%) and required >= 2/3 seeds IN band. ARM_0
per-seed entropies [1.177, 0.530, 0.693] have CROSS-SEED MEAN 0.7999
(~= 0.800, no regression) but 0/3 land in the +/-10% band because of high
cross-seed variance -- a per-seed-vs-cross-seed-mean category error.
  FIX: 614d's C1 compares the ARM_0_LEGACY CROSS-SEED MEAN of
  selected_class_entropy_nats against the band. The selected-class
  (score-layer argmin) metric is retained ONLY for C1, because that is the
  exact quantity 614b's 0.800 reference was measured on, and ARM_0
  (within_temp=None, legacy argmin) is the apples-to-apples regression
  control for the pre-amend 614b ALL_ON path. C1 is purely a "did the amend
  break the legacy bit-identical-OFF path / did the substrate drift" guard;
  it does NOT carry the within-class signal (C2 does).

Arms (4, all on the same SD-056-amended baseline as V3-EXQ-614b ARM_2 /
614c -- identical to 614c so the only difference between 614c and 614d is
the measurement harness)
----------------------------------------------------------------------
  ARM_0_LEGACY:   stratified_within_class_temperature = None   (legacy argmin)
  ARM_1_T_0_5:    stratified_within_class_temperature = 0.5    (sharpened)
  ARM_2_T_1_0:    stratified_within_class_temperature = 1.0    (mid-T)
  ARM_3_T_2_0:    stratified_within_class_temperature = 2.0    (flatter)

All other levers held at the V3-EXQ-614b ARM_2 ALL_ON config:
  Layer A: SP-CEM ON (use_support_preserving_cem=True)
  Layer B: MECH-341 ON (use_e3_score_diversity=True; both sub-flavours ON;
           entropy_bias_scale=2.0)
  Layer C: MECH-313 ON (use_noise_floor=True; noise_floor_alpha=0.1)
  Layer D: V_s minimal stack ON (use_per_stream_vs=True + use_vs_rollout_gating=True)
  SD-056 amend: all 5 lever flags ON (multi-step contrastive h=5 +
                per-step output norm clamp ratio=2.0 + t=1 contrastive)
  use_differentiable_cem: NOT FLIPPED (default False; SD-055 safety note
                          per substrate_queue.json preserved)

Pre-registered acceptance criteria (corrected)
----------------------------------------------
  C1 (regression guard, Defect-2 fix): ARM_0_LEGACY CROSS-SEED MEAN of
     mean_selected_class_entropy_nats in [0.720, 0.880] (= V3-EXQ-614b
     ARM_2 ALL_ON 0.800 nats +/- 10%). Cross-seed mean -- NOT per-seed band
     membership. Direct test that the amend's default-None legacy path
     still reproduces the pre-amend 614b ALL_ON behaviour and that the
     upstream SD-056 substrate has not drifted.
  C2 (within-class lift, Defect-1 fix): at least one of {ARM_1, ARM_2,
     ARM_3} produces a per-seed PAIRED lift in committed_class_entropy_nats
     over ARM_0_LEGACY of at least C2_LIFT_MARGIN_NATS on a majority
     (>= 2/3) of seeds. Paired by seed index (same seed -> same env ->
     fair within-seed comparison). Direct test that within-class
     proportional sampling adds COMMITTED-action class diversity over the
     legacy argmin-within-class baseline.
  C3 (substrate-readiness): all 4 arms produce frac_pre_ge2 > 0.3 on a
     majority (>= 2/3) of seeds. Substrate-readiness check at the SP-CEM-ON
     layer; should be high since SP-CEM is active across all arms.

Within-class firing gate (interpretive, not a hard PASS criterion)
------------------------------------------------------------------
  Per seed, mech341_n_within_class_sampled is accumulated over the P1
  window. A positive-temperature arm whose within-class branch fired on a
  majority of seeds (>= WITHIN_CLASS_FIRE_FLOOR samples on >= 2/3 seeds) is
  "branch-active"; a null C2 on a branch-active arm is a genuine
  no-committed-class-lift finding. A null C2 where the branch barely fired
  (env starves multi-candidate-per-class pools) is an env/candidate-pool
  limitation, NOT a MECH-341 falsification. The interpretation grid below
  routes these two cases differently.

Overall outcome
---------------
  PASS = C1 (regression guard holds) AND (C2 OR C3).
  FAIL = C1 fails (amend broke legacy bit-identical OFF, or upstream
         substrate drifted) OR neither C2 nor C3 fires.

Interpretation grid (diagnostic -- one row per outcome -> next action)
----------------------------------------------------------------------
  PASS via C1 + C2:
    Within-class proportional sampling lever VALIDATED at the committed-
    action layer. MECH-341 within-class sub-axis is load-bearing on
    committed-class diversity under the SD-056-amended substrate. Route to
    /governance: MECH-341 v3_pending clearance candidate; record the
    winning within-class temperature for the Q-054 re-issue (V3-EXQ-616a)
    and arc_062 GAP-B (V3-EXQ-543l cohort) wiring decisions. Do NOT
    auto-flip the within-class default (governance ratification gate).

  PASS via C1 + C3 only (no committed-class lift), within-class branch
  ACTIVE (fired on majority of seeds in the positive arms):
    Amend is functional and the branch fires, but within-class proportional
    sampling produces no marginal committed-class diversity over legacy
    argmin under SP-CEM-ON. Layer B within-class sub-axis may be redundant
    with the across-class layer / Layer A SP-CEM. Route to /governance with
    MECH-341 within-class sub-axis = mixed; propagate to the Q-054
    collapse-with-ARC-065 question; the within-class lever is not the lift
    source for arc_062 GAP-B.

  PASS via C1 + C3 only (no committed-class lift), within-class branch
  STARVED (rarely fired -- the SD-054 reef env produces few multi-candidate
  -per-class pools):
    The lever could not express itself: the precondition (>= 2 candidates
    in one first-action class) is rarely met, so within-class temperature
    is a near-no-op regardless of T. This is an ENV / candidate-pool
    limitation, NOT a MECH-341 falsification. Route to /queue-experiment
    for a successor that enlarges the candidate pool or lowers
    min_classes_for_stratification pressure (or reduces SP-CEM class
    separation) so the within-class branch fires often enough to be
    testable. MECH-341 stays pending_retest_after_corrected_harness.

  FAIL via C1 fail (ARM_0 cross-seed mean outside [0.720, 0.880]):
    The amend broke the legacy bit-identical-OFF guarantee OR the upstream
    SD-056-amended substrate drifted between the 614b run and this run.
    Route to /failure-autopsy on the amend + upstream substrate -- the only
    path the contract suite cannot catch (contracts verify per-call output
    determinism, not end-to-end environment-level behaviour).

  FAIL via neither C2 nor C3 (C1 holds):
    Legacy path intact but the substrate is not surfacing committed-class
    diversity at all and within-class proportional sampling adds none.
    Substrate-redesign signal. Route to /failure-autopsy on the combined
    MECH-341 + SP-CEM stack.

Claims: [MECH-341] (single claim; ARC-065 dropped per the claim_ids
accuracy rule -- only the Layer-B within-class sub-axis is varied here).
experiment_purpose=diagnostic (excluded from confidence/conflict scoring).

Phases
------
P0 (30 ep, instrumentation OFF): encoder warmup.
P1 (60 ep, instrumentation ON): behavioural measurement window. Matches the
V3-EXQ-614b / 614c P1 budget for direct manifest comparability.

Budget: 4 arms x 3 seeds x 90 ep x 200 steps = 216k steps total.
Estimated ~3-4 h on Mac (DLAPTOP-4.local) under the full SD-056-amended
4-substrate stack (SP-CEM + MECH-341 + MECH-313 + V_s + multi-step
contrastive). Mac-pinned for direct numerical comparability with the
614b / 614c reference manifests (the C1 regression guard compares against
the 614b ARM_2 ALL_ON value measured on the same machine).

Implementation notes
--------------------
- env_kwargs and the 4-substrate + SD-056 config are IDENTICAL to
  V3-EXQ-614c -- the ONLY difference between 614c and 614d is the
  measurement harness (committed-class metric for C2; cross-seed-mean for
  C1; within-class firing diagnostics recorded).
- Within-class diagnostics are read from agent.score_diversity.get_state()
  at the END of each P1 episode and accumulated per seed. agent.reset()
  (called at the top of each episode) resets E3ScoreDiversity diagnostics
  per episode (ree_core/agent.py:1544 -> E3ScoreDiversity.reset()), so the
  per-episode get_state() reflects that episode's cumulative counts; 614d
  sums them across the P1 window.

See REE_assembly/docs/architecture/mech_341_e3_score_diversity_preservation.md
("Amend 2026-06-01" section), ree-v3/CLAUDE.md "MECH-341 Amend:
stratified_within_class_temperature" section,
REE_assembly/evidence/planning/failure_autopsy_V3-EXQ-614c_2026-06-01.md
(the autopsy that routed this corrected-harness re-run),
REE_assembly/evidence/planning/outstanding_tasks_triage_2026-06-02.md item 3,
REE_assembly/evidence/planning/behavioral_diversity_isolation_plan.md GAP-B,
REE_assembly/evidence/planning/substrate_queue.json MECH-341 amend_history.
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


EXPERIMENT_TYPE = "v3_exq_614d_mech341_within_class_temperature_committed_class"
QUEUE_ID = "V3-EXQ-614d"
SUPERSEDES = "V3-EXQ-614c"
CLAIM_IDS: List[str] = ["MECH-341"]
EXPERIMENT_PURPOSE = "diagnostic"

# V3-EXQ-614d sweep axis: within-class temperature in {None, 0.5, 1.0, 2.0}.
# None = legacy argmin within-class (bit-identical OFF; regression guard arm).
# Set on E3ScoreDiversityConfig.stratified_within_class_temperature via
# REEConfig.e3_diversity_stratified_within_class_temperature. See MECH-341
# amend 2026-06-01 (ree-v3/CLAUDE.md + design doc + substrate_queue
# amend_history).
WITHIN_CLASS_T_BY_ARM: Dict[str, Optional[float]] = {
    "ARM_0_LEGACY": None,
    "ARM_1_T_0_5": 0.5,
    "ARM_2_T_1_0": 1.0,
    "ARM_3_T_2_0": 2.0,
}

# V3-EXQ-614b ARM_2 ALL_ON reference (C1 regression-guard target). This is a
# SELECTED-class (score-layer argmin) entropy value -- the exact quantity
# 614b measured -- so the C1 regression guard reads the same metric on the
# ARM_0_LEGACY arm (within_temp=None, the apples-to-apples legacy control).
EXQ_614B_ALL_ON_ENTROPY_NATS = 0.800
C1_REGRESSION_FRACTION = 0.10   # +/- 10% of the reference value
C1_REGRESSION_LOWER = EXQ_614B_ALL_ON_ENTROPY_NATS * (1.0 - C1_REGRESSION_FRACTION)
C1_REGRESSION_UPPER = EXQ_614B_ALL_ON_ENTROPY_NATS * (1.0 + C1_REGRESSION_FRACTION)

# C2 within-class committed-class lift: at least one of T in {0.5, 1.0, 2.0}
# produces a per-seed PAIRED lift in committed_class_entropy_nats over
# ARM_0_LEGACY of at least this margin on a majority of seeds. Pre-registered
# margin chosen small relative to the committed-class entropy magnitudes
# (~0.5-1.2 nats) but above per-seed measurement noise.
C2_LIFT_MARGIN_NATS = 0.05

# Within-class firing gate (interpretive). A positive-temperature arm whose
# within-class branch accumulated at least this many samples on a majority of
# seeds is "branch-active"; below it the lever is starved by the env (rare
# multi-candidate-per-class pools) and a null C2 is an env limitation, not a
# MECH-341 falsification.
WITHIN_CLASS_FIRE_FLOOR = 10

# SD-056 amend lever defaults applied uniformly across all 4 arms
# (multi-step contrastive horizon + per-step output norm clamp ratio).
# See ree-v3/CLAUDE.md "SD-056 multi-step rollout stability amend (2026-05-31)".
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

# Pre-registered behavioural thresholds (preserved from the 614 lineage so
# the P1 per-tick semantics remain comparable across the cluster).
RUNG1_ENTROPY_THRESHOLD = 0.3
RUNG1_MIN_CLASSES = 2
MIN_SEEDS_PER_ARM_FOR_PASS = 2  # of 3

# Pre-registered measurement gates (preserved from V3-EXQ-611 lineage so the
# P1 per-tick semantics remain comparable across the cluster).
PRE_GE2_FRAC_GATE = 0.5

# MECH-341 sub-flavour scale used in the entropy-ON arms. 2.0 was the
# upper-range value selected for the V3-EXQ-611b retune sweep target.
MECH341_ENTROPY_BIAS_SCALE = 2.0

# V_s (D) thresholds (minimal stack). 0.5 / 0.4 match the V3-EXQ-601
# MECH-269b substrate-readiness PASS defaults.
VS_SNAPSHOT_REFRESH_THRESHOLD = 0.5
VS_E1_THRESHOLD = 0.4


# IDENTICAL to V3-EXQ-611 / 611b / 611c / 614c for direct manifest
# comparability across the cluster.
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
    """Build a REEAgent with the SD-056-amended 4-substrate baseline.

    Config is bit-identical to V3-EXQ-614c _make_agent; the only swept axis
    is e3_diversity_stratified_within_class_temperature (None on the legacy
    arm, a positive float on the swept arms). See the V3-EXQ-614c docstring /
    ree-v3/CLAUDE.md SD Design Decisions sections for the axis -> flag map.
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
        # sampling temperature. None = legacy argmin within-class
        # (bit-identical to pre-amend, V3-EXQ-614d ARM_0_LEGACY regression
        # guard). Positive float = sample within each first-action class via
        # softmax(-class_scores / T) before the existing across-class
        # softmax step.
        e3_diversity_stratified_within_class_temperature=within_class_temperature,
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
# Per-tick measurement helpers (mirror EXQ-608 / EXQ-611 / 614c conventions)
# ---------------------------------------------------------------------------


def _trajectory_first_action_class(traj) -> int:
    return int(
        traj.actions[:, 0, :].argmax(dim=-1).detach().reshape(-1)[0].item()
    )


def _per_class_score_stats(
    candidates: List, last_scores: Optional[torch.Tensor]
) -> Tuple[Dict[int, Dict[str, float]], Optional[int], Optional[float], bool]:
    """Score-layer per-class stats. The SELECTED class here is the
    score-layer argmin -- temperature-INVARIANT by construction. It is used
    ONLY for the C1 regression guard (apples-to-apples with the 614b
    selected-class 0.800 reference); the temperature-sensitive C2 signal is
    the COMMITTED-action class distribution measured separately below.
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

    # MECH-341 within-class firing diagnostics, accumulated over the P1
    # window (E3ScoreDiversity diagnostics reset per episode via agent.reset).
    n_within_class_sampled_total = 0
    n_stratified_fired_total = 0
    last_within_class_temperature = 0.0

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

        # MECH-341 within-class diagnostics: read AFTER the inner step loop
        # (so the per-episode E3ScoreDiversity counters reflect this episode's
        # cumulative firing) and BEFORE the next episode's agent.reset() clears
        # them. Accumulate only over the P1 measurement window.
        if is_p1 and getattr(agent, "score_diversity", None) is not None:
            sd_state = agent.score_diversity.get_state()
            n_within_class_sampled_total += int(
                sd_state.get("mech341_n_within_class_sampled", 0)
            )
            n_stratified_fired_total += int(
                sd_state.get("mech341_n_stratified_fired", 0)
            )
            ep_within_temp = float(
                sd_state.get("mech341_last_within_class_temperature", 0.0)
            )
            if ep_within_temp != 0.0:
                last_within_class_temperature = ep_within_temp

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
    else:
        frac_pre_ge2 = 0.0

    if top2_gaps_pre_ge2:
        mean_top2_gap = float(sum(top2_gaps_pre_ge2) / len(top2_gaps_pre_ge2))
    else:
        mean_top2_gap = None

    n_selected_classes = len(selected_classes_p1)
    n_committed_classes = len(committed_classes_p1)
    selected_class_entropy = _entropy_from_counts(selected_classes_p1)
    committed_class_entropy = _entropy_from_counts(committed_classes_p1)

    # Per-seed substrate-readiness flag (C3 input): pre-E3 pool diversity gate.
    seed_substrate_ready = bool(frac_pre_ge2 > 0.3)

    # Per-seed within-class branch activity (interpretive gate).
    within_class_branch_active = bool(
        n_within_class_sampled_total >= WITHIN_CLASS_FIRE_FLOOR
    )

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
        # C1 metric (score-layer argmin; temperature-invariant; regression guard):
        "selected_class_entropy_nats": round(selected_class_entropy, 6),
        # C2 metric (committed action; temperature-sensitive; lift signal):
        "committed_class_entropy_nats": round(committed_class_entropy, 6),
        # MECH-341 within-class firing diagnostics (per-seed accumulated):
        "mech341_n_within_class_sampled": int(n_within_class_sampled_total),
        "mech341_n_stratified_fired": int(n_stratified_fired_total),
        "mech341_last_within_class_temperature": round(
            last_within_class_temperature, 6
        ),
        "within_class_branch_active": within_class_branch_active,
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
        "within_class_samples_per_seed": [int(s) for s in within_class_samples],
    }


def _classify_outcome(
    c1: bool, c2: bool, c3: bool, any_positive_arm_active: bool
) -> Tuple[str, str, str]:
    """V3-EXQ-614d interpretation grid.

    Map (c1, c2, c3, any_positive_arm_active) -> (outcome,
    mech341_direction, interpretation_label).

    C1: regression guard. ARM_0_LEGACY CROSS-SEED MEAN of
        selected_class_entropy in [0.720, 0.880] (614b ARM_2 ALL_ON 0.800
        +/- 10%).
    C2: committed-class lift. At least one of ARM_1/2/3 has a per-seed
        paired committed_class_entropy lift over ARM_0 >= margin on majority
        of seeds.
    C3: substrate-readiness. All 4 arms frac_pre_ge2 > 0.3 on majority of
        seeds.
    any_positive_arm_active: at least one positive-temp arm fired its
        within-class branch on a majority of seeds (interpretive gate that
        distinguishes "lever functional but no lift" from "lever starved").

    PASS = C1 AND (C2 OR C3).
    """
    if c1 and c2:
        return (
            "PASS", "supports",
            "PASS_C1_C2_within_class_lever_lifts_committed_class_diversity",
        )
    if c1 and (not c2) and c3:
        if any_positive_arm_active:
            return (
                "PASS", "mixed",
                "PASS_C1_C3_only_within_class_active_no_committed_class_lift",
            )
        return (
            "PASS", "non_contributory",
            "PASS_C1_C3_only_within_class_branch_starved_env_limited",
        )
    if not c1:
        return (
            "FAIL", "weakens",
            "FAIL_C1_regression_against_614b_routes_to_amend_autopsy",
        )
    # c1=True, c2=False, c3=False
    return (
        "FAIL", "weakens",
        "FAIL_neither_c2_nor_c3_substrate_not_surfacing_diversity",
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
            f"steps_per_episode={steps_per_episode}, dry_run={dry_run})",
            flush=True,
        )
        seed_rows: List[Dict[str, Any]] = []
        for s in seeds:
            print(
                f"Seed {s} Condition {arm['label']}", flush=True,
            )
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

    # ----- Cross-arm acceptance criteria (V3-EXQ-614d corrected harness) -----
    by_id = {a["arm_id"]: a for a in arms_out}
    arm_legacy = by_id["ARM_0_LEGACY"]
    swept_arms = [by_id["ARM_1_T_0_5"], by_id["ARM_2_T_1_0"], by_id["ARM_3_T_2_0"]]

    # C1 (regression guard, Defect-2 fix): ARM_0_LEGACY CROSS-SEED MEAN of
    # selected_class_entropy_nats within +/- 10% of the V3-EXQ-614b ARM_2
    # ALL_ON 0.800 nats reference.
    legacy_seed_sel_entropies = [
        r["selected_class_entropy_nats"]
        for r in arm_legacy["per_seed_results"]
        if r["error_note"] is None
    ]
    legacy_mean_sel_entropy = (
        sum(legacy_seed_sel_entropies) / len(legacy_seed_sel_entropies)
        if legacy_seed_sel_entropies else 0.0
    )
    c1_holds = bool(
        C1_REGRESSION_LOWER <= legacy_mean_sel_entropy <= C1_REGRESSION_UPPER
    )

    # C2 (committed-class within-class lift, Defect-1 fix): per-seed PAIRED
    # lift of committed_class_entropy_nats over ARM_0_LEGACY on majority of
    # seeds, for at least one positive-temperature arm. Paired by seed index.
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
    c2_holds = any(
        n >= MIN_SEEDS_PER_ARM_FOR_PASS for n in c2_per_arm_lift.values()
    )

    # Per-arm committed-class entropy means (audit context for C2).
    c2_per_arm_committed_mean = {
        a["arm_id"]: a["cross_seed_interpretation"]["mean_committed_class_entropy_nats"]
        for a in swept_arms
    }
    legacy_committed_mean = arm_legacy["cross_seed_interpretation"][
        "mean_committed_class_entropy_nats"
    ]

    # C3 (substrate-readiness): all 4 arms frac_pre_ge2 > 0.3 on majority of
    # seeds.
    c3_per_arm_ready = {
        a["arm_id"]: a["cross_seed_interpretation"]["n_seeds_substrate_ready"]
        for a in arms_out
    }
    c3_holds = all(
        n >= MIN_SEEDS_PER_ARM_FOR_PASS for n in c3_per_arm_ready.values()
    )

    # Within-class firing interpretive gate: at least one positive-temp arm
    # fired its within-class branch on a majority of seeds.
    any_positive_arm_active = any(
        a["cross_seed_interpretation"]["majority_within_class_active"]
        for a in swept_arms
    )
    within_class_active_per_arm = {
        a["arm_id"]: a["cross_seed_interpretation"]["n_seeds_within_class_active"]
        for a in swept_arms
    }
    within_class_samples_per_arm = {
        a["arm_id"]: a["cross_seed_interpretation"]["within_class_samples_per_seed"]
        for a in swept_arms
    }

    (
        outcome_label, mech341_direction, interpretation_label,
    ) = _classify_outcome(c1_holds, c2_holds, c3_holds, any_positive_arm_active)
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
            "exq_614b_all_on_reference_nats": float(EXQ_614B_ALL_ON_ENTROPY_NATS),
            "c1_regression_band_lower": float(C1_REGRESSION_LOWER),
            "c1_regression_band_upper": float(C1_REGRESSION_UPPER),
            "c2_lift_margin_nats": float(C2_LIFT_MARGIN_NATS),
            "within_class_fire_floor": int(WITHIN_CLASS_FIRE_FLOOR),
        },
        "acceptance_criteria": {
            "C1_legacy_regression_band_cross_seed_mean": c1_holds,
            "C1_legacy_cross_seed_mean_selected_entropy": round(
                legacy_mean_sel_entropy, 6
            ),
            "C1_legacy_per_seed_selected_entropies": [
                round(e, 6) for e in legacy_seed_sel_entropies
            ],
            "C2_committed_class_within_class_lift": c2_holds,
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
        "within_class_firing_diagnostics": {
            "any_positive_arm_branch_active": any_positive_arm_active,
            "within_class_active_seed_counts_per_arm": {
                k: int(v) for k, v in within_class_active_per_arm.items()
            },
            "within_class_samples_per_seed_per_arm": within_class_samples_per_arm,
            "fire_floor": int(WITHIN_CLASS_FIRE_FLOOR),
        },
        "interpretation_grid_header_note_614d": (
            "V3-EXQ-614d corrected harness. Supersedes V3-EXQ-614c, which RAN "
            "but was reclassified non_contributory for MECH-341 due to two "
            "test-harness defects (NOT substrate falsification): (1) C2 "
            "measured selected_class_entropy at the score-layer argmin, which "
            "is temperature-INVARIANT to the within-class lever -> ARM_1/2/3 "
            "were bit-identical; (2) C1 compared per-seed entropies to a band "
            "built from a cross-seed mean. 614d fixes both: C2 reads the "
            "COMMITTED-action class distribution (committed_class_entropy_nats, "
            "the temperature-sensitive signal) with a per-seed paired lift "
            "over ARM_0_LEGACY, and records the within-class firing "
            "diagnostics (mech341_n_within_class_sampled) so a null result is "
            "readable as 'branch starved by env' vs 'branch active but no "
            "lift'; C1 compares the ARM_0_LEGACY CROSS-SEED MEAN to the "
            "[0.720, 0.880] band. Substrate config is bit-identical to 614c "
            "(SD-056-amended 4-substrate baseline). experiment_purpose="
            "diagnostic (corrected-harness measurability re-run)."
        ),
        "interpretation_grid": {
            "PASS_C1_C2": (
                "Within-class proportional sampling lever VALIDATED at the "
                "committed-action layer. MECH-341 within-class sub-axis is "
                "load-bearing on committed-class diversity under the "
                "SD-056-amended substrate. Route to /governance for MECH-341 "
                "v3_pending clearance candidate + record the winning "
                "within-class temperature for the Q-054 re-issue "
                "(V3-EXQ-616a) and arc_062 GAP-B (V3-EXQ-543l cohort). Do NOT "
                "auto-flip the within-class default (governance ratification "
                "gate)."
            ),
            "PASS_C1_C3_only_branch_active": (
                "Amend functional and within-class branch fires, but "
                "within-class proportional sampling adds no marginal "
                "committed-class diversity over legacy argmin under SP-CEM-ON. "
                "Layer B within-class sub-axis may be redundant with the "
                "across-class layer / Layer A. Route to /governance with "
                "MECH-341 within-class sub-axis = mixed; propagate to the "
                "Q-054 collapse-with-ARC-065 question."
            ),
            "PASS_C1_C3_only_branch_starved": (
                "The within-class branch rarely fired (the SD-054 reef env "
                "produces few multi-candidate-per-class pools), so the lever "
                "could not express itself regardless of T. This is an ENV / "
                "candidate-pool limitation, NOT a MECH-341 falsification. "
                "Route to /queue-experiment for a successor enlarging the "
                "candidate pool / lowering min_classes_for_stratification "
                "pressure. MECH-341 stays pending_retest_after_corrected_harness."
            ),
            "FAIL_C1_regression": (
                "ARM_0_LEGACY cross-seed mean outside [0.720, 0.880]: the "
                "amend broke the legacy bit-identical-OFF guarantee OR the "
                "upstream SD-056-amended substrate drifted between the 614b "
                "run and this run. Route to /failure-autopsy on the amend + "
                "upstream substrate (the only path the contract suite cannot "
                "catch: contracts verify per-call output determinism, not "
                "end-to-end environment-level behaviour)."
            ),
            "FAIL_neither_c2_nor_c3": (
                "Legacy path intact but the substrate is not surfacing "
                "committed-class diversity at all and within-class "
                "proportional sampling adds none. Substrate-redesign signal. "
                "Route to /failure-autopsy on the combined MECH-341 + SP-CEM "
                "stack."
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
            f"experiment_purpose=diagnostic; V3-EXQ-614d MECH-341 within-class "
            f"temperature CORRECTED-HARNESS re-run (supersedes V3-EXQ-614c "
            f"non_contributory). C2 reads committed-action class distribution "
            f"(temperature-sensitive); C1 compares ARM_0 cross-seed mean to "
            f"the 614b ALL_ON band; within-class firing diagnostics recorded. "
            f"interpretation_label={result['interpretation_label']}. "
            f"C1 (legacy regression, cross-seed mean)="
            f"{result['acceptance_criteria']['C1_legacy_regression_band_cross_seed_mean']}, "
            f"C2 (committed-class within-class lift)="
            f"{result['acceptance_criteria']['C2_committed_class_within_class_lift']}, "
            f"C3 (substrate-readiness)={result['acceptance_criteria']['C3_substrate_readiness']}. "
            f"Diagnostic (excluded from confidence/conflict scoring); gates "
            f"MECH-341 v3_pending clearance, Q-054 re-issue (V3-EXQ-616a), and "
            f"arc_062 GAP-B (V3-EXQ-543l cohort) at the governance-decision level."
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
        f"C1={result['acceptance_criteria']['C1_legacy_regression_band_cross_seed_mean']} "
        f"C2={result['acceptance_criteria']['C2_committed_class_within_class_lift']} "
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
