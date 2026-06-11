#!/opt/local/bin/python3
"""
V3-EXQ-660b -- MECH-341 within-class-representative-diversity GRADED CONFIRMATION,
READOUT REDESIGN of V3-EXQ-660a (supersedes V3-EXQ-660a).

WHY THIS RUN (routed by failure_autopsy_V3-EXQ-660a_2026-06-11)
--------------------------------------------------------------
SAME scientific question as 660a: does the within-class-representative-diversity
LIFT (sampled stratified-select minus legacy argmin) GROW as more within-class
representatives become available (CEM pool size K)? 660a self-routed weakens but
was adjudicated non_contributory (measurement_test_design_defect): its primary
readout `within_class_rep_cond_entropy = H(rep_signature | committed_class)` was
a PHASE-AGGREGATE over EVERY P1 tick, computed once at phase end. That readout
(a) SATURATES -- absolute sampled entropy stayed flat ~4.6-4.9 nats while per-tick
within-class availability rose 5->34 (660a's C_ABS, a supporting/non-gating
criterion, FAILED -- the direct tell); and (b) is CONFOUNDED BY TICK COUNT --
seed 43 (~10k P1 ticks) sat ~6.3 nats in BOTH legacy and sampled arms vs seeds
42/44 (~1-2k ticks) ~3.7-4.6. The graded axis (pool size) provably moved the
INPUT (660a's availability non-vacuity gate passed) but the phase-aggregate
readout could not register a per-decision diversity benefit, so the lift was a
noise-dominated difference of two saturating tick-count-confounded aggregates.
660a guarded the INPUT but not READOUT SENSITIVITY -- the exact gap that let a
blind-readout FAIL masquerade as a weakens.

THE TWO FIXES (this run)
------------------------
1. PER-DECISION / WINDOWED READOUT WITH HEADROOM (replaces the phase-aggregate).
   The ordered P1 selected (committed_class, rep_signature) pairs are chunked into
   fixed-length windows of WINDOW_TICKS ticks; H(rep|class) is computed PER WINDOW
   and AVERAGED across complete windows (windowed_rep_cond_entropy_nats, the new
   PRIMARY readout). Fixed window length removes the tick-count confound (every
   window has the same tick budget, so a 10k-tick seed and a 1k-tick seed are
   compared window-for-window, not by accumulated coverage). A small window
   relative to the signature space preserves headroom: within a window the
   per-class selected-signature entropy is capped at log(min(ticks_in_window_for_
   class, available_distinct)), so as availability rises the windowed entropy can
   RISE rather than saturating toward the phase-marginal (the 660a failure). A
   normalised [0,1] SECONDARY readout (windowed_within_class_efficiency = distinct
   selected within-class reps / distinct available within-class reps, pooled per
   class per window) reports what FRACTION of available within-class diversity the
   lever actually exploits -- bounded, so it cannot tick-count-confound or
   saturate. The 660a phase-aggregate metric is ALSO reported per cell
   (phase_aggregate_rep_cond_entropy_nats) so the within-run contrast
   (saturating-aggregate vs sensitive-windowed) is auditable.

2. READOUT-SENSITIVITY GATE (the load-bearing 660a fix; same statistic as the
   criterion). Before C_GRADED is scored, the SAMPLED-arm absolute windowed
   readout must MOVE across K -- range(windowed_rep_cond_entropy across K in the
   SAMPLED positive-control arm) >= READOUT_SENSITIVITY_FLOOR. This asserts the
   SAME statistic C_GRADED routes its lift on (windowed H), measured on the
   positive control (the lever that actively samples diverse within-class reps),
   as a RANGE (not a magnitude proxy). If the windowed readout is itself flat
   across K, it is STILL blind to the swept axis at this scale -> self-route
   substrate_not_ready_requeue (FAIL/non_contributory), NEVER a weakens. Only when
   BOTH readiness gates hold (INPUT availability rises AND the windowed readout
   registers it) is a flat lift a GENUINE negative -- a weakens verdict is then
   trustworthy because the readout is demonstrably sensitive yet the marginal
   lift over legacy does not grow.

GRADED AXIS / ARMS / ENV / AGENT: IDENTICAL to V3-EXQ-660a.
cfg.hippocampal.num_candidates = K, swept K in {16, 32, 64, 128}; 8 arms
(2 lever states x 4 K), 3 seeds [42,43,44]; SD-056-amended 4-substrate baseline +
modulatory-bias selection authority (gain 0.5; V3-EXQ-643a) + GAP-A shared-channel
candidate_summary_source="e2_world_forward" (V3-EXQ-649); MECH-341 both sub-flavours
(entropy bonus + stratified select), within-class temperature FIXED (None legacy /
1.0 sampled -- 660 proved every positive T is byte-identical). P0 30 ep (instrument
OFF) + P1 60 ep (instrument ON), 200 steps/ep -- same budget + ENV_KWARGS as 660a
for direct comparability. The ONLY change vs 660a is the readout + the
readout-sensitivity gate.

PRE-REGISTERED ACCEPTANCE (paired by seed index, on the WINDOWED readout)
------------------------------------------------------------------------
  Delta(K)[seed] = windowed_rep_cond_entropy[SAMPLED,K,seed]
                 - windowed_rep_cond_entropy[LEGACY,K,seed]

  READINESS GATE 1 -- INPUT non-vacuity (carried from 660a, unchanged): mean
    distinct within-class reps AVAILABLE per committed class rises across K in the
    SAMPLED arms (monotone-nondecreasing AND avail(K_max)-avail(K_min) >=
    AVAIL_RISE_FLOOR). UNMET -> substrate_not_ready_requeue.

  READINESS GATE 2 -- READOUT sensitivity (NEW, the 660a fix; same statistic as
    C_GRADED): range of the SAMPLED-arm absolute windowed_rep_cond_entropy across K
    >= READOUT_SENSITIVITY_FLOOR. UNMET -> the windowed readout is still flat
    across the swept axis -> substrate_not_ready_requeue (NOT a weakens).

  C_GRADED (PRIMARY / load-bearing): for >= MIN_SEEDS_FOR_GRADED of 3 seeds, the
    per-seed Delta(K) (on the WINDOWED readout) is monotone-nondecreasing across K
    (within MONO_TOL) AND Delta(K_max)-Delta(K_min) >= C2_LIFT_MARGIN_NATS.

  OUTCOME MAP:
    INPUT non-vacuity UNMET           -> FAIL / non_contributory (substrate_not_ready_requeue)
    READOUT sensitivity UNMET         -> FAIL / non_contributory (substrate_not_ready_requeue)
    both readiness MET and C_GRADED   -> PASS / supports (within-class sub-axis
                                          load-bearing AND graded; v3_pending
                                          clearance candidate pending governance
                                          ratification; do NOT auto-flip the
                                          within-class default)
    both readiness MET but C_GRADED   -> FAIL / weakens (GENUINE negative: the
      fails                               windowed readout demonstrably registers K
                                          yet the marginal lift over legacy does NOT
                                          grow -- a fixed structural artifact, not a
                                          graded contribution)

Claims: [MECH-341]. experiment_purpose=evidence. supersedes=V3-EXQ-660a (the
readout was the bug that invalidated 660a's weakens; this is the corrected re-ask).
V3-EXQ-660 (the standing base) is NOT superseded.

See REE_assembly/evidence/planning/failure_autopsy_V3-EXQ-660a_2026-06-11.md,
REE_assembly/docs/architecture/mech_341_e3_score_diversity_preservation.md,
REE_assembly/evidence/planning/behavioral_diversity_isolation_plan.md GAP-B,
ree-v3/experiments/v3_exq_660a_mech341_within_class_pool_size_graded_confirmation.py.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from experiment_protocol import emit_outcome
from experiments._lib.arm_fingerprint import arm_cell
from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE = "v3_exq_660b_mech341_within_class_pool_size_graded_windowed_readout"
QUEUE_ID = "V3-EXQ-660b"
SUPERSEDES = "V3-EXQ-660a"  # readout was the bug invalidating 660a's weakens; corrected re-ask.
CLAIM_IDS: List[str] = ["MECH-341"]
EXPERIMENT_PURPOSE = "evidence"

# Graded axis: CEM candidate-pool size. K=32 reproduces 660. IDENTICAL to 660a.
POOL_SIZES: List[int] = [16, 32, 64, 128]

# Fixed within-class temperature for the SAMPLED lever state (None = legacy
# argmin control). 660 proved every positive T is byte-identical; value immaterial.
SAMPLED_WITHIN_CLASS_T: float = 1.0

# GAP-A shared-channel re-sourcing (V3-EXQ-649 PASS). ON for ALL arms.
CANDIDATE_SUMMARY_SOURCE = "e2_world_forward"

# modulatory-bias-selection-authority (V3-EXQ-643a-fixed substrate). ON for ALL arms.
USE_MODULATORY_SELECTION_AUTHORITY = True
MODULATORY_AUTHORITY_GAIN = 0.5

# ---- READOUT REDESIGN (the 660b change) ----
# Fixed-length tick window for the per-decision windowed readout. Removes the
# 660a tick-count confound (every window has the same tick budget) and preserves
# headroom (per-class within-window entropy capped at log(min(ticks_for_class,
# available_distinct)) -- rises with availability instead of saturating).
WINDOW_TICKS = 50
# A seed-arm needs at least this many COMPLETE windows for its windowed readout
# to be treated as well-estimated (informational flag; full P1 yields 20-220).
MIN_WINDOWS_FOR_READOUT = 3
# READINESS GATE 2: the SAMPLED-arm absolute windowed readout must MOVE across K
# by at least this range (nats). Same statistic as C_GRADED (windowed H), asserted
# as a RANGE on the positive-control arm. Sub-floor -> substrate_not_ready_requeue.
READOUT_SENSITIVITY_FLOOR = 0.05

# C_GRADED PRIMARY threshold: per-seed paired-lift growth across the pool-size
# sweep (on the WINDOWED readout). Margin matches the 660 lift-margin scale.
C2_LIFT_MARGIN_NATS = 0.05
MIN_SEEDS_FOR_GRADED = 2   # >= 2/3 seeds must show the graded dose-response
MONO_TOL = 1e-6            # tolerance for "monotone-nondecreasing"

# READINESS GATE 1 (INPUT non-vacuity, carried from 660a): mean distinct
# within-class representatives available per committed class must rise by at least
# this many from K_min to K_max in the SAMPLED arms.
AVAIL_RISE_FLOOR = 0.25

# Basic operativeness diagnostics (per-seed), retained from 660a.
WITHIN_CLASS_FIRE_FLOOR = 10
MULTI_REP_TICK_FLOOR = 5

# MECH-341 sub-flavour scale used in the entropy-ON arms (matches 660 / 660a).
MECH341_ENTROPY_BIAS_SCALE = 2.0

# SD-056 amend lever defaults applied uniformly across all arms (match 660a).
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
DRY_RUN_STEPS = 60  # >= WINDOW_TICKS so the dry run exercises the windowed path

MIN_SEEDS_PER_ARM_FOR_PASS = 2  # of 3 (operativeness majority checks)

# V_s (D) thresholds (minimal stack) -- match 660a.
VS_SNAPSHOT_REFRESH_THRESHOLD = 0.5
VS_E1_THRESHOLD = 0.4


# IDENTICAL to V3-EXQ-660a / 660 for direct manifest comparability.
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


def _build_arms() -> List[Dict[str, Any]]:
    arms: List[Dict[str, Any]] = []
    for k in POOL_SIZES:
        arms.append({
            "arm_id": f"ARM_LEGACY_K{k}",
            "label": f"legacy_argmin_pool{k}",
            "num_candidates": int(k),
            "within_class_temperature": None,
            "lever": "legacy",
        })
        arms.append({
            "arm_id": f"ARM_SAMPLED_K{k}",
            "label": f"within_class_sampled_pool{k}",
            "num_candidates": int(k),
            "within_class_temperature": SAMPLED_WITHIN_CLASS_T,
            "lever": "sampled",
        })
    return arms


ARMS: List[Dict[str, Any]] = _build_arms()


def _make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(seed=seed, **ENV_KWARGS)


def _make_agent(
    within_class_temperature: Optional[float],
    num_candidates: int,
    env: CausalGridWorldV2,
) -> REEAgent:
    """SD-056-amended 4-substrate baseline + modulatory authority + GAP-A
    shared-channel candidate_summary_source -- IDENTICAL to V3-EXQ-660a EXCEPT the
    per-arm CEM candidate-pool size. cfg.hippocampal.num_candidates is set AFTER
    REEConfig.from_dims (which does NOT assign it) and BEFORE REEAgent(cfg)."""
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
        use_support_preserving_cem=True,
        support_preserving_stratified_elites=True,
        support_preserving_ao_std_floor=0.2,
        support_preserving_min_first_action_classes=2,
        # B (MECH-341)
        use_e3_score_diversity=True,
        use_e3_diversity_entropy_bonus=True,
        use_e3_diversity_stratified_select=True,
        e3_diversity_entropy_bias_scale=MECH341_ENTROPY_BIAS_SCALE,
        e3_diversity_stratified_within_class_temperature=within_class_temperature,
        # GAP-A shared-channel candidate-summary re-sourcing (V3-EXQ-649 PASS).
        candidate_summary_source=CANDIDATE_SUMMARY_SOURCE,
        # modulatory-bias-selection-authority (V3-EXQ-643a-fixed). ON for all arms.
        use_modulatory_selection_authority=USE_MODULATORY_SELECTION_AUTHORITY,
        modulatory_authority_gain=MODULATORY_AUTHORITY_GAIN,
        # C (MECH-313)
        use_noise_floor=True,
        noise_floor_alpha=0.1,
        # D (V_s minimal)
        use_per_stream_vs=True,
        use_vs_rollout_gating=True,
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
    cfg.hippocampal.num_candidates = int(num_candidates)
    return REEAgent(cfg)


# ---------------------------------------------------------------------------
# Per-tick measurement helpers (rep signature / class -- identical to 660a)
# ---------------------------------------------------------------------------


def _first_action_class(traj) -> int:
    return int(
        traj.actions[:, 0, :].argmax(dim=-1).detach().reshape(-1)[0].item()
    )


def _rep_signature(traj) -> Tuple[int, ...]:
    """Within-class representative discriminator: argmax of each action step
    AFTER the first (members of one first-action class share step 0)."""
    actions = traj.actions
    if actions.dim() < 3 or actions.shape[1] < 2:
        return ()
    horizon = int(actions.shape[1])
    sig: List[int] = []
    for t in range(1, horizon):
        sig.append(int(actions[:, t, :].argmax(dim=-1).detach().reshape(-1)[0].item()))
    return tuple(sig)


def _obs_harm(obs_dict):
    h = obs_dict.get("harm_obs")
    return h.float().unsqueeze(0) if h is not None else None


def _obs_harm_a(obs_dict):
    h = obs_dict.get("harm_obs_a")
    return h.float().unsqueeze(0) if h is not None else None


def _obs_harm_history(obs_dict):
    h = obs_dict.get("harm_history")
    return h.float().unsqueeze(0) if h is not None else None


def _entropy_from_counts(counts: Dict[Any, int]) -> float:
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


def _conditional_entropy(pairs: List[Tuple[int, Tuple[int, ...]]]) -> float:
    """H(rep_signature | committed_class), weighted by committed-class freq."""
    if not pairs:
        return 0.0
    by_class: Dict[int, Dict[Tuple[int, ...], int]] = {}
    for cls, sig in pairs:
        by_class.setdefault(cls, {})
        by_class[cls][sig] = by_class[cls].get(sig, 0) + 1
    total = len(pairs)
    h_cond = 0.0
    for cls, sig_counts in by_class.items():
        n_c = sum(sig_counts.values())
        h_cond += (n_c / total) * _entropy_from_counts(sig_counts)
    return float(h_cond)


# ---------------------------------------------------------------------------
# READOUT REDESIGN helpers (the 660b change): per-decision / windowed metrics
# ---------------------------------------------------------------------------


def _windows(seq: List[Any], window: int) -> List[List[Any]]:
    """Consecutive COMPLETE windows of `window` elements (trailing partial dropped)."""
    if window <= 0 or len(seq) < window:
        return []
    return [seq[i:i + window] for i in range(0, len(seq) - window + 1, window)]


def _windowed_conditional_entropy(
    selected_pairs: List[Tuple[int, Tuple[int, ...]]], window: int
) -> Tuple[float, int]:
    """PRIMARY readout: mean over fixed-length windows of H(rep|class) computed
    PER WINDOW. Fixed window removes the 660a tick-count confound; a small window
    preserves headroom (per-class within-window entropy capped at
    log(min(ticks_for_class, available_distinct)) -- rises with availability)."""
    wins = _windows(selected_pairs, window)
    if not wins:
        return 0.0, 0
    hs = [_conditional_entropy(w) for w in wins]
    return float(sum(hs) / len(hs)), len(wins)


def _windowed_mean_distinct_selected(
    selected_pairs: List[Tuple[int, Tuple[int, ...]]], window: int
) -> float:
    """Interpretability secondary: mean over windows of the mean-over-committed-
    classes distinct selected within-class rep signatures (the count the windowed
    entropy reflects)."""
    wins = _windows(selected_pairs, window)
    if not wins:
        return 0.0
    vals: List[float] = []
    for w in wins:
        by_class: Dict[int, set] = {}
        for cls, sig in w:
            by_class.setdefault(cls, set()).add(sig)
        if by_class:
            vals.append(sum(len(s) for s in by_class.values()) / len(by_class))
    return float(sum(vals) / len(vals)) if vals else 0.0


def _windowed_efficiency(
    triples: List[Tuple[int, Tuple[int, ...], FrozenSet[Tuple[int, ...]]]], window: int
) -> float:
    """Normalised [0,1] secondary readout: per committed class per window,
    (distinct selected within-class reps) / (distinct AVAILABLE within-class reps,
    pooled over the window). Averaged over classes (per window) then over windows.
    Bounded -> cannot tick-count-confound or saturate; measures what FRACTION of
    available within-class diversity the lever actually exploits."""
    wins = _windows(triples, window)
    if not wins:
        return 0.0
    effs: List[float] = []
    for w in wins:
        sel_by_class: Dict[int, set] = {}
        avail_by_class: Dict[int, set] = {}
        for cls, sig, avail in w:
            sel_by_class.setdefault(cls, set()).add(sig)
            avail_by_class.setdefault(cls, set()).update(avail)
        per_class: List[float] = []
        for cls, sels in sel_by_class.items():
            n_avail = len(avail_by_class.get(cls, set()))
            if n_avail > 0:
                per_class.append(len(sels) / n_avail)
        if per_class:
            effs.append(sum(per_class) / len(per_class))
    return float(sum(effs) / len(effs)) if effs else 0.0


# ---------------------------------------------------------------------------


def _run_seed_arm(
    arm: Dict[str, Any],
    seed: int,
    p0_episodes: int,
    p1_episodes: int,
    steps_per_episode: int,
) -> Dict[str, Any]:
    within_class_temperature = arm.get("within_class_temperature")
    num_candidates = int(arm["num_candidates"])

    config_slice = {
        "arm_id": arm["arm_id"],
        "within_class_temperature": within_class_temperature,
        "num_candidates": num_candidates,
        "candidate_summary_source": CANDIDATE_SUMMARY_SOURCE,
        "use_modulatory_selection_authority": USE_MODULATORY_SELECTION_AUTHORITY,
        "modulatory_authority_gain": MODULATORY_AUTHORITY_GAIN,
        "mech341_entropy_bias_scale": MECH341_ENTROPY_BIAS_SCALE,
        "window_ticks": int(WINDOW_TICKS),
        "p0_episodes": int(p0_episodes),
        "p1_episodes": int(p1_episodes),
        "steps_per_episode": int(steps_per_episode),
        "env_kwargs": dict(ENV_KWARGS),
    }

    with arm_cell(
        seed,
        config_slice=config_slice,
        script_path=Path(__file__),
    ) as cell:
        env = _make_env(seed)
        agent = _make_agent(within_class_temperature, num_candidates, env)
        agent.eval()

        total_train_eps = p0_episodes + p1_episodes
        error_note: Optional[str] = None
        n_p0_ticks = 0
        n_p1_ticks = 0
        n_p1_pre_ge2 = 0
        n_p1_pre_eq1 = 0
        committed_classes_p1: Dict[int, int] = {}

        # Ordered P1 selection record (one entry per P1 tick, execution order):
        selected_pairs: List[Tuple[int, Tuple[int, ...]]] = []
        # Parallel record carrying the available within-class signature set per tick
        # (for the windowed efficiency readout).
        avail_triples: List[Tuple[int, Tuple[int, ...], FrozenSet[Tuple[int, ...]]]] = []
        n_multi_rep_ticks = 0

        # Availability accumulators (INPUT non-vacuity, byte-comparable to 660a).
        sum_distinct_within_class_reps = 0
        n_avail_ticks = 0
        sum_pool_size = 0
        max_pool_size = 0
        n_pool_obs = 0

        n_within_class_sampled_total = 0
        n_stratified_fired_total = 0
        n_authority_normalized_total = 0

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
                        _first_action_class(t) for t in candidates
                    })
                    sum_pool_size += len(candidates)
                    max_pool_size = max(max_pool_size, len(candidates))
                    n_pool_obs += 1

                action = agent.select_action(candidates, ticks)
                if not torch.isfinite(action).all():
                    if error_note is None:
                        error_note = (
                            f"non-finite action at arm={arm['arm_id']} seed={seed} "
                            f"phase={phase_label} ep={ep} step={_step}"
                        )
                    break

                if is_p1:
                    committed_class = int(action[0].argmax().item())
                    committed_classes_p1[committed_class] = (
                        committed_classes_p1.get(committed_class, 0) + 1
                    )

                    avail_set: FrozenSet[Tuple[int, ...]] = frozenset()
                    if candidates:
                        pool_sigs_in_class = {
                            _rep_signature(t)
                            for t in candidates
                            if _first_action_class(t) == committed_class
                        }
                        avail_set = frozenset(pool_sigs_in_class)
                        # AVAILABILITY statistic (INPUT non-vacuity gate, == 660a):
                        sum_distinct_within_class_reps += len(pool_sigs_in_class)
                        n_avail_ticks += 1
                        if len(pool_sigs_in_class) >= 2:
                            n_multi_rep_ticks += 1

                    sel_traj = getattr(agent.e3, "_last_selected_trajectory", None)
                    if sel_traj is not None:
                        sel_cls = _first_action_class(sel_traj)
                        sel_sig = _rep_signature(sel_traj)
                        selected_pairs.append((sel_cls, sel_sig))
                        avail_triples.append((sel_cls, sel_sig, avail_set))

                    pre_count = len(pre_e3_classes)
                    if pre_count >= 2:
                        n_p1_pre_ge2 += 1
                    elif pre_count == 1:
                        n_p1_pre_eq1 += 1

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

        committed_class_entropy = _entropy_from_counts(committed_classes_p1)

        # PRIMARY readout: windowed H(rep|class) averaged across complete windows.
        windowed_rep_cond_entropy, n_windows = _windowed_conditional_entropy(
            selected_pairs, WINDOW_TICKS
        )
        # Secondary readouts:
        windowed_mean_distinct_selected = _windowed_mean_distinct_selected(
            selected_pairs, WINDOW_TICKS
        )
        windowed_efficiency = _windowed_efficiency(avail_triples, WINDOW_TICKS)
        # The 660a phase-aggregate metric, carried for the saturation contrast:
        phase_aggregate_rep_cond_entropy = _conditional_entropy(selected_pairs)

        mean_distinct_within_class_reps = (
            float(sum_distinct_within_class_reps / n_avail_ticks)
            if n_avail_ticks > 0 else 0.0
        )
        realized_pool_size_mean = (
            float(sum_pool_size / n_pool_obs) if n_pool_obs > 0 else 0.0
        )

        seed_substrate_ready = bool(frac_pre_ge2 > 0.3)
        within_class_branch_active = bool(
            n_within_class_sampled_total >= WITHIN_CLASS_FIRE_FLOOR
        )
        multi_rep_available = bool(n_multi_rep_ticks >= MULTI_REP_TICK_FLOOR)
        readout_well_estimated = bool(n_windows >= MIN_WINDOWS_FOR_READOUT)

        row = {
            "arm_id": arm["arm_id"],
            "lever": arm["lever"],
            "num_candidates": int(num_candidates),
            "seed": int(seed),
            "within_class_temperature": within_class_temperature,
            "p0_episodes_run": int(min(p0_episodes, total_train_eps)),
            "p1_episodes_run": int(max(0, total_train_eps - p0_episodes)),
            "n_p0_ticks": int(n_p0_ticks),
            "n_p1_ticks": int(n_p1_ticks),
            "n_p1_pre_ge2": int(n_p1_pre_ge2),
            "n_p1_pre_eq1": int(n_p1_pre_eq1),
            "frac_pre_ge2": round(frac_pre_ge2, 6),
            "committed_classes_p1_counts": {
                str(k): int(v) for k, v in sorted(committed_classes_p1.items())
            },
            "n_unique_committed_classes": int(len(committed_classes_p1)),
            # PRIMARY readout (windowed):
            "windowed_rep_cond_entropy_nats": round(windowed_rep_cond_entropy, 6),
            "n_windows": int(n_windows),
            "window_ticks": int(WINDOW_TICKS),
            "readout_well_estimated": readout_well_estimated,
            # Secondary readouts:
            "windowed_mean_distinct_selected_reps": round(
                windowed_mean_distinct_selected, 6
            ),
            "windowed_within_class_efficiency": round(windowed_efficiency, 6),
            # 660a phase-aggregate metric (saturation contrast):
            "phase_aggregate_rep_cond_entropy_nats": round(
                phase_aggregate_rep_cond_entropy, 6
            ),
            "n_selected_pairs": int(len(selected_pairs)),
            "n_multi_rep_ticks": int(n_multi_rep_ticks),
            # AVAILABILITY statistic (INPUT non-vacuity gate, == 660a):
            "mean_distinct_within_class_reps": round(
                mean_distinct_within_class_reps, 6
            ),
            "n_avail_ticks": int(n_avail_ticks),
            # realized pool size (confirms K is honored):
            "realized_pool_size_mean": round(realized_pool_size_mean, 4),
            "realized_pool_size_max": int(max_pool_size),
            # SECONDARY / negative-control readout:
            "committed_class_entropy_nats": round(committed_class_entropy, 6),
            # MECH-341 within-class firing diagnostics:
            "mech341_n_within_class_sampled": int(n_within_class_sampled_total),
            "mech341_n_stratified_fired": int(n_stratified_fired_total),
            "mech341_n_authority_normalized": int(n_authority_normalized_total),
            # Per-seed flags:
            "within_class_branch_active": within_class_branch_active,
            "multi_rep_available": multi_rep_available,
            "seed_substrate_ready": seed_substrate_ready,
            "error_note": error_note,
        }
        cell.stamp(row)

    return row


def _interpret_arm(seed_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    completed = [r for r in seed_rows if r["error_note"] is None]
    n_seeds_completed = len(completed)

    win_ent = [r["windowed_rep_cond_entropy_nats"] for r in completed]
    mean_win_ent = sum(win_ent) / len(win_ent) if win_ent else 0.0
    eff = [r["windowed_within_class_efficiency"] for r in completed]
    mean_eff = sum(eff) / len(eff) if eff else 0.0
    win_sel = [r["windowed_mean_distinct_selected_reps"] for r in completed]
    mean_win_sel = sum(win_sel) / len(win_sel) if win_sel else 0.0
    phase_agg = [r["phase_aggregate_rep_cond_entropy_nats"] for r in completed]
    mean_phase_agg = sum(phase_agg) / len(phase_agg) if phase_agg else 0.0
    avail = [r["mean_distinct_within_class_reps"] for r in completed]
    mean_avail = sum(avail) / len(avail) if avail else 0.0
    pool_means = [r["realized_pool_size_mean"] for r in completed]
    mean_realized_pool = sum(pool_means) / len(pool_means) if pool_means else 0.0

    n_substrate_ready = sum(1 for r in completed if r["seed_substrate_ready"])
    n_within_class_active = sum(
        1 for r in completed if r["within_class_branch_active"]
    )
    n_multi_rep = sum(1 for r in completed if r["multi_rep_available"])
    n_readout_well = sum(1 for r in completed if r["readout_well_estimated"])

    return {
        "n_seeds_completed": int(n_seeds_completed),
        "mean_windowed_rep_cond_entropy_nats": round(mean_win_ent, 6),
        "mean_windowed_within_class_efficiency": round(mean_eff, 6),
        "mean_windowed_mean_distinct_selected_reps": round(mean_win_sel, 6),
        "mean_phase_aggregate_rep_cond_entropy_nats": round(mean_phase_agg, 6),
        "mean_distinct_within_class_reps": round(mean_avail, 6),
        "mean_realized_pool_size": round(mean_realized_pool, 4),
        "n_seeds_substrate_ready": int(n_substrate_ready),
        "majority_substrate_ready": bool(
            n_substrate_ready >= MIN_SEEDS_PER_ARM_FOR_PASS
        ),
        "n_seeds_within_class_active": int(n_within_class_active),
        "n_seeds_multi_rep_available": int(n_multi_rep),
        "n_seeds_readout_well_estimated": int(n_readout_well),
    }


def _is_monotone_nondecreasing(values: List[float], tol: float = MONO_TOL) -> bool:
    return all(values[i + 1] >= values[i] - tol for i in range(len(values) - 1))


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
            f"(K={arm['num_candidates']}, within_T={arm['within_class_temperature']}, "
            f"window_ticks={WINDOW_TICKS}, P0={p0_episodes} ep, P1={p1_episodes} ep, "
            f"steps={steps_per_episode}, "
            f"candidate_summary_source={CANDIDATE_SUMMARY_SOURCE}, "
            f"authority=ON gain={MODULATORY_AUTHORITY_GAIN}, dry_run={dry_run})",
            flush=True,
        )
        seed_rows: List[Dict[str, Any]] = []
        for s in seeds:
            print(f"Seed {s} Condition {arm['label']}", flush=True)
            row = _run_seed_arm(arm, s, p0_episodes, p1_episodes, steps_per_episode)
            seed_rows.append(row)
            verdict = "PASS" if row["error_note"] is None else "FAIL"
            print(
                f"  arm={arm['arm_id']} seed={s} K={arm['num_candidates']} "
                f"realized_pool_mean={row['realized_pool_size_mean']} "
                f"mean_distinct_reps={row['mean_distinct_within_class_reps']} "
                f"n_windows={row['n_windows']} "
                f"windowed_rep_cond_entropy={row['windowed_rep_cond_entropy_nats']} "
                f"windowed_efficiency={row['windowed_within_class_efficiency']} "
                f"phase_aggregate_entropy={row['phase_aggregate_rep_cond_entropy_nats']}",
                flush=True,
            )
            print(f"verdict: {verdict}", flush=True)
        cross = _interpret_arm(seed_rows)
        arms_out.append({
            "arm_id": arm["arm_id"],
            "label": arm["label"],
            "lever": arm["lever"],
            "num_candidates": int(arm["num_candidates"]),
            "within_class_temperature": arm.get("within_class_temperature"),
            "per_seed_results": seed_rows,
            "cross_seed_interpretation": cross,
        })

    by_id = {a["arm_id"]: a for a in arms_out}

    def _windowed_by_seed(arm) -> Dict[int, float]:
        return {
            int(r["seed"]): r["windowed_rep_cond_entropy_nats"]
            for r in arm["per_seed_results"]
            if r["error_note"] is None
        }

    # ---- Realized-pool-size honored check (K is actually applied) ----
    realized_pool_by_K = {
        k: by_id[f"ARM_SAMPLED_K{k}"]["cross_seed_interpretation"][
            "mean_realized_pool_size"
        ]
        for k in POOL_SIZES
    }
    pool_size_honored = _is_monotone_nondecreasing(
        [realized_pool_by_K[k] for k in POOL_SIZES]
    ) and (
        realized_pool_by_K[POOL_SIZES[-1]] > realized_pool_by_K[POOL_SIZES[0]]
    )

    # ---- READINESS GATE 1: INPUT availability rises across K (== 660a) ----
    avail_by_K_sampled = {
        k: by_id[f"ARM_SAMPLED_K{k}"]["cross_seed_interpretation"][
            "mean_distinct_within_class_reps"
        ]
        for k in POOL_SIZES
    }
    avail_series = [avail_by_K_sampled[k] for k in POOL_SIZES]
    avail_rise = float(avail_series[-1] - avail_series[0])
    non_vacuity_met = bool(
        _is_monotone_nondecreasing(avail_series)
        and avail_rise >= AVAIL_RISE_FLOOR
    )

    # ---- READINESS GATE 2 (NEW, the 660a fix): the SAMPLED-arm WINDOWED readout
    # MOVES across K. Same statistic C_GRADED routes on (windowed H), asserted as a
    # RANGE on the positive-control arm -- NOT a magnitude proxy. ----
    windowed_by_K_sampled = {
        k: by_id[f"ARM_SAMPLED_K{k}"]["cross_seed_interpretation"][
            "mean_windowed_rep_cond_entropy_nats"
        ]
        for k in POOL_SIZES
    }
    windowed_sampled_series = [windowed_by_K_sampled[k] for k in POOL_SIZES]
    windowed_sampled_range = float(
        max(windowed_sampled_series) - min(windowed_sampled_series)
    )
    readout_sensitive = bool(windowed_sampled_range >= READOUT_SENSITIVITY_FLOOR)

    # ---- C_GRADED: per-seed paired-lift dose-response across K (WINDOWED) ----
    legacy_win_by_K = {k: _windowed_by_seed(by_id[f"ARM_LEGACY_K{k}"]) for k in POOL_SIZES}
    sampled_win_by_K = {k: _windowed_by_seed(by_id[f"ARM_SAMPLED_K{k}"]) for k in POOL_SIZES}

    delta_by_K_by_seed: Dict[int, Dict[int, float]] = {}  # seed -> {K: delta}
    for k in POOL_SIZES:
        leg = legacy_win_by_K[k]
        sam = sampled_win_by_K[k]
        for s in seeds:
            if s in leg and s in sam:
                delta_by_K_by_seed.setdefault(s, {})[k] = float(sam[s] - leg[s])

    per_seed_graded: Dict[int, Dict[str, Any]] = {}
    n_seeds_graded = 0
    for s in seeds:
        dmap = delta_by_K_by_seed.get(s, {})
        if len(dmap) != len(POOL_SIZES):
            per_seed_graded[s] = {
                "complete": False, "monotone": False, "margin": None, "graded": False,
            }
            continue
        deltas = [dmap[k] for k in POOL_SIZES]
        monotone = _is_monotone_nondecreasing(deltas)
        margin = float(deltas[-1] - deltas[0])
        graded = bool(monotone and margin >= C2_LIFT_MARGIN_NATS)
        per_seed_graded[s] = {
            "complete": True,
            "deltas_by_K": {str(k): round(dmap[k], 6) for k in POOL_SIZES},
            "monotone": bool(monotone),
            "margin": round(margin, 6),
            "graded": graded,
        }
        if graded:
            n_seeds_graded += 1

    c_graded = bool(n_seeds_graded >= MIN_SEEDS_FOR_GRADED)

    # ---- Phase-aggregate-saturation contrast (reported; NOT gating). The 660a
    # readout: SAMPLED phase-aggregate entropy range across K (expected ~flat). ----
    phase_agg_by_K_sampled = {
        k: by_id[f"ARM_SAMPLED_K{k}"]["cross_seed_interpretation"][
            "mean_phase_aggregate_rep_cond_entropy_nats"
        ]
        for k in POOL_SIZES
    }
    phase_agg_sampled_range = float(
        max(phase_agg_by_K_sampled.values()) - min(phase_agg_by_K_sampled.values())
    )

    # ---- Outcome map ----
    if not non_vacuity_met:
        outcome_label = "FAIL"
        mech341_direction = "non_contributory"
        interpretation_label = (
            "FAIL_substrate_not_ready_requeue_within_class_availability_flat_across_pool_size"
        )
    elif not readout_sensitive:
        outcome_label = "FAIL"
        mech341_direction = "non_contributory"
        interpretation_label = (
            "FAIL_substrate_not_ready_requeue_windowed_readout_insensitive_to_pool_size"
        )
    elif c_graded:
        outcome_label = "PASS"
        mech341_direction = "supports"
        interpretation_label = (
            "PASS_graded_within_class_representative_diversity_dose_response_windowed_readout"
        )
    else:
        outcome_label = "FAIL"
        mech341_direction = "weakens"
        interpretation_label = (
            "FAIL_lift_not_graded_under_sensitive_windowed_readout_fixed_structural_artifact"
        )

    overall_direction = mech341_direction

    total_seeds = len(ARMS) * len(seeds)
    total_completed = sum(
        a["cross_seed_interpretation"]["n_seeds_completed"] for a in arms_out
    )

    return {
        "outcome": outcome_label,
        "overall_direction": overall_direction,
        "evidence_direction_per_claim": {"MECH-341": mech341_direction},
        "interpretation_label": interpretation_label,
        "seeds": seeds,
        "pool_sizes": list(POOL_SIZES),
        "n_arms": len(arms_out),
        "total_seeds_attempted": int(total_seeds),
        "total_seeds_completed": int(total_completed),
        "p0_episodes": int(p0_episodes),
        "p1_episodes": int(p1_episodes),
        "steps_per_episode": int(steps_per_episode),
        "decision_rule_thresholds": {
            "window_ticks": int(WINDOW_TICKS),
            "min_windows_for_readout": int(MIN_WINDOWS_FOR_READOUT),
            "readout_sensitivity_floor": float(READOUT_SENSITIVITY_FLOOR),
            "c2_lift_margin_nats": float(C2_LIFT_MARGIN_NATS),
            "min_seeds_for_graded": int(MIN_SEEDS_FOR_GRADED),
            "avail_rise_floor": float(AVAIL_RISE_FLOOR),
            "mono_tol": float(MONO_TOL),
            "mech341_entropy_bias_scale": float(MECH341_ENTROPY_BIAS_SCALE),
            "candidate_summary_source": CANDIDATE_SUMMARY_SOURCE,
            "use_modulatory_selection_authority": bool(
                USE_MODULATORY_SELECTION_AUTHORITY
            ),
            "modulatory_authority_gain": float(MODULATORY_AUTHORITY_GAIN),
            "sampled_within_class_temperature": float(SAMPLED_WITHIN_CLASS_T),
        },
        "acceptance_criteria": {
            "non_vacuity_within_class_availability_rises": bool(non_vacuity_met),
            "non_vacuity_avail_by_K_sampled": {
                str(k): round(float(v), 6) for k, v in avail_by_K_sampled.items()
            },
            "non_vacuity_avail_rise": round(avail_rise, 6),
            "readout_sensitive_windowed_moves_across_K": bool(readout_sensitive),
            "readout_windowed_by_K_sampled": {
                str(k): round(float(v), 6) for k, v in windowed_by_K_sampled.items()
            },
            "readout_windowed_sampled_range": round(windowed_sampled_range, 6),
            "C_GRADED_paired_lift_dose_response": bool(c_graded),
            "C_GRADED_n_seeds": int(n_seeds_graded),
            "C_GRADED_per_seed": {str(s): per_seed_graded[s] for s in seeds},
            "phase_aggregate_sampled_by_K": {
                str(k): round(float(v), 6) for k, v in phase_agg_by_K_sampled.items()
            },
            "phase_aggregate_sampled_range_contrast": round(phase_agg_sampled_range, 6),
            "realized_pool_size_honored": bool(pool_size_honored),
            "realized_pool_by_K_sampled": {
                str(k): round(float(v), 4) for k, v in realized_pool_by_K.items()
            },
        },
        "interpretation": {
            "label": interpretation_label,
            "preconditions": [
                {
                    "name": "within_class_representative_availability_rises_with_pool_size",
                    "description": (
                        "INPUT non-vacuity: mean distinct within-class rep "
                        "signatures AVAILABLE per committed class (a count/range "
                        "statistic) rises monotonically across the swept CEM pool "
                        "sizes K in the SAMPLED arms. Below-floor -> the graded axis "
                        "does not move the substrate's within-class availability -> "
                        "substrate_not_ready_requeue, NOT a weakens."
                    ),
                    "measured": round(avail_rise, 6),
                    "threshold": float(AVAIL_RISE_FLOOR),
                    "direction": "lower",
                    "control": "SAMPLED arms availability across K (pool sizes 16->128)",
                    "met": bool(non_vacuity_met),
                },
                {
                    "name": "windowed_readout_sensitive_to_pool_size",
                    "description": (
                        "READOUT sensitivity (the V3-EXQ-660a fix): the windowed "
                        "within-class rep conditional entropy -- the SAME "
                        "per-decision statistic C_GRADED routes its LIFT on, asserted "
                        "here as a RANGE/spread (NOT a magnitude/mean-abs proxy) -- "
                        "MOVES across the swept pool sizes K in the SAMPLED "
                        "positive-control arm. 660a's phase-aggregate readout was "
                        "blind to K (saturated + tick-count-confounded); this gate "
                        "requires the windowed readout to register K before any "
                        "weakens verdict is reachable. Below-floor -> the readout is "
                        "still insensitive at this scale -> substrate_not_ready_requeue, "
                        "NOT a weakens."
                    ),
                    "measured": round(windowed_sampled_range, 6),
                    "threshold": float(READOUT_SENSITIVITY_FLOOR),
                    "direction": "lower",
                    "control": (
                        "SAMPLED arms windowed-H range across K=16..128 (positive "
                        "control: the lever that actively samples diverse within-class reps)"
                    ),
                    "met": bool(readout_sensitive),
                },
                {
                    "name": "cem_pool_size_honored",
                    "description": (
                        "realized candidate-pool size rises with the configured "
                        "num_candidates K, confirming cfg.hippocampal.num_candidates "
                        "is honored (not silently overridden by from_dims/__init__)."
                    ),
                    "measured": round(
                        float(realized_pool_by_K[POOL_SIZES[-1]]
                              - realized_pool_by_K[POOL_SIZES[0]]),
                        4,
                    ),
                    "threshold": 1.0,
                    "direction": "lower",
                    "control": "SAMPLED arms K=16 vs K=128 realized pool size",
                    "met": bool(pool_size_honored),
                },
            ],
            "criteria": [
                {
                    "name": "C_GRADED_paired_windowed_lift_monotone_in_pool_size",
                    "load_bearing": True,
                    "passed": bool(c_graded),
                },
            ],
            "criteria_non_degenerate": {
                "input_non_vacuity": bool(non_vacuity_met),
                "readout_sensitive": bool(readout_sensitive),
                "pool_size_honored": bool(pool_size_honored),
                # C_GRADED is only a meaningful (non-degenerate) test once BOTH
                # readiness gates hold (input moves AND the readout registers it).
                "C_GRADED": bool(non_vacuity_met and readout_sensitive),
            },
        },
        "interpretation_grid": {
            "FAIL_substrate_not_ready_requeue_availability_flat": (
                "Available distinct within-class representatives did NOT rise across "
                "the CEM pool-size sweep (or pool size was not honored). The graded "
                "axis has no leverage at this scale. NOT an MECH-341 falsification; "
                "re-queue at a scale/substrate where pool size moves within-class "
                "availability (substrate_not_ready_requeue)."
            ),
            "FAIL_substrate_not_ready_requeue_readout_insensitive": (
                "Availability rose, but the WINDOWED readout itself was flat across K "
                "in the SAMPLED arm (range < floor) -- it cannot register the swept "
                "axis at this scale (the V3-EXQ-660a failure mode, now caught as a "
                "readiness gate instead of mislabelled a weakens). NOT an MECH-341 "
                "falsification; re-queue with a window/scale where the readout is "
                "sensitive (substrate_not_ready_requeue)."
            ),
            "PASS_graded_dose_response_windowed": (
                "Both readiness gates hold AND the per-seed paired WINDOWED lift "
                "(sampled - legacy) GROWS monotonically with available within-class "
                "diversity (CEM pool size): a positive dose-response under a "
                "demonstrably-sensitive readout. The MECH-341 within-class sub-axis "
                "is load-bearing AND graded. Route to /governance: MECH-341 v3_pending "
                "clearance candidate (supports). Do NOT auto-flip the within-class "
                "default (governance ratification gate)."
            ),
            "FAIL_lift_not_graded_under_sensitive_readout": (
                "Both readiness gates hold (availability rose AND the windowed readout "
                "demonstrably moves with K), yet the per-seed paired WINDOWED lift "
                "(sampled - legacy) did NOT grow / was non-monotone. With the 660a "
                "measurement defect ruled out by the readout-sensitivity gate, this is "
                "a GENUINE negative for the MECH-341 within-class sub-axis as a GRADED "
                "lever: the marginal contribution over legacy argmin is a fixed "
                "structural artifact independent of available within-class diversity. "
                "Route to /governance (weakens) + propagate to Q-054 + the arc_062 "
                "GAP-B wiring decision."
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
    manifest = {
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "queue_id": QUEUE_ID,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": timestamp_utc,
        "outcome": result["outcome"],
        "evidence_direction": result["overall_direction"],
        "evidence_direction_per_claim": result["evidence_direction_per_claim"],
        "evidence_direction_note": (
            f"V3-EXQ-660b MECH-341 within-class-representative-diversity GRADED "
            f"CONFIRMATION, READOUT REDESIGN of V3-EXQ-660a (supersedes 660a; routed "
            f"by failure_autopsy_V3-EXQ-660a_2026-06-11). Same scientific question + "
            f"same agent/arms/env/budget; the ONLY change is the readout + a "
            f"readout-sensitivity gate. 660a's phase-aggregate H(rep|class) saturated "
            f"and was tick-count-confounded, so its weakens self-route was a "
            f"measurement_test_design_defect (non_contributory). 660b replaces it with "
            f"a per-decision WINDOWED H(rep|class) (mean over fixed {WINDOW_TICKS}-tick "
            f"windows -- removes the tick-count confound, preserves headroom) + a "
            f"normalised [0,1] selected/available efficiency, and gates C_GRADED behind "
            f"TWO readiness checks: (1) INPUT availability rises across K (== 660a) AND "
            f"(2) the SAMPLED-arm windowed readout MOVES across K (range >= "
            f"{READOUT_SENSITIVITY_FLOOR} nats -- the SAME statistic C_GRADED routes "
            f"its lift on, asserted as a RANGE on the positive control). Either "
            f"readiness UNMET -> substrate_not_ready_requeue (non_contributory, NOT a "
            f"weakens). Only with BOTH readiness MET is a flat lift a GENUINE weakens "
            f"(readout demonstrably sensitive). C_GRADED (PRIMARY, on the WINDOWED "
            f"readout): per-seed paired lift Delta(K)=sampled-legacy monotone-"
            f"nondecreasing across K AND Delta(K_max)-Delta(K_min) >= "
            f"{C2_LIFT_MARGIN_NATS} nats on >= {MIN_SEEDS_FOR_GRADED}/3 seeds. "
            f"interpretation_label={result['interpretation_label']}. non_vacuity="
            f"{result['acceptance_criteria']['non_vacuity_within_class_availability_rises']}, "
            f"readout_sensitive="
            f"{result['acceptance_criteria']['readout_sensitive_windowed_moves_across_K']}, "
            f"C_GRADED={result['acceptance_criteria']['C_GRADED_paired_lift_dose_response']}, "
            f"pool_honored="
            f"{result['acceptance_criteria']['realized_pool_size_honored']}. "
            f"experiment_purpose=evidence; MECH-341 is a v3_pending candidate; "
            f"supersedes V3-EXQ-660a (V3-EXQ-660 stays standing)."
        ),
        "dry_run": bool(dry_run),
        "env_kwargs": dict(ENV_KWARGS),
        "config_summary": {
            "graded_axis": "cem_pool_size (cfg.hippocampal.num_candidates)",
            "pool_sizes": list(POOL_SIZES),
            "readout_redesign": (
                "windowed H(rep|class) over fixed-length tick windows (primary) + "
                "selected/available within-class efficiency in [0,1] (secondary); "
                "phase-aggregate H(rep|class) carried for the 660a saturation contrast"
            ),
            "window_ticks": int(WINDOW_TICKS),
            "readout_sensitivity_gate": (
                "sampled windowed-H range across K >= "
                f"{READOUT_SENSITIVITY_FLOOR} (same statistic as C_GRADED)"
            ),
            "within_class_temperature_fixed": {
                "legacy": None, "sampled": SAMPLED_WITHIN_CLASS_T,
            },
            "vs_stack": "minimal (use_per_stream_vs + use_vs_rollout_gating)",
            "mech341_sub_flavours": "both (entropy_bonus + stratified_select)",
            "mech341_entropy_bias_scale": MECH341_ENTROPY_BIAS_SCALE,
            "candidate_summary_source": CANDIDATE_SUMMARY_SOURCE,
            "use_modulatory_selection_authority": USE_MODULATORY_SELECTION_AUTHORITY,
            "modulatory_authority_gain": MODULATORY_AUTHORITY_GAIN,
            "primary_readout": "windowed_rep_cond_entropy_nats (mean over windows of H(rep|class))",
            "non_vacuity_statistic": "mean_distinct_within_class_reps (availability)",
            "z_goal_enabled": True,
            "drive_weight": 2.0,
            "alpha_world": 0.9,
            "reef_enabled": True,
            "reef_bipartite_layout": True,
            "sd056_amend_active": True,
            "elite_fraction": "default 0.2 (untouched)",
            "use_differentiable_cem": "NOT FLIPPED (default False; SD-055 safety note)",
        },
        "result": result,
    }
    if SUPERSEDES is not None:
        manifest["supersedes"] = SUPERSEDES
    return manifest


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
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{manifest['run_id']}.json"
    if args.dry_run:
        out_path = out_dir / f"_dry_{manifest['run_id']}.json"

    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"manifest: {out_path}", flush=True)
    if not args.dry_run:
        print(f"Result written to: {out_path}", flush=True)
    print(
        f"outcome: {result['outcome']} "
        f"completed={result['total_seeds_completed']}/{result['total_seeds_attempted']} "
        f"non_vacuity={result['acceptance_criteria']['non_vacuity_within_class_availability_rises']} "
        f"readout_sensitive={result['acceptance_criteria']['readout_sensitive_windowed_moves_across_K']} "
        f"C_GRADED={result['acceptance_criteria']['C_GRADED_paired_lift_dose_response']} "
        f"pool_honored={result['acceptance_criteria']['realized_pool_size_honored']} "
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
