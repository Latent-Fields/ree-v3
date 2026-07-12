#!/opt/local/bin/python3
"""
V3-EXQ-611 -- MECH-341 substrate-readiness 4-arm diagnostic.

Claims: [MECH-341] (diagnostic; experiment_purpose=diagnostic; not weighted
        in confidence/conflict scoring per Phase-3 governance rules).

Purpose
-------
Validate that the MECH-341 substrate (landed 2026-05-27 in
ree-v3/ree_core/predictors/e3_score_diversity.py) actually preserves
post-E3-scoring trajectory-class diversity in the EXQ-608 env. 4-arm
ablation across the master switch and its two sub-flavours:

  ARM_0 ALL_OFF   -- use_e3_score_diversity=False. Reproduces V3-EXQ-608
                     baseline; expected `R2a_e3_collapse_confirmed_large_gap`
                     majority across seeds.
  ARM_1 OPT1_ONLY -- use_e3_score_diversity=True, entropy bonus ON,
                     stratified select OFF. Tests whether the soft entropy
                     bias alone is enough to drag selected_class count up
                     to >= 2.
  ARM_2 OPT2_ONLY -- entropy bonus OFF, stratified select ON. Tests whether
                     the categorical stratification alone forces >= 2-class
                     survival.
  ARM_3 BOTH_ON   -- both sub-flavours ON. Default substrate configuration.

Pre-registered acceptance criteria (substrate-readiness, NOT behavioural):

  C1 (master fires): in ARM_1 / ARM_2 / ARM_3 the substrate's diagnostic
     counters n_calls_total > 0 across all seeds AND
     n_entropy_bonus_fired > 0 (ARM_1, ARM_3) or n_stratified_fired > 0
     (ARM_2, ARM_3).
  C2 (no spurious activity in OFF arm): ARM_0 reproduces EXQ-608's
     `R2a_e3_collapse_confirmed_large_gap` majority (>= 2 of 3 seeds).
  C3 (single-option arm produces diversity): at least one of ARM_1 /
     ARM_2 produces `selected_classes_count >= 2` AND
     `frac_pre_ge2 >= 0.5` (i.e., when CEM delivers diversity, the
     substrate preserves at least 2 classes through E3 scoring) on the
     majority of seeds.
  C4 (option dissociation): selected_action_entropy across class-count
     domain is ordered ARM_0 <= ARM_2 <= ARM_3, with ARM_1 free
     (Option 1 soft-biases without strictly forcing class survival, so
     it may either help or be insufficient depending on lambda).
  R2.c readiness: at least ARM_2 OR ARM_3 produces
     selected_classes_count >= 2 AND first_action_entropy_proxy > 0.3
     (the threshold that downstream P3 behavioural arms would need to
     clear). PASS on R2.c readiness routes the next governance walk to
     queue B_only / ablate_B / ALL_ON behavioural successor.

PASS = (a) >= 9 of 12 (4 arms x 3 seeds) seeds run to P1 completion
without ERROR; (b) C1 holds across the three substrate-ON arms; (c) C2
or C3 holds (i.e., the substrate either fails-as-OFF or succeeds-as-ON
unambiguously).

Phases
------
P0 (30 ep, instrumentation OFF): warmup. Online updates accumulate
residue field / EWMA / V_s anchor sets. Per V3-EXQ-603b failure-autopsy,
the warmup must clear ~14 step/ep mortality on SD-054 reef.
P1 (20 ep, instrumentation ON): measurement window. Per-tick pre/post-E3
class accounting + per-arm MECH-341 diagnostic counter readout.

Budget: 4 arms x 3 seeds x 50 ep x 200 steps = 120k steps total.
Estimated ~100-150 min on Mac (DLAPTOP-4.local @ ~14 steps/sec).

Implementation notes
--------------------
- Reuses EXQ-608's per-tick measurement helpers verbatim
  (_trajectory_first_action_class, _per_class_score_stats). Same metric
  stack so ARM_0 results are directly comparable to EXQ-608 manifest.
- Each arm is an independent REEAgent build (no in-process flag toggling
  mid-run; clean comparison surface).
- experiment_purpose=diagnostic. Manifest evidence_direction set to
  non_contributory per Phase-3 governance rules; substrate-readiness
  diagnostics do not weight claim confidence.

See REE_assembly/docs/architecture/mech_341_e3_score_diversity_preservation.md
and REE_assembly/evidence/planning/behavioral_diversity_isolation_plan.md.
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


EXPERIMENT_TYPE = "v3_exq_611_mech341_substrate_readiness_4arm"
QUEUE_ID = "V3-EXQ-611"
CLAIM_IDS: List[str] = ["MECH-341"]
EXPERIMENT_PURPOSE = "diagnostic"

SEEDS = [42, 43, 44]
P0_WARMUP_EPISODES = 30
P1_MEASUREMENT_EPISODES = 20
STEPS_PER_EPISODE = 200

DRY_RUN_SEEDS = [42]
DRY_RUN_P0 = 2
DRY_RUN_P1 = 2
DRY_RUN_STEPS = 30

# Decision-rule thresholds (mirror EXQ-608 so ARM_0 results are
# directly comparable to the manifest already in evidence/experiments).
PRE_GE2_FRAC_GATE = 0.5
E3_COLLAPSE_FRAC_GATE = 0.8
E3_COLLAPSE_FRAC_FLOOR = 0.5
NEAR_TIE_FRAC_GATE = 0.2
SCORE_GAP_EPSILON_RANGE_FRAC = 0.05
MIN_SEEDS_PER_ARM_FOR_PASS = 2  # of 3
SELECTED_CLASSES_GE_FOR_C3 = 2
R2C_READINESS_ENTROPY_THRESHOLD = 0.3

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


# ---------------------------------------------------------------------------
# Arm definitions
# ---------------------------------------------------------------------------


ARMS: List[Dict[str, Any]] = [
    {
        "arm_id": "ARM_0_ALL_OFF",
        "label": "all_off",
        "cfg_overrides": dict(
            use_e3_score_diversity=False,
        ),
    },
    {
        "arm_id": "ARM_1_OPT1_ONLY",
        "label": "opt1_entropy_bonus_only",
        "cfg_overrides": dict(
            use_e3_score_diversity=True,
            use_e3_diversity_entropy_bonus=True,
            use_e3_diversity_stratified_select=False,
        ),
    },
    {
        "arm_id": "ARM_2_OPT2_ONLY",
        "label": "opt2_stratified_only",
        "cfg_overrides": dict(
            use_e3_score_diversity=True,
            use_e3_diversity_entropy_bonus=False,
            use_e3_diversity_stratified_select=True,
        ),
    },
    {
        "arm_id": "ARM_3_BOTH_ON",
        "label": "both_on",
        "cfg_overrides": dict(
            use_e3_score_diversity=True,
            use_e3_diversity_entropy_bonus=True,
            use_e3_diversity_stratified_select=True,
        ),
    },
]


def _make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(seed=seed, **ENV_KWARGS)


def _make_agent(env: CausalGridWorldV2, cfg_overrides: Dict[str, Any]) -> REEAgent:
    """Main-path SP-CEM + MECH-313 + V_s + SD-054 stack with MECH-341 overrides."""
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
        use_support_preserving_cem=True,
        support_preserving_stratified_elites=True,
        support_preserving_ao_std_floor=0.2,
        support_preserving_min_first_action_classes=2,
        use_noise_floor=True,
        noise_floor_alpha=0.1,
        use_per_stream_vs=True,
        use_per_region_vs=True,
        use_event_segmenter=True,
        use_invalidation_trigger=True,
        use_anchor_sets=True,
    )
    for k, v in cfg_overrides.items():
        setattr(cfg, k, v)
    return REEAgent(cfg)


# ---------------------------------------------------------------------------
# Per-tick measurement helpers (mirror EXQ-608 conventions)
# ---------------------------------------------------------------------------


def _trajectory_first_action_class(traj) -> int:
    return int(
        traj.actions[:, 0, :].argmax(dim=-1).detach().reshape(-1)[0].item()
    )


def _per_class_score_stats(
    candidates: List, last_scores: Optional[torch.Tensor]
) -> Tuple[Dict[int, Dict[str, float]], Optional[int], Optional[float], bool]:
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
    """Shannon entropy in nats over a class-count dict."""
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
    agent = _make_agent(env, arm["cfg_overrides"])
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
        score_range = max(top2_gaps_pre_ge2) - min(top2_gaps_pre_ge2)
        epsilon = max(score_range * SCORE_GAP_EPSILON_RANGE_FRAC, 1e-6)
        n_above_eps = sum(1 for g in top2_gaps_pre_ge2 if g > epsilon)
        n_below_eps = sum(1 for g in top2_gaps_pre_ge2 if g <= epsilon)
        mean_top2_gap = float(sum(top2_gaps_pre_ge2) / len(top2_gaps_pre_ge2))
        frac_e3_collapse = float(n_above_eps / len(top2_gaps_pre_ge2))
        frac_near_tie = float(n_below_eps / len(top2_gaps_pre_ge2))
    else:
        epsilon = None
        mean_top2_gap = None
        frac_e3_collapse = None
        frac_near_tie = None

    if frac_pre_ge2 < PRE_GE2_FRAC_GATE:
        interp = "low_precondition_re_evaluate_theory1"
    elif frac_e3_collapse is None:
        interp = "no_pre_ge2_measurements"
    elif frac_e3_collapse >= E3_COLLAPSE_FRAC_GATE:
        if frac_near_tie is not None and frac_near_tie >= NEAR_TIE_FRAC_GATE:
            interp = "R2a_e3_collapse_confirmed_near_tie_or_mixed"
        else:
            interp = "R2a_e3_collapse_confirmed_large_gap"
    elif frac_e3_collapse < E3_COLLAPSE_FRAC_FLOOR:
        interp = "R2b_e3_preserves_diversity"
    else:
        interp = "inconclusive_resample_heavier"

    n_selected_classes = len(selected_classes_p1)
    selected_class_entropy = _entropy_from_counts(selected_classes_p1)
    committed_class_entropy = _entropy_from_counts(committed_classes_p1)

    diag = (
        agent.score_diversity.get_state()
        if agent.score_diversity is not None
        else None
    )

    return {
        "arm_id": arm["arm_id"],
        "seed": int(seed),
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
        "score_gap_epsilon": (
            None if epsilon is None else round(epsilon, 6)
        ),
        "frac_e3_collapse_above_eps": (
            None if frac_e3_collapse is None else round(frac_e3_collapse, 6)
        ),
        "frac_near_tie_below_eps": (
            None if frac_near_tie is None else round(frac_near_tie, 6)
        ),
        "selected_classes_p1_counts": {
            str(k): int(v) for k, v in sorted(selected_classes_p1.items())
        },
        "committed_classes_p1_counts": {
            str(k): int(v) for k, v in sorted(committed_classes_p1.items())
        },
        "n_unique_selected_classes": int(n_selected_classes),
        "selected_class_entropy_nats": round(selected_class_entropy, 6),
        "committed_class_entropy_nats": round(committed_class_entropy, 6),
        "interpretation_label": interp,
        "mech341_diagnostics": diag,
        "error_note": error_note,
    }


def _interpret_arm(seed_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    labels = [r["interpretation_label"] for r in seed_rows]
    label_counts: Dict[str, int] = {}
    for l in labels:
        label_counts[l] = label_counts.get(l, 0) + 1
    majority = max(label_counts.items(), key=lambda kv: kv[1])

    n_seeds_completed = sum(1 for r in seed_rows if r["error_note"] is None)
    n_seeds_with_ge2_classes = sum(
        1 for r in seed_rows
        if r["n_unique_selected_classes"] >= SELECTED_CLASSES_GE_FOR_C3
        and r["frac_pre_ge2"] >= PRE_GE2_FRAC_GATE
    )
    selected_entropies = [r["selected_class_entropy_nats"] for r in seed_rows]
    mean_selected_entropy = (
        sum(selected_entropies) / len(selected_entropies)
        if selected_entropies else 0.0
    )

    return {
        "n_seeds_completed": int(n_seeds_completed),
        "per_seed_labels": labels,
        "label_counts": label_counts,
        "majority_label": majority[0],
        "majority_n_of_total": [int(majority[1]), int(len(labels))],
        "unanimous": bool(len(label_counts) == 1),
        "n_seeds_with_ge2_selected_classes": int(n_seeds_with_ge2_classes),
        "mean_selected_class_entropy_nats": round(mean_selected_entropy, 6),
    }


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
            "cfg_overrides": arm["cfg_overrides"],
            "per_seed_results": seed_rows,
            "cross_seed_interpretation": cross,
        })

    # ----- Cross-arm acceptance criteria -----
    by_id = {a["arm_id"]: a for a in arms_out}

    arm0 = by_id["ARM_0_ALL_OFF"]["cross_seed_interpretation"]
    arm1 = by_id["ARM_1_OPT1_ONLY"]["cross_seed_interpretation"]
    arm2 = by_id["ARM_2_OPT2_ONLY"]["cross_seed_interpretation"]
    arm3 = by_id["ARM_3_BOTH_ON"]["cross_seed_interpretation"]

    # C1: substrate fires in ON arms.
    def _diag_fires_entropy(arm_out):
        return all(
            (r["mech341_diagnostics"] or {}).get(
                "mech341_n_entropy_bonus_fired", 0
            ) > 0
            for r in arm_out["per_seed_results"]
            if r["error_note"] is None
        )

    def _diag_fires_stratified(arm_out):
        return all(
            (r["mech341_diagnostics"] or {}).get(
                "mech341_n_stratified_fired", 0
            ) > 0
            for r in arm_out["per_seed_results"]
            if r["error_note"] is None
        )

    c1_arm1 = _diag_fires_entropy(by_id["ARM_1_OPT1_ONLY"])
    c1_arm2 = _diag_fires_stratified(by_id["ARM_2_OPT2_ONLY"])
    c1_arm3 = (
        _diag_fires_entropy(by_id["ARM_3_BOTH_ON"])
        and _diag_fires_stratified(by_id["ARM_3_BOTH_ON"])
    )
    c1_holds = bool(c1_arm1 and c1_arm2 and c1_arm3)

    # C2: ARM_0 reproduces EXQ-608 baseline (R2a majority).
    c2_holds = bool(arm0["majority_label"] == "R2a_e3_collapse_confirmed_large_gap")

    # C3: at least one of ARM_1 / ARM_2 produces selected_classes >= 2 with
    # frac_pre_ge2 >= 0.5 on the majority of seeds.
    c3_arm1 = arm1["n_seeds_with_ge2_selected_classes"] >= MIN_SEEDS_PER_ARM_FOR_PASS
    c3_arm2 = arm2["n_seeds_with_ge2_selected_classes"] >= MIN_SEEDS_PER_ARM_FOR_PASS
    c3_holds = bool(c3_arm1 or c3_arm2)

    # C4: ordering check (informational; does not gate PASS).
    c4_arm0_le_arm2 = arm0["mean_selected_class_entropy_nats"] <= arm2["mean_selected_class_entropy_nats"]
    c4_arm2_le_arm3 = arm2["mean_selected_class_entropy_nats"] <= arm3["mean_selected_class_entropy_nats"] + 1e-6

    # R2.c readiness: at least ARM_2 or ARM_3 clears threshold.
    r2c_arm2 = (
        arm2["mean_selected_class_entropy_nats"] >= R2C_READINESS_ENTROPY_THRESHOLD
        and arm2["n_seeds_with_ge2_selected_classes"] >= MIN_SEEDS_PER_ARM_FOR_PASS
    )
    r2c_arm3 = (
        arm3["mean_selected_class_entropy_nats"] >= R2C_READINESS_ENTROPY_THRESHOLD
        and arm3["n_seeds_with_ge2_selected_classes"] >= MIN_SEEDS_PER_ARM_FOR_PASS
    )
    r2c_readiness = bool(r2c_arm2 or r2c_arm3)

    total_seeds = 4 * len(seeds)
    total_completed = sum(
        a["cross_seed_interpretation"]["n_seeds_completed"] for a in arms_out
    )

    passed = bool(
        total_completed >= max(MIN_SEEDS_PER_ARM_FOR_PASS * 4 - 3, 9)
        and c1_holds
        and (c2_holds or c3_holds)
    )

    return {
        "outcome": "PASS" if passed else "FAIL",
        "seeds": seeds,
        "n_arms": len(arms_out),
        "total_seeds_attempted": int(total_seeds),
        "total_seeds_completed": int(total_completed),
        "p0_episodes": int(p0_episodes),
        "p1_episodes": int(p1_episodes),
        "steps_per_episode": int(steps_per_episode),
        "decision_rule_thresholds": {
            "pre_ge2_frac_gate": float(PRE_GE2_FRAC_GATE),
            "e3_collapse_frac_gate": float(E3_COLLAPSE_FRAC_GATE),
            "e3_collapse_frac_floor": float(E3_COLLAPSE_FRAC_FLOOR),
            "near_tie_frac_gate": float(NEAR_TIE_FRAC_GATE),
            "score_gap_epsilon_range_frac": float(SCORE_GAP_EPSILON_RANGE_FRAC),
            "selected_classes_ge_for_c3": int(SELECTED_CLASSES_GE_FOR_C3),
            "r2c_readiness_entropy_threshold": float(R2C_READINESS_ENTROPY_THRESHOLD),
        },
        "acceptance_criteria": {
            "C1_substrate_fires_in_on_arms": c1_holds,
            "C2_arm0_reproduces_exq608_r2a": c2_holds,
            "C3_single_option_arm_produces_diversity": c3_holds,
            "C4_arm0_le_arm2_entropy": bool(c4_arm0_le_arm2),
            "C4_arm2_le_arm3_entropy": bool(c4_arm2_le_arm3),
            "R2c_readiness": r2c_readiness,
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
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": timestamp_utc,
        "outcome": result["outcome"],
        "evidence_direction": "non_contributory",
        "evidence_direction_note": (
            "experiment_purpose=diagnostic; substrate-readiness ablation, "
            "not behavioural evidence. Not weighted in confidence or conflict "
            "scoring per Phase-3 governance rules. PASS routes the next "
            "governance walk to queue B_only / ablate_B / ALL_ON behavioural "
            "successor; FAIL routes to substrate revisit (option-3 jitter or "
            "lambda tuning). See "
            "behavioral_diversity_isolation_plan.md R2.c rule."
        ),
        "dry_run": bool(dry_run),
        "env_kwargs": dict(ENV_KWARGS),
        "config_summary": {
            "use_support_preserving_cem": True,
            "use_noise_floor": True,
            "use_per_stream_vs": True,
            "use_anchor_sets": True,
            "z_goal_enabled": True,
            "drive_weight": 2.0,
            "alpha_world": 0.9,
            "reef_enabled": True,
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
        f"C1={result['acceptance_criteria']['C1_substrate_fires_in_on_arms']} "
        f"C2={result['acceptance_criteria']['C2_arm0_reproduces_exq608_r2a']} "
        f"C3={result['acceptance_criteria']['C3_single_option_arm_produces_diversity']} "
        f"R2c={result['acceptance_criteria']['R2c_readiness']}",
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
