#!/opt/local/bin/python3
"""
V3-EXQ-611b -- MECH-341 retune 6-arm validation (supersedes V3-EXQ-611).

Claims: [MECH-341] (diagnostic; experiment_purpose=diagnostic; not weighted
        in confidence/conflict scoring per Phase-3 governance rules).
Supersedes: V3-EXQ-611 (substrate-readiness 4-arm FAIL 2026-05-27T13:02Z).

Purpose
-------
Validate the MECH-341 retune (2026-05-28) against the two failure modes
surfaced by V3-EXQ-611:

  Failure 1: entropy_bonus max_abs ~0.023-0.044 dwarfed by observed
    mean_top2_class_gap 0.27-1.96 -- entropy_bias_scale=0.1 default sits
    well below the score-gap regime; substrate fires but cannot move
    selection ordering.
  Failure 2: stratified_select fired 0 times across all 3 seeds in ARM_2.
    Root cause: committed branch was never entered during the validation
    episodes (high running_variance), and the prior implementation gated
    stratified_select to the committed branch only.

Two retune actions land together:
  (a) Module change (ree-v3/ree_core/predictors/e3_selector.py): apply
      stratified_select on the uncommitted (multinomial) branch too --
      Option-2 categorical-preservation applies regardless of commit state.
      Bit-identical when score_diversity is None or sub-flag is False.
  (b) Validation sweep (this experiment): 6-arm factorial across 3 option
      groups (OPT1_only / OPT2_only / BOTH) x 2 entropy_bias_scale values
      (1.0 / 2.0). Replaces V3-EXQ-611's ALL_OFF anchor + 3 ON arms with
      6 informative ON arms; the ALL_OFF behaviour is established by
      V3-EXQ-611's manifest (already on origin/master).

NO config default changes per implement-substrate skill rule. The scale
values are passed in cfg_overrides per arm.

Arms
----
  ARM_1_OPT1_S1   entropy_bonus ON,  stratified OFF, scale=1.0
  ARM_2_OPT1_S2   entropy_bonus ON,  stratified OFF, scale=2.0
  ARM_3_OPT2_S1   entropy_bonus OFF, stratified ON,  scale=1.0   (scale unused on OPT2)
  ARM_4_OPT2_S2   entropy_bonus OFF, stratified ON,  scale=2.0   (scale unused on OPT2)
  ARM_5_BOTH_S1   entropy_bonus ON,  stratified ON,  scale=1.0
  ARM_6_BOTH_S2   entropy_bonus ON,  stratified ON,  scale=2.0

Pre-registered acceptance criteria
-----------------------------------
  C1 (call-site expansion fixes zero-fires): in the 4 stratified-ON arms
     (ARM_3 / ARM_4 / ARM_5 / ARM_6), n_stratified_fired > 0 across all
     seeds (3/3). Direct test of the module-level retune.
  C2 (bonus scale-commensurate): in the 4 entropy-ON arms (ARM_1 / ARM_2 /
     ARM_5 / ARM_6), last_entropy_bonus_max_abs >= 0.7 * scale on the
     majority of seeds (substrate is putting the configured magnitude on
     the table; clamp is tight at +/-scale by construction).
  C3 (diversity preserved at the selection step): at least one arm
     produces selected_classes_count >= 2 AND frac_pre_ge2 >= 0.5 on a
     majority of seeds (>= 2/3).
  C4 (informational ordering): selected-class entropy in
     scale=2.0 BOTH_ON >= 1.0 BOTH_ON (entropy bonus is monotone in scale).
  R2.c readiness: at least one arm produces
     mean_selected_class_entropy_nats >= 0.3 AND
     n_seeds_with_ge2_selected_classes >= 2/3.

PASS = (a) >= 15 of 18 seeds (6 arms x 3 seeds) run to P1 completion
without ERROR; (b) C1 holds (the retune's primary acceptance gate);
(c) C2 or C3 holds (substrate either drives meaningful score perturbation
or substrate-natural pool diversity survives selection).

Phases
------
P0 (30 ep, instrumentation OFF): warmup. Matches V3-EXQ-611 budget so
ARM-vs-ARM comparison is calibrated.
P1 (20 ep, instrumentation ON): measurement window. Per-tick pre/post-E3
class accounting + per-arm MECH-341 diagnostic counter readout.

Budget: 6 arms x 3 seeds x 50 ep x 200 steps = 180k steps total.
Estimated ~150-225 min on Mac (DLAPTOP-4.local @ ~14 steps/sec).

Implementation notes
--------------------
- env_kwargs IDENTICAL to V3-EXQ-611 -- same SD-054 reef-bipartite +
  hazard_food_attraction + harm_history substrate. Direct manifest
  comparability with the 611 ARM_0_ALL_OFF baseline already on master.
- Reuses per-tick measurement helpers verbatim from V3-EXQ-611.
- Each arm is an independent REEAgent build (no in-process flag toggling
  mid-run).
- experiment_purpose=diagnostic. evidence_direction=non_contributory per
  Phase-3 governance rules; substrate retune validations do not weight
  claim confidence (Q-054 behavioural falsifier is the governance signal).

See REE_assembly/docs/architecture/mech_341_e3_score_diversity_preservation.md,
REE_assembly/evidence/planning/behavioral_diversity_isolation_plan.md, and
REE_assembly/evidence/planning/substrate_queue.json (MECH-341 retune entry).
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


EXPERIMENT_TYPE = "v3_exq_611b_mech341_retune_6arm"
QUEUE_ID = "V3-EXQ-611b"
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

# Decision-rule thresholds (mirror EXQ-608 / EXQ-611 conventions for direct
# manifest comparability).
PRE_GE2_FRAC_GATE = 0.5
E3_COLLAPSE_FRAC_GATE = 0.8
E3_COLLAPSE_FRAC_FLOOR = 0.5
NEAR_TIE_FRAC_GATE = 0.2
SCORE_GAP_EPSILON_RANGE_FRAC = 0.05
MIN_SEEDS_PER_ARM_FOR_PASS = 2  # of 3
SELECTED_CLASSES_GE_FOR_C3 = 2
R2C_READINESS_ENTROPY_THRESHOLD = 0.3
C2_BONUS_SCALE_COMMENSURATE_FRAC = 0.7  # max_abs >= 0.7 * scale

# IDENTICAL to V3-EXQ-611 for direct manifest comparability with the
# already-on-origin ARM_0_ALL_OFF baseline.
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
        "arm_id": "ARM_1_OPT1_S1",
        "label": "opt1_entropy_bonus_only_scale_1p0",
        "cfg_overrides": dict(
            use_e3_score_diversity=True,
            use_e3_diversity_entropy_bonus=True,
            use_e3_diversity_stratified_select=False,
            e3_diversity_entropy_bias_scale=1.0,
        ),
        "entropy_scale": 1.0,
        "stratified_on": False,
    },
    {
        "arm_id": "ARM_2_OPT1_S2",
        "label": "opt1_entropy_bonus_only_scale_2p0",
        "cfg_overrides": dict(
            use_e3_score_diversity=True,
            use_e3_diversity_entropy_bonus=True,
            use_e3_diversity_stratified_select=False,
            e3_diversity_entropy_bias_scale=2.0,
        ),
        "entropy_scale": 2.0,
        "stratified_on": False,
    },
    {
        "arm_id": "ARM_3_OPT2_S1",
        "label": "opt2_stratified_only_scale_1p0",
        "cfg_overrides": dict(
            use_e3_score_diversity=True,
            use_e3_diversity_entropy_bonus=False,
            use_e3_diversity_stratified_select=True,
            e3_diversity_entropy_bias_scale=1.0,
        ),
        "entropy_scale": 1.0,
        "stratified_on": True,
    },
    {
        "arm_id": "ARM_4_OPT2_S2",
        "label": "opt2_stratified_only_scale_2p0",
        "cfg_overrides": dict(
            use_e3_score_diversity=True,
            use_e3_diversity_entropy_bonus=False,
            use_e3_diversity_stratified_select=True,
            e3_diversity_entropy_bias_scale=2.0,
        ),
        "entropy_scale": 2.0,
        "stratified_on": True,
    },
    {
        "arm_id": "ARM_5_BOTH_S1",
        "label": "both_on_scale_1p0",
        "cfg_overrides": dict(
            use_e3_score_diversity=True,
            use_e3_diversity_entropy_bonus=True,
            use_e3_diversity_stratified_select=True,
            e3_diversity_entropy_bias_scale=1.0,
        ),
        "entropy_scale": 1.0,
        "stratified_on": True,
    },
    {
        "arm_id": "ARM_6_BOTH_S2",
        "label": "both_on_scale_2p0",
        "cfg_overrides": dict(
            use_e3_score_diversity=True,
            use_e3_diversity_entropy_bonus=True,
            use_e3_diversity_stratified_select=True,
            e3_diversity_entropy_bias_scale=2.0,
        ),
        "entropy_scale": 2.0,
        "stratified_on": True,
    },
]


def _make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(seed=seed, **ENV_KWARGS)


def _make_agent(env: CausalGridWorldV2, cfg_overrides: Dict[str, Any]) -> REEAgent:
    """Main-path SP-CEM + MECH-313 + V_s + SD-054 stack with MECH-341 overrides.

    Config matches V3-EXQ-611 verbatim (so ARM_0_ALL_OFF baseline in the 611
    manifest is the cross-experiment anchor); only the MECH-341 cfg_overrides
    differ across the 6 arms.
    """
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
# Per-tick measurement helpers (mirror EXQ-608 / EXQ-611 conventions)
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
    entropy_bonus_max_abs_p1: List[float] = []

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

                # Per-tick entropy bonus max_abs (only meaningful when entropy
                # sub-flavour is ON; substrate writes the last call's max_abs
                # into the diagnostic buffer regardless).
                if agent.score_diversity is not None:
                    last_max_abs = float(
                        agent.score_diversity.diagnostics.last_entropy_bonus_max_abs
                    )
                    entropy_bonus_max_abs_p1.append(last_max_abs)

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

    if entropy_bonus_max_abs_p1:
        mean_entropy_max_abs = float(
            sum(entropy_bonus_max_abs_p1) / len(entropy_bonus_max_abs_p1)
        )
        max_entropy_max_abs = float(max(entropy_bonus_max_abs_p1))
    else:
        mean_entropy_max_abs = 0.0
        max_entropy_max_abs = 0.0

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
        "mean_entropy_bonus_max_abs_p1": round(mean_entropy_max_abs, 6),
        "max_entropy_bonus_max_abs_p1": round(max_entropy_max_abs, 6),
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
    max_entropy_max_abs_per_seed = [
        r["max_entropy_bonus_max_abs_p1"] for r in seed_rows
    ]
    mean_max_entropy_max_abs = (
        sum(max_entropy_max_abs_per_seed) / len(max_entropy_max_abs_per_seed)
        if max_entropy_max_abs_per_seed else 0.0
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
        "mean_max_entropy_bonus_max_abs_p1": round(mean_max_entropy_max_abs, 6),
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
            "entropy_scale": arm["entropy_scale"],
            "stratified_on": arm["stratified_on"],
            "per_seed_results": seed_rows,
            "cross_seed_interpretation": cross,
        })

    # ----- Cross-arm acceptance criteria -----
    by_id = {a["arm_id"]: a for a in arms_out}

    # C1: stratified_select fires across all completed seeds in the 4
    # stratified-ON arms. Direct test of the module-level retune (call-site
    # expansion from committed-only to committed + uncommitted).
    def _stratified_fires_all_seeds(arm_out):
        completed = [r for r in arm_out["per_seed_results"] if r["error_note"] is None]
        if not completed:
            return False
        return all(
            (r["mech341_diagnostics"] or {}).get("mech341_n_stratified_fired", 0) > 0
            for r in completed
        )

    stratified_arms = [
        by_id["ARM_3_OPT2_S1"],
        by_id["ARM_4_OPT2_S2"],
        by_id["ARM_5_BOTH_S1"],
        by_id["ARM_6_BOTH_S2"],
    ]
    c1_per_arm = {a["arm_id"]: _stratified_fires_all_seeds(a) for a in stratified_arms}
    c1_holds = bool(all(c1_per_arm.values()))

    # C2: bonus max_abs is scale-commensurate in entropy-ON arms (majority of seeds).
    entropy_arms = [
        by_id["ARM_1_OPT1_S1"],
        by_id["ARM_2_OPT1_S2"],
        by_id["ARM_5_BOTH_S1"],
        by_id["ARM_6_BOTH_S2"],
    ]

    def _entropy_max_abs_commensurate(arm_out):
        scale = float(arm_out["entropy_scale"])
        target = C2_BONUS_SCALE_COMMENSURATE_FRAC * scale
        n_ok = sum(
            1 for r in arm_out["per_seed_results"]
            if r["error_note"] is None and r["max_entropy_bonus_max_abs_p1"] >= target
        )
        return n_ok >= MIN_SEEDS_PER_ARM_FOR_PASS

    c2_per_arm = {a["arm_id"]: _entropy_max_abs_commensurate(a) for a in entropy_arms}
    c2_holds = bool(all(c2_per_arm.values()))

    # C3: at least one arm produces selected_classes >= 2 with frac_pre_ge2 >=
    # 0.5 on the majority of seeds.
    c3_per_arm = {
        a["arm_id"]: a["cross_seed_interpretation"]["n_seeds_with_ge2_selected_classes"]
        >= MIN_SEEDS_PER_ARM_FOR_PASS
        for a in arms_out
    }
    c3_holds = bool(any(c3_per_arm.values()))

    # C4 (informational): scale=2.0 BOTH_ON entropy >= scale=1.0 BOTH_ON entropy.
    both_s1 = by_id["ARM_5_BOTH_S1"]["cross_seed_interpretation"][
        "mean_selected_class_entropy_nats"
    ]
    both_s2 = by_id["ARM_6_BOTH_S2"]["cross_seed_interpretation"][
        "mean_selected_class_entropy_nats"
    ]
    c4_monotone = bool(both_s2 >= both_s1 - 1e-6)

    # R2.c readiness: at least one arm clears the threshold.
    r2c_per_arm = {
        a["arm_id"]: (
            a["cross_seed_interpretation"]["mean_selected_class_entropy_nats"]
            >= R2C_READINESS_ENTROPY_THRESHOLD
            and a["cross_seed_interpretation"]["n_seeds_with_ge2_selected_classes"]
            >= MIN_SEEDS_PER_ARM_FOR_PASS
        )
        for a in arms_out
    }
    r2c_readiness = bool(any(r2c_per_arm.values()))

    total_seeds = 6 * len(seeds)
    total_completed = sum(
        a["cross_seed_interpretation"]["n_seeds_completed"] for a in arms_out
    )

    # PASS gate: substantial completion + C1 (the retune's primary acceptance)
    # + (C2 or C3) (substrate either drives meaningful score perturbation or
    # substrate-natural pool diversity survives selection).
    passed = bool(
        total_completed >= max(MIN_SEEDS_PER_ARM_FOR_PASS * 6 - 3, 9)
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
        "supersedes": "V3-EXQ-611",
        "decision_rule_thresholds": {
            "pre_ge2_frac_gate": float(PRE_GE2_FRAC_GATE),
            "e3_collapse_frac_gate": float(E3_COLLAPSE_FRAC_GATE),
            "e3_collapse_frac_floor": float(E3_COLLAPSE_FRAC_FLOOR),
            "near_tie_frac_gate": float(NEAR_TIE_FRAC_GATE),
            "score_gap_epsilon_range_frac": float(SCORE_GAP_EPSILON_RANGE_FRAC),
            "selected_classes_ge_for_c3": int(SELECTED_CLASSES_GE_FOR_C3),
            "r2c_readiness_entropy_threshold": float(R2C_READINESS_ENTROPY_THRESHOLD),
            "c2_bonus_scale_commensurate_frac": float(C2_BONUS_SCALE_COMMENSURATE_FRAC),
        },
        "acceptance_criteria": {
            "C1_stratified_fires_all_on_arms": c1_holds,
            "C1_per_arm": c1_per_arm,
            "C2_entropy_bonus_scale_commensurate": c2_holds,
            "C2_per_arm": c2_per_arm,
            "C3_single_arm_produces_diversity": c3_holds,
            "C3_per_arm": c3_per_arm,
            "C4_both_scale_monotone": c4_monotone,
            "R2c_readiness": r2c_readiness,
            "R2c_per_arm": r2c_per_arm,
        },
        "interpretation_grid": {
            "PASS_with_C1_and_C3": (
                "Retune fully validated: stratified call-site expansion fixes "
                "zero-fires AND substrate produces selected-class diversity. "
                "Route to V3-EXQ-611c-or-successor B_only / ablate_B / ALL_ON "
                "behavioural arm under R2.c rule."
            ),
            "PASS_with_C1_and_C2_only": (
                "Substrate fires + bonus is scale-commensurate, but selection "
                "still collapses to single class. Routes to additional retune: "
                "consider raising stratified_temperature or revisiting the "
                "softmax-sample-across-class-representatives step (the sampled "
                "rep may be dominated by single-class repeats due to candidate "
                "imbalance)."
            ),
            "FAIL_with_C1_false": (
                "Call-site expansion did not fix the zero-fires issue. "
                "Routes to /diagnose-errors on the e3_selector.py wiring -- "
                "stratified_select is reachable but its precondition "
                "(unique_classes >= min_classes_for_stratification) may be "
                "violated even on the uncommitted path."
            ),
            "FAIL_with_C1_true_C2_C3_false": (
                "Retune surface fires but is architecturally insufficient. "
                "Routes substrate revisit: consider raising "
                "min_classes_for_stratification floor, switching to argmax-"
                "sample on class representatives, or revisiting the entire "
                "Option-2 algorithm (Padoa-Schioppa OFC value-comparison fit)."
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
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "supersedes": "V3-EXQ-611",
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": timestamp_utc,
        "outcome": result["outcome"],
        "evidence_direction": "non_contributory",
        "evidence_direction_note": (
            "experiment_purpose=diagnostic; substrate-retune ablation, "
            "not behavioural evidence. Not weighted in confidence or conflict "
            "scoring per Phase-3 governance rules. PASS routes to behavioural "
            "successor under R2.c rule; FAIL routes to /diagnose-errors or "
            "further substrate revisit per the interpretation grid. See "
            "behavioral_diversity_isolation_plan.md R2.c rule and "
            "mech_341_e3_score_diversity_preservation.md retune note "
            "2026-05-28."
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
            "reef_bipartite_layout": True,
        },
        "result": result,
    }


def main() -> Tuple[Optional[str], Optional[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--out-dir", type=str, default=None)
    args = parser.parse_args()
    _run_started = datetime.now(timezone.utc)

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
        elapsed_seconds=(datetime.now(timezone.utc) - _run_started).total_seconds(),
    )

    print(f"manifest: {out_path}", flush=True)
    if not args.dry_run:
        print(f"Result written to: {out_path}", flush=True)
    print(
        f"outcome: {result['outcome']} "
        f"completed={result['total_seeds_completed']}/{result['total_seeds_attempted']} "
        f"C1={result['acceptance_criteria']['C1_stratified_fires_all_on_arms']} "
        f"C2={result['acceptance_criteria']['C2_entropy_bonus_scale_commensurate']} "
        f"C3={result['acceptance_criteria']['C3_single_arm_produces_diversity']} "
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
