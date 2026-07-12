#!/opt/local/bin/python3
"""V3-EXQ-573 -- ARC-065 bias-scale 5-10x sweep.

Calibration follow-up to V3-EXQ-571 (E3 score variance decomposition).

EXQ-571 found that F (forward model trajectory variance) dominates 88-89%
of E3 temporal variance while ALL diversity bias components (MECH-313, 314,
320 and others) contribute bias_fraction = 0.0000 -- effectively zero. The
diagnostic explicitly recommended "bias_scale on all diversity components
may need 5-10x increase to compete with F term".

EXQ-569 found that SP-CEM baseline entropy = ~0.496, and all 6 arms
(MECH-313/314/320/260 individually and combined) produced identical
selected_action_entropy (~0.496). This confirms the bias signals are too
small to move E3 scores.

This experiment sweeps bias_scale 1x (current) -> 5x -> 10x for each of
MECH-313, MECH-314, MECH-320 individually and combined, all on SP-CEM base.

MECH-313 (noise_floor) works via temperature addition (noise_floor_alpha),
not a score_bias. At default alpha=0.1, effective_T = 1.0 + 0.1 = 1.1.
At 5x: alpha=0.5 (effective_T=1.5). At 10x: alpha=1.0 (effective_T=2.0).

MECH-314 (structured_curiosity) and MECH-320 (tonic_vigor) work via
score_bias with clamp at curiosity_bias_scale and tonic_vigor_bias_scale.
Default clamp = 0.1 each. At 5x: clamp=0.5. At 10x: clamp=1.0.
Also scale the individual weights proportionally.

Arms:
  ARM_0: SP-CEM baseline (no diversity stack) -- control
  ARM_1: MECH-313 noise_floor at 1x (alpha=0.1) -- current
  ARM_2: MECH-313 noise_floor at 5x (alpha=0.5)
  ARM_3: MECH-313 noise_floor at 10x (alpha=1.0)
  ARM_4: MECH-314 curiosity at 5x (curiosity_bias_scale=0.5, weights x5)
  ARM_5: MECH-314 curiosity at 10x (curiosity_bias_scale=1.0, weights x10)
  ARM_6: MECH-320 tonic_vigor at 5x (tonic_vigor_bias_scale=0.5, w x5)
  ARM_7: MECH-320 tonic_vigor at 10x (tonic_vigor_bias_scale=1.0, w x10)
  ARM_8: Combined all three at 5x
  ARM_9: Combined all three at 10x

Acceptance criteria (pre-registered):
  P1: ANY diversity arm (ARM_1..ARM_9) has selected_action_entropy
      > ARM_0 + 0.05
      (ARM_0 baseline expected ~0.496 from EXQ-569)
  P2 (secondary): Best diversity arm entropy > 0.55
      (meaningful lift above SP-CEM noise floor level)
  P3 (secondary): Combined 10x arm (ARM_9) entropy >= combined 5x arm (ARM_8)
      (monotone relationship -- confirms scale is the lever)

PASS if P1.
FAIL if no arm shows entropy lift.

EXPERIMENT_PURPOSE = "evidence"
Claim IDs: ARC-065, MECH-313, MECH-314, MECH-320
evidence_direction_per_claim: determined per-claim based on arm outcomes.

Smoke:
  /opt/local/bin/python3 experiments/v3_exq_573_arc065_bias_scale_sweep.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiment_protocol import emit_outcome  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorld  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402
from experiments.pack_writer import write_flat_manifest  # noqa: E402


EXPERIMENT_TYPE = "v3_exq_573_arc065_bias_scale_sweep"
QUEUE_ID = "V3-EXQ-573"
CLAIM_IDS: List[str] = ["ARC-065", "MECH-313", "MECH-314", "MECH-320"]
EXPERIMENT_PURPOSE = "evidence"

ENV_SIZE = 8
SEEDS = [42, 43, 44]
EVAL_EPISODES = 30
STEPS_PER_EPISODE = 150

DRY_RUN_SEEDS = [42]
DRY_RUN_EPISODES = 2
DRY_RUN_STEPS = 50

STD_FLOOR = 0.2   # support_preserving_ao_std_floor for SP-CEM

# Acceptance thresholds (pre-registered)
P1_ENTROPY_MARGIN = 0.05      # any arm > ARM_0 + this
P2_ENTROPY_THRESHOLD = 0.55   # best arm entropy above this (secondary)
# P3: ARM_9 entropy >= ARM_8 entropy (monotone, secondary)

# Default (1x) curiosity weights
CURIOSITY_NOVELTY_WEIGHT_1X = 0.05
CURIOSITY_UNCERTAINTY_WEIGHT_1X = 0.05
CURIOSITY_LP_WEIGHT_1X = 0.05
CURIOSITY_BIAS_SCALE_1X = 0.1

# Default (1x) tonic_vigor weights
TONIC_VIGOR_W_ACTION_1X = 0.1
TONIC_VIGOR_W_PASSIVE_1X = 0.1
TONIC_VIGOR_BIAS_SCALE_1X = 0.1

# Default (1x) noise_floor alpha
NOISE_FLOOR_ALPHA_1X = 0.1

# ARM definitions
ARMS: List[Dict[str, Any]] = [
    # ---- ARM_0: SP-CEM baseline, no diversity stack ----
    {
        "arm": "ARM_0_sp_cem_baseline",
        "use_noise_floor": False,
        "noise_floor_alpha": 0.0,
        "use_structured_curiosity": False,
        "curiosity_bias_scale": 0.1,
        "curiosity_novelty_weight": CURIOSITY_NOVELTY_WEIGHT_1X,
        "curiosity_uncertainty_weight": CURIOSITY_UNCERTAINTY_WEIGHT_1X,
        "curiosity_lp_weight": CURIOSITY_LP_WEIGHT_1X,
        "use_tonic_vigor": False,
        "tonic_vigor_bias_scale": 0.1,
        "tonic_vigor_w_action": TONIC_VIGOR_W_ACTION_1X,
        "tonic_vigor_w_passive": TONIC_VIGOR_W_PASSIVE_1X,
        "scale_label": "1x",
        "scale_factor": 1.0,
    },
    # ---- ARM_1: MECH-313 noise_floor at 1x (current default) ----
    {
        "arm": "ARM_1_noise_floor_1x",
        "use_noise_floor": True,
        "noise_floor_alpha": NOISE_FLOOR_ALPHA_1X,
        "use_structured_curiosity": False,
        "curiosity_bias_scale": 0.1,
        "curiosity_novelty_weight": CURIOSITY_NOVELTY_WEIGHT_1X,
        "curiosity_uncertainty_weight": CURIOSITY_UNCERTAINTY_WEIGHT_1X,
        "curiosity_lp_weight": CURIOSITY_LP_WEIGHT_1X,
        "use_tonic_vigor": False,
        "tonic_vigor_bias_scale": 0.1,
        "tonic_vigor_w_action": TONIC_VIGOR_W_ACTION_1X,
        "tonic_vigor_w_passive": TONIC_VIGOR_W_PASSIVE_1X,
        "scale_label": "1x",
        "scale_factor": 1.0,
    },
    # ---- ARM_2: MECH-313 noise_floor at 5x (alpha=0.5) ----
    {
        "arm": "ARM_2_noise_floor_5x",
        "use_noise_floor": True,
        "noise_floor_alpha": NOISE_FLOOR_ALPHA_1X * 5.0,
        "use_structured_curiosity": False,
        "curiosity_bias_scale": 0.1,
        "curiosity_novelty_weight": CURIOSITY_NOVELTY_WEIGHT_1X,
        "curiosity_uncertainty_weight": CURIOSITY_UNCERTAINTY_WEIGHT_1X,
        "curiosity_lp_weight": CURIOSITY_LP_WEIGHT_1X,
        "use_tonic_vigor": False,
        "tonic_vigor_bias_scale": 0.1,
        "tonic_vigor_w_action": TONIC_VIGOR_W_ACTION_1X,
        "tonic_vigor_w_passive": TONIC_VIGOR_W_PASSIVE_1X,
        "scale_label": "5x",
        "scale_factor": 5.0,
    },
    # ---- ARM_3: MECH-313 noise_floor at 10x (alpha=1.0) ----
    {
        "arm": "ARM_3_noise_floor_10x",
        "use_noise_floor": True,
        "noise_floor_alpha": NOISE_FLOOR_ALPHA_1X * 10.0,
        "use_structured_curiosity": False,
        "curiosity_bias_scale": 0.1,
        "curiosity_novelty_weight": CURIOSITY_NOVELTY_WEIGHT_1X,
        "curiosity_uncertainty_weight": CURIOSITY_UNCERTAINTY_WEIGHT_1X,
        "curiosity_lp_weight": CURIOSITY_LP_WEIGHT_1X,
        "use_tonic_vigor": False,
        "tonic_vigor_bias_scale": 0.1,
        "tonic_vigor_w_action": TONIC_VIGOR_W_ACTION_1X,
        "tonic_vigor_w_passive": TONIC_VIGOR_W_PASSIVE_1X,
        "scale_label": "10x",
        "scale_factor": 10.0,
    },
    # ---- ARM_4: MECH-314 curiosity at 5x ----
    {
        "arm": "ARM_4_curiosity_5x",
        "use_noise_floor": False,
        "noise_floor_alpha": 0.0,
        "use_structured_curiosity": True,
        "curiosity_bias_scale": CURIOSITY_BIAS_SCALE_1X * 5.0,
        "curiosity_novelty_weight": CURIOSITY_NOVELTY_WEIGHT_1X * 5.0,
        "curiosity_uncertainty_weight": CURIOSITY_UNCERTAINTY_WEIGHT_1X * 5.0,
        "curiosity_lp_weight": CURIOSITY_LP_WEIGHT_1X * 5.0,
        "use_tonic_vigor": False,
        "tonic_vigor_bias_scale": 0.1,
        "tonic_vigor_w_action": TONIC_VIGOR_W_ACTION_1X,
        "tonic_vigor_w_passive": TONIC_VIGOR_W_PASSIVE_1X,
        "scale_label": "5x",
        "scale_factor": 5.0,
    },
    # ---- ARM_5: MECH-314 curiosity at 10x ----
    {
        "arm": "ARM_5_curiosity_10x",
        "use_noise_floor": False,
        "noise_floor_alpha": 0.0,
        "use_structured_curiosity": True,
        "curiosity_bias_scale": CURIOSITY_BIAS_SCALE_1X * 10.0,
        "curiosity_novelty_weight": CURIOSITY_NOVELTY_WEIGHT_1X * 10.0,
        "curiosity_uncertainty_weight": CURIOSITY_UNCERTAINTY_WEIGHT_1X * 10.0,
        "curiosity_lp_weight": CURIOSITY_LP_WEIGHT_1X * 10.0,
        "use_tonic_vigor": False,
        "tonic_vigor_bias_scale": 0.1,
        "tonic_vigor_w_action": TONIC_VIGOR_W_ACTION_1X,
        "tonic_vigor_w_passive": TONIC_VIGOR_W_PASSIVE_1X,
        "scale_label": "10x",
        "scale_factor": 10.0,
    },
    # ---- ARM_6: MECH-320 tonic_vigor at 5x ----
    {
        "arm": "ARM_6_tonic_vigor_5x",
        "use_noise_floor": False,
        "noise_floor_alpha": 0.0,
        "use_structured_curiosity": False,
        "curiosity_bias_scale": 0.1,
        "curiosity_novelty_weight": CURIOSITY_NOVELTY_WEIGHT_1X,
        "curiosity_uncertainty_weight": CURIOSITY_UNCERTAINTY_WEIGHT_1X,
        "curiosity_lp_weight": CURIOSITY_LP_WEIGHT_1X,
        "use_tonic_vigor": True,
        "tonic_vigor_bias_scale": TONIC_VIGOR_BIAS_SCALE_1X * 5.0,
        "tonic_vigor_w_action": TONIC_VIGOR_W_ACTION_1X * 5.0,
        "tonic_vigor_w_passive": TONIC_VIGOR_W_PASSIVE_1X * 5.0,
        "scale_label": "5x",
        "scale_factor": 5.0,
    },
    # ---- ARM_7: MECH-320 tonic_vigor at 10x ----
    {
        "arm": "ARM_7_tonic_vigor_10x",
        "use_noise_floor": False,
        "noise_floor_alpha": 0.0,
        "use_structured_curiosity": False,
        "curiosity_bias_scale": 0.1,
        "curiosity_novelty_weight": CURIOSITY_NOVELTY_WEIGHT_1X,
        "curiosity_uncertainty_weight": CURIOSITY_UNCERTAINTY_WEIGHT_1X,
        "curiosity_lp_weight": CURIOSITY_LP_WEIGHT_1X,
        "use_tonic_vigor": True,
        "tonic_vigor_bias_scale": TONIC_VIGOR_BIAS_SCALE_1X * 10.0,
        "tonic_vigor_w_action": TONIC_VIGOR_W_ACTION_1X * 10.0,
        "tonic_vigor_w_passive": TONIC_VIGOR_W_PASSIVE_1X * 10.0,
        "scale_label": "10x",
        "scale_factor": 10.0,
    },
    # ---- ARM_8: Combined MECH-313+314+320 at 5x ----
    {
        "arm": "ARM_8_combined_5x",
        "use_noise_floor": True,
        "noise_floor_alpha": NOISE_FLOOR_ALPHA_1X * 5.0,
        "use_structured_curiosity": True,
        "curiosity_bias_scale": CURIOSITY_BIAS_SCALE_1X * 5.0,
        "curiosity_novelty_weight": CURIOSITY_NOVELTY_WEIGHT_1X * 5.0,
        "curiosity_uncertainty_weight": CURIOSITY_UNCERTAINTY_WEIGHT_1X * 5.0,
        "curiosity_lp_weight": CURIOSITY_LP_WEIGHT_1X * 5.0,
        "use_tonic_vigor": True,
        "tonic_vigor_bias_scale": TONIC_VIGOR_BIAS_SCALE_1X * 5.0,
        "tonic_vigor_w_action": TONIC_VIGOR_W_ACTION_1X * 5.0,
        "tonic_vigor_w_passive": TONIC_VIGOR_W_PASSIVE_1X * 5.0,
        "scale_label": "5x",
        "scale_factor": 5.0,
    },
    # ---- ARM_9: Combined MECH-313+314+320 at 10x ----
    {
        "arm": "ARM_9_combined_10x",
        "use_noise_floor": True,
        "noise_floor_alpha": NOISE_FLOOR_ALPHA_1X * 10.0,
        "use_structured_curiosity": True,
        "curiosity_bias_scale": CURIOSITY_BIAS_SCALE_1X * 10.0,
        "curiosity_novelty_weight": CURIOSITY_NOVELTY_WEIGHT_1X * 10.0,
        "curiosity_uncertainty_weight": CURIOSITY_UNCERTAINTY_WEIGHT_1X * 10.0,
        "curiosity_lp_weight": CURIOSITY_LP_WEIGHT_1X * 10.0,
        "use_tonic_vigor": True,
        "tonic_vigor_bias_scale": TONIC_VIGOR_BIAS_SCALE_1X * 10.0,
        "tonic_vigor_w_action": TONIC_VIGOR_W_ACTION_1X * 10.0,
        "tonic_vigor_w_passive": TONIC_VIGOR_W_PASSIVE_1X * 10.0,
        "scale_label": "10x",
        "scale_factor": 10.0,
    },
]


def _entropy(counts: Counter) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    value = 0.0
    for count in counts.values():
        p = count / total
        if p > 0.0:
            value -= p * math.log(p)
    return float(value)


def _mean(
    values: Iterable[float], default: Optional[float] = 0.0
) -> Optional[float]:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not vals:
        return default
    return sum(vals) / len(vals)


def _round6(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), 6)


def _counter_from_dict(data: Dict[Any, Any]) -> Counter:
    counter: Counter = Counter()
    for key, value in data.items():
        counter[int(key)] += int(value)
    return counter


def _make_env(seed: int) -> CausalGridWorld:
    return CausalGridWorld(
        size=ENV_SIZE,
        num_hazards=2,
        num_resources=10,
        hazard_harm=0.01,
        resource_benefit=0.25,
        energy_decay=0.015,
        use_proxy_fields=True,
        proximity_benefit_scale=0.18,
        proximity_harm_scale=0.01,
        resource_respawn_on_consume=True,
        seed=seed,
    )


def _make_config(env: CausalGridWorld, arm: Dict[str, Any]) -> REEConfig:
    return REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        alpha_world=0.9,
        world_dim=32,
        # SP-CEM base (same as EXQ-567 ARM_1, EXQ-569 all arms)
        use_action_class_scaffold_candidates=False,
        use_support_preserving_cem=True,
        support_preserving_stratified_elites=True,
        support_preserving_ao_std_floor=STD_FLOOR,
        support_preserving_min_first_action_classes=2,
        forced_score_bias_per_class=None,
        # MECH-313 noise_floor
        use_noise_floor=bool(arm["use_noise_floor"]),
        noise_floor_alpha=float(arm["noise_floor_alpha"]),
        noise_floor_min_temperature=1.0,
        # MECH-314 structured_curiosity
        use_structured_curiosity=bool(arm["use_structured_curiosity"]),
        use_curiosity_novelty=True,
        use_curiosity_uncertainty=True,
        use_curiosity_learning_progress=True,
        curiosity_novelty_weight=float(arm["curiosity_novelty_weight"]),
        curiosity_uncertainty_weight=float(arm["curiosity_uncertainty_weight"]),
        curiosity_learning_progress_weight=float(arm["curiosity_lp_weight"]),
        curiosity_bias_scale=float(arm["curiosity_bias_scale"]),
        # MECH-320 tonic_vigor
        use_tonic_vigor=bool(arm["use_tonic_vigor"]),
        tonic_vigor_half_life=100.0,
        tonic_vigor_w_action=float(arm["tonic_vigor_w_action"]),
        tonic_vigor_w_passive=float(arm["tonic_vigor_w_passive"]),
        tonic_vigor_bias_scale=float(arm["tonic_vigor_bias_scale"]),
    )


def _run_arm_seed(
    arm: Dict[str, Any],
    seed: int,
    episodes: int,
    steps_per_episode: int,
) -> Dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = _make_env(seed)
    cfg = _make_config(env, arm)
    agent = REEAgent(cfg)

    action_counts: Counter = Counter()
    candidate_first_action_counts: Counter = Counter()
    unique_candidate_classes: List[float] = []
    candidate_entropies: List[float] = []
    visited_positions: Set[Tuple[int, int]] = set()
    total_steps = 0

    for ep in range(episodes):
        _, obs_dict = env.reset()
        agent.reset()

        for _ in range(steps_per_episode):
            obs_body = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]

            visited_positions.add((int(env.agent_x), int(env.agent_y)))

            with torch.no_grad():
                action = agent.act_with_split_obs(
                    torch.tensor(obs_body).float() if not isinstance(obs_body, torch.Tensor) else obs_body,
                    torch.tensor(obs_world).float() if not isinstance(obs_world, torch.Tensor) else obs_world,
                )
                if action is None:
                    idx = random.randint(0, env.action_dim - 1)
                    action = torch.zeros(1, env.action_dim, device=agent.device)
                    action[0, idx] = 1.0

            idx = int(action.argmax(dim=-1).item())
            action_counts[idx] += 1

            hdiag = agent.hippocampal.get_last_propose_diagnostics()
            if hdiag:
                candidate_first_action_counts.update(
                    _counter_from_dict(
                        hdiag.get("candidate_first_action_counts", {})
                    )
                )
                unique_candidate_classes.append(
                    float(hdiag.get("candidate_unique_first_action_classes", 0))
                )
                candidate_entropies.append(
                    float(hdiag.get("candidate_first_action_entropy", 0.0))
                )

            flat_obs, harm_signal, done, info, next_obs_dict = env.step(action)

            with torch.no_grad():
                agent.update_residue(
                    harm_signal=float(harm_signal),
                    world_delta=None,
                    hypothesis_tag=False,
                    owned=True,
                )

            total_steps += 1
            obs_dict = next_obs_dict
            if done:
                break

        if (ep + 1) % 10 == 0 or (ep + 1) == episodes:
            print(
                f"  [train] arm={arm['arm']} seed={seed}"
                f" ep {ep + 1}/{episodes}",
                flush=True,
            )

    action_total = sum(action_counts.values())
    selected_entropy = _entropy(action_counts)
    candidate_unique_mean = _round6(_mean(unique_candidate_classes, None))
    candidate_entropy_mean = _round6(_mean(candidate_entropies, None))
    state_coverage = len(visited_positions) / float(ENV_SIZE * ENV_SIZE)

    return {
        "arm": arm["arm"],
        "seed": seed,
        "total_steps": int(total_steps),
        "selected_action_class_entropy": round(selected_entropy, 6),
        "action_0_fraction": round(
            action_counts.get(0, 0) / action_total if action_total else 0.0,
            6,
        ),
        "unique_actions_taken": int(len(action_counts)),
        "action_counts": dict(sorted(action_counts.items())),
        "state_coverage_fraction": round(state_coverage, 6),
        "unique_positions_visited": int(len(visited_positions)),
        "candidate_unique_first_action_classes_mean": candidate_unique_mean,
        "candidate_first_action_entropy_mean": candidate_entropy_mean,
        "candidate_first_action_counts": dict(
            sorted(candidate_first_action_counts.items())
        ),
    }


def _arm_rows(
    rows: List[Dict[str, Any]], arm_name: str
) -> List[Dict[str, Any]]:
    return [row for row in rows if row.get("arm") == arm_name]


def _mean_key(
    rows: List[Dict[str, Any]], key: str, default: float = 0.0
) -> float:
    values = [row.get(key) for row in rows if row.get(key) is not None]
    return float(_mean(values, default) or default)


def _evaluate(arm_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    arm_names = [arm["arm"] for arm in ARMS]
    entropy_by_arm: Dict[str, float] = {}
    coverage_by_arm: Dict[str, float] = {}

    for arm_name in arm_names:
        rows = _arm_rows(arm_results, arm_name)
        entropy_by_arm[arm_name] = _mean_key(
            rows, "selected_action_class_entropy"
        )
        coverage_by_arm[arm_name] = _mean_key(rows, "state_coverage_fraction")

    arm0_entropy = entropy_by_arm.get("ARM_0_sp_cem_baseline", 0.0)
    arm8_entropy = entropy_by_arm.get("ARM_8_combined_5x", 0.0)
    arm9_entropy = entropy_by_arm.get("ARM_9_combined_10x", 0.0)

    # P1: any diversity arm shows entropy lift > ARM_0 + 0.05
    diversity_arms = [n for n in arm_names if n != "ARM_0_sp_cem_baseline"]
    best_arm_name = max(
        diversity_arms,
        key=lambda n: entropy_by_arm.get(n, 0.0),
    )
    best_arm_entropy = entropy_by_arm.get(best_arm_name, 0.0)
    p1 = bool(best_arm_entropy > arm0_entropy + P1_ENTROPY_MARGIN)

    # P2: best arm entropy > threshold
    p2 = bool(best_arm_entropy > P2_ENTROPY_THRESHOLD)

    # P3: combined 10x >= combined 5x (monotone)
    p3 = bool(arm9_entropy >= arm8_entropy)

    # Per-arm entropy deltas (all vs ARM_0)
    arm_entropy_deltas: Dict[str, float] = {}
    for arm_name in arm_names:
        delta = entropy_by_arm.get(arm_name, 0.0) - arm0_entropy
        arm_entropy_deltas[f"delta_{arm_name}"] = round(delta, 6)

    summary: Dict[str, Any] = {
        "arm0_entropy": round(arm0_entropy, 6),
        "best_diversity_arm": best_arm_name,
        "best_diversity_arm_entropy": round(best_arm_entropy, 6),
        "p1_any_arm_entropy_lift": p1,
        "p2_best_arm_above_threshold": p2,
        "p3_combined_10x_ge_5x": p3,
        "overall_pass": bool(p1),
    }
    # Add per-arm entropies
    for arm_name in arm_names:
        summary[f"entropy_{arm_name}"] = round(
            entropy_by_arm.get(arm_name, 0.0), 6
        )
        summary[f"coverage_{arm_name}"] = round(
            coverage_by_arm.get(arm_name, 0.0), 6
        )
    summary.update(arm_entropy_deltas)

    return summary


def _evidence_direction_per_claim(
    summary: Dict[str, Any],
) -> Dict[str, str]:
    """Determine per-claim evidence direction from sweep results."""
    arm0_entropy = summary.get("arm0_entropy", 0.0)

    # MECH-313 (noise_floor): ARM_1/2/3 -- does any show lift?
    m313_entropies = [
        summary.get("entropy_ARM_1_noise_floor_1x", 0.0),
        summary.get("entropy_ARM_2_noise_floor_5x", 0.0),
        summary.get("entropy_ARM_3_noise_floor_10x", 0.0),
    ]
    m313_best = max(m313_entropies)
    m313_lifts = bool(m313_best > arm0_entropy + P1_ENTROPY_MARGIN)

    # MECH-314 (curiosity): ARM_4/5
    m314_entropies = [
        summary.get("entropy_ARM_4_curiosity_5x", 0.0),
        summary.get("entropy_ARM_5_curiosity_10x", 0.0),
    ]
    m314_best = max(m314_entropies)
    m314_lifts = bool(m314_best > arm0_entropy + P1_ENTROPY_MARGIN)

    # MECH-320 (tonic_vigor): ARM_6/7
    m320_entropies = [
        summary.get("entropy_ARM_6_tonic_vigor_5x", 0.0),
        summary.get("entropy_ARM_7_tonic_vigor_10x", 0.0),
    ]
    m320_best = max(m320_entropies)
    m320_lifts = bool(m320_best > arm0_entropy + P1_ENTROPY_MARGIN)

    # ARC-065: supports if any arm lifts entropy
    arc065_lifts = bool(
        summary.get("p1_any_arm_entropy_lift", False)
    )

    return {
        "ARC-065": "supports" if arc065_lifts else "weakens",
        "MECH-313": "supports" if m313_lifts else "does_not_support",
        "MECH-314": "supports" if m314_lifts else "does_not_support",
        "MECH-320": "supports" if m320_lifts else "does_not_support",
    }


def run_experiment(dry_run: bool = False) -> Dict[str, Any]:
    seeds = DRY_RUN_SEEDS if dry_run else SEEDS
    episodes = DRY_RUN_EPISODES if dry_run else EVAL_EPISODES
    steps = DRY_RUN_STEPS if dry_run else STEPS_PER_EPISODE

    arm_results: List[Dict[str, Any]] = []
    for arm in ARMS:
        for seed in seeds:
            print(f"Seed {seed} Condition {arm['arm']}", flush=True)
            cell = _run_arm_seed(arm, seed, episodes, steps)
            arm_results.append(cell)
            passed = True  # always emit verdict (per-seed runs always complete)
            print(
                f"verdict: {'PASS' if passed else 'FAIL'}",
                flush=True,
            )

    summary = _evaluate(arm_results)
    outcome = "PASS" if summary["overall_pass"] else "FAIL"
    edpc = _evidence_direction_per_claim(summary)
    overall_direction = "supports" if outcome == "PASS" else "weakens"

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{timestamp}_v3"

    manifest: Dict[str, Any] = {
        "run_id": run_id,
        "queue_id": QUEUE_ID,
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "timestamp_utc": timestamp,
        "outcome": outcome,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "claim_ids": CLAIM_IDS,
        "evidence_direction": overall_direction,
        "evidence_direction_per_claim": edpc,
        "evidence_direction_note": (
            "10-arm bias-scale sweep (ARM_0 SP-CEM baseline, ARM_1-3 "
            "MECH-313 at 1x/5x/10x, ARM_4-5 MECH-314 at 5x/10x, "
            "ARM_6-7 MECH-320 at 5x/10x, ARM_8-9 combined at 5x/10x). "
            "PASS = at least one diversity arm shows selected_action_entropy "
            "> ARM_0 + 0.05. Follow-up to EXQ-571 diagnosis that F dominates "
            "88-89% of temporal variance and all bias components = 0."
        ),
        "dry_run": dry_run,
        "config": {
            "seeds": seeds,
            "eval_episodes": episodes,
            "steps_per_episode": steps,
            "env_size": ENV_SIZE,
            "std_floor": STD_FLOOR,
            "p1_entropy_margin": P1_ENTROPY_MARGIN,
            "p2_entropy_threshold": P2_ENTROPY_THRESHOLD,
            "arms": [arm["arm"] for arm in ARMS],
            "scale_factors": {
                "noise_floor_1x_alpha": NOISE_FLOOR_ALPHA_1X,
                "noise_floor_5x_alpha": round(NOISE_FLOOR_ALPHA_1X * 5.0, 4),
                "noise_floor_10x_alpha": round(NOISE_FLOOR_ALPHA_1X * 10.0, 4),
                "curiosity_1x_bias_scale": CURIOSITY_BIAS_SCALE_1X,
                "curiosity_5x_bias_scale": round(CURIOSITY_BIAS_SCALE_1X * 5.0, 4),
                "curiosity_10x_bias_scale": round(CURIOSITY_BIAS_SCALE_1X * 10.0, 4),
                "tonic_vigor_1x_bias_scale": TONIC_VIGOR_BIAS_SCALE_1X,
                "tonic_vigor_5x_bias_scale": round(TONIC_VIGOR_BIAS_SCALE_1X * 5.0, 4),
                "tonic_vigor_10x_bias_scale": round(TONIC_VIGOR_BIAS_SCALE_1X * 10.0, 4),
            },
        },
        "acceptance_criteria": {
            "P1_any_arm_entropy_lift": summary["p1_any_arm_entropy_lift"],
            "P2_best_arm_above_threshold": summary["p2_best_arm_above_threshold"],
            "P3_combined_10x_ge_5x": summary["p3_combined_10x_ge_5x"],
        },
        "summary": summary,
        "arm_results": arm_results,
    }

    out_dir = REPO_ROOT.parent / "REE_assembly" / "evidence" / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_id}.json"

    if not dry_run:
        out_path = write_flat_manifest(
            manifest,
            out_dir,
            dry_run=False,
            config=manifest.get("config"),
            seeds=SEEDS,
            script_path=Path(__file__),
        )
        print(f"Manifest written: {out_path}", flush=True)
    else:
        out_path = Path("/dev/null")
        print("Dry run -- manifest not written.", flush=True)

    print(f"Outcome: {outcome}", flush=True)
    for key, value in manifest["acceptance_criteria"].items():
        print(f"  {key}: {value}", flush=True)
    print(f"  Best arm: {summary.get('best_diversity_arm', 'n/a')}", flush=True)
    print(
        f"  Best arm entropy: {summary.get('best_diversity_arm_entropy', 0.0):.4f}"
        f" vs ARM_0: {summary.get('arm0_entropy', 0.0):.4f}",
        flush=True,
    )

    manifest["manifest_path"] = str(out_path)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="V3-EXQ-573 ARC-065 bias-scale 5-10x sweep"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Short smoke run; no manifest written.",
    )
    args = parser.parse_args()

    result = run_experiment(dry_run=args.dry_run)

    if args.dry_run:
        sys.exit(0)

    _outcome_raw = str(result.get("outcome", "FAIL")).upper()
    _outcome = _outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL"
    emit_outcome(
        outcome=_outcome,
        manifest_path=str(result.get("manifest_path", Path("/dev/null"))),
    )
    sys.exit(0)
