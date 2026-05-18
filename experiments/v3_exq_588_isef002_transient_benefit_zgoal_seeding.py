"""
V3-EXQ-588: EXQ-ISEF-002 Transient Benefit Patches -- z_goal Seeding Rate.

infant_substrate:GAP-11 closure experiment.

Scientific question: does transient_benefit_enabled=True (multiplier=3.0) produce
earlier z_goal.norm() threshold crossing than uniform resource placement?

The z_goal.norm() > 0.4 criterion is the primary blocking gate for the
infant-to-childhood transition (DEV-NEED-006). If transient benefit patches
accelerate z_goal seeding via the MECH-189 wanting pathway, that gate can
be met earlier and more reliably.

Design:
  ARM_0_control: transient_benefit_enabled=False (standard uniform resource)
  ARM_1_treatment: transient_benefit_enabled=True, transient_benefit_multiplier=3.0

Both arms share the same agent config (z_goal_enabled=True, alpha_world=0.9).
Agent persists across all 1000 episodes within a seed -- goal state accumulates
naturally as in the developmental model.

N=5 seeds x 2 arms = 10 runs; 1000 episodes x 200 steps per run.

Interpretation grid:
  Outcome                                      | Diagnosis / next action
  ---------------------------------------------|--------------------------------------
  C1 PASS: ARM_1 median first-crossing         | Transient patches effective; enable
    < 0.7x ARM_0                               |   as infant default; proceed to
                                               |   7-criterion gate update (GAP-15)
  C1 FAIL: similar first-crossing both arms    | Patches alone insufficient; z_goal
                                               |   encoder or MECH-189 write path may
                                               |   be the bottleneck; queue goal-
                                               |   seeding pipeline diagnostic
  C1 FAIL: ARM_1 crosses but ARM_0 does not   | Patches necessary but not at the
    within 1000 eps                            |   right multiplier; try higher
                                               |   transient_benefit_multiplier or
                                               |   combine with SD-049 multi-resource
  C3 FAIL: z_goal_active_fraction similar      | Single resource type dominating;
    despite C1 PASS                            |   need SD-049 multi-resource to
                                               |   diversify goal anchors
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

from experiment_protocol import emit_outcome  # noqa: E402
from ree_core.agent import REEAgent  # noqa: E402
from ree_core.environment.causal_grid_world import CausalGridWorldV2  # noqa: E402
from ree_core.utils.config import REEConfig  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_588_isef002_transient_benefit_zgoal_seeding"
QUEUE_ID = "V3-EXQ-588"
CLAIM_IDS: List[str] = ["MECH-189"]
EXPERIMENT_PURPOSE = "evidence"

SEEDS = [42, 43, 44, 45, 46]
BODY_OBS_DIM = 12
WORLD_OBS_DIM = 250
ACTION_DIM = 4
N_EPISODES = 1000
STEPS_PER_EPISODE = 200

ARM_NAMES = ["ARM_0_control", "ARM_1_treatment"]

# Pre-registered acceptance thresholds
Z_GOAL_CROSSING_THRESHOLD = 0.4   # z_goal.norm() target
CROSSING_SENTINEL = N_EPISODES + 1  # assigned when never crosses
C1_TREATMENT_FRACTION = 0.7  # treatment median < this fraction of control median


def _extract_obs(obs_dict: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
    obs_body = obs_dict["body_state"].float()
    if obs_body.shape[0] < BODY_OBS_DIM:
        obs_body = torch.cat([obs_body, torch.zeros(BODY_OBS_DIM - obs_body.shape[0])])
    elif obs_body.shape[0] > BODY_OBS_DIM:
        obs_body = obs_body[:BODY_OBS_DIM]

    obs_world = obs_dict["world_state"].float()
    if obs_world.shape[0] < WORLD_OBS_DIM:
        obs_world = torch.cat([obs_world, torch.zeros(WORLD_OBS_DIM - obs_world.shape[0])])
    elif obs_world.shape[0] > WORLD_OBS_DIM:
        obs_world = obs_world[:WORLD_OBS_DIM]

    return obs_body, obs_world


def _benefit_and_drive(obs_body: torch.Tensor) -> Tuple[float, float]:
    benefit = float(obs_body[11].item()) if obs_body.shape[0] > 11 else 0.0
    energy = float(obs_body[3].item()) if obs_body.shape[0] > 3 else 0.5
    drive = max(0.0, min(1.0, 1.0 - energy))
    return benefit, drive


def _build_agent() -> REEAgent:
    cfg = REEConfig.from_dims(
        body_obs_dim=BODY_OBS_DIM,
        world_obs_dim=WORLD_OBS_DIM,
        action_dim=ACTION_DIM,
        z_goal_enabled=True,
        drive_weight=2.0,
    )
    cfg.latent.alpha_world = 0.9
    return REEAgent(cfg)


def _build_env(*, transient_benefit_enabled: bool, seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        seed=seed,
        resource_respawn_on_consume=True,
        transient_benefit_enabled=transient_benefit_enabled,
        transient_benefit_multiplier=3.0 if transient_benefit_enabled else 2.0,
    )


def _run_arm(
    *,
    seed: int,
    arm_name: str,
    transient_benefit_enabled: bool,
    dry_run: bool,
) -> Dict[str, Any]:
    torch.manual_seed(seed)

    agent = _build_agent()
    env = _build_env(transient_benefit_enabled=transient_benefit_enabled, seed=seed)

    n_episodes = 2 if dry_run else N_EPISODES

    first_crossing_episode: Optional[int] = None
    z_goal_norms: List[float] = []
    transient_contacts_per_ep: List[int] = []
    active_episode_count: int = 0

    for ep in range(n_episodes):
        _flat, obs_dict = env.reset()
        obs_body, obs_world = _extract_obs(obs_dict)

        ep_transient_contacts: int = 0

        for _step in range(STEPS_PER_EPISODE):
            with torch.no_grad():
                action = agent.act_with_split_obs(
                    obs_body=obs_body, obs_world=obs_world
                )

            _flat, harm_signal, done, info, obs_dict = env.step(action)

            agent.update_residue(float(harm_signal))

            obs_body, obs_world = _extract_obs(obs_dict)
            benefit, drive = _benefit_and_drive(obs_body)
            agent.update_z_goal(benefit_exposure=benefit, drive_level=drive)

            ep_transient_contacts += int(
                float(info.get("transient_benefit_contact_this_tick", 0.0)) > 0.0
            )

            if done:
                _flat, obs_dict = env.reset()
                obs_body, obs_world = _extract_obs(obs_dict)

        z_norm = agent.goal_state.goal_norm() if agent.goal_state is not None else 0.0
        is_active = (
            agent.goal_state.is_active() if agent.goal_state is not None else False
        )

        z_goal_norms.append(z_norm)
        transient_contacts_per_ep.append(ep_transient_contacts)
        if is_active:
            active_episode_count += 1

        if first_crossing_episode is None and z_norm >= Z_GOAL_CROSSING_THRESHOLD:
            first_crossing_episode = ep

        if (ep + 1) % 100 == 0 or (ep + 1) == n_episodes:
            print(
                f"  [train] {arm_name} seed={seed} ep {ep + 1}/{n_episodes} "
                f"z_goal_norm={z_norm:.4f} "
                f"contacts_ep={ep_transient_contacts}",
                flush=True,
            )

    if first_crossing_episode is None:
        first_crossing_episode = CROSSING_SENTINEL

    mean_contacts = (
        sum(transient_contacts_per_ep) / len(transient_contacts_per_ep)
        if transient_contacts_per_ep else 0.0
    )
    active_fraction = active_episode_count / n_episodes if n_episodes > 0 else 0.0
    final_z_norm = z_goal_norms[-1] if z_goal_norms else 0.0

    passed = first_crossing_episode < CROSSING_SENTINEL
    print(
        f"verdict: {'PASS' if passed else 'FAIL'}",
        flush=True,
    )

    return {
        "arm": arm_name,
        "first_crossing_episode": first_crossing_episode,
        "final_z_goal_norm": final_z_norm,
        "mean_transient_contacts_per_ep": mean_contacts,
        "active_episode_fraction": active_fraction,
        "z_goal_norms_last10": z_goal_norms[-10:] if not dry_run else z_goal_norms,
        "transient_contacts_last10": (
            transient_contacts_per_ep[-10:] if not dry_run else transient_contacts_per_ep
        ),
    }


def _run_seed(*, seed: int, dry_run: bool) -> Dict[str, Any]:
    arm_results: Dict[str, Dict[str, Any]] = {}
    for arm_name, tb_enabled in [
        ("ARM_0_control", False),
        ("ARM_1_treatment", True),
    ]:
        print(f"Seed {seed} Condition {arm_name}", flush=True)
        result = _run_arm(
            seed=seed,
            arm_name=arm_name,
            transient_benefit_enabled=tb_enabled,
            dry_run=dry_run,
        )
        arm_results[arm_name] = result
    return {"seed": seed, "arm_results": arm_results}


def _median(vals: List[int]) -> float:
    if not vals:
        return float(CROSSING_SENTINEL)
    s = sorted(vals)
    n = len(s)
    if n % 2 == 1:
        return float(s[n // 2])
    return (s[n // 2 - 1] + s[n // 2]) / 2.0


def run_experiment(*, dry_run: bool = False) -> Dict[str, Any]:
    seeds = [SEEDS[0]] if dry_run else SEEDS
    print(
        f"V3-EXQ-588: EXQ-ISEF-002 transient benefit z_goal seeding rate",
        flush=True,
    )
    print(
        f"  dry_run={dry_run} seeds={seeds} "
        f"n_episodes={2 if dry_run else N_EPISODES} "
        f"steps={STEPS_PER_EPISODE} z_cross_thr={Z_GOAL_CROSSING_THRESHOLD}",
        flush=True,
    )

    all_seed_results: List[Dict[str, Any]] = []
    for seed in seeds:
        result = _run_seed(seed=seed, dry_run=dry_run)
        all_seed_results.append(result)

    arm0_crossings: List[int] = [
        r["arm_results"]["ARM_0_control"]["first_crossing_episode"]
        for r in all_seed_results
    ]
    arm1_crossings: List[int] = [
        r["arm_results"]["ARM_1_treatment"]["first_crossing_episode"]
        for r in all_seed_results
    ]
    arm0_contacts: List[float] = [
        r["arm_results"]["ARM_0_control"]["mean_transient_contacts_per_ep"]
        for r in all_seed_results
    ]
    arm1_contacts: List[float] = [
        r["arm_results"]["ARM_1_treatment"]["mean_transient_contacts_per_ep"]
        for r in all_seed_results
    ]
    arm0_active: List[float] = [
        r["arm_results"]["ARM_0_control"]["active_episode_fraction"]
        for r in all_seed_results
    ]
    arm1_active: List[float] = [
        r["arm_results"]["ARM_1_treatment"]["active_episode_fraction"]
        for r in all_seed_results
    ]

    def _mean_f(xs: List[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    arm0_median = _median(arm0_crossings)
    arm1_median = _median(arm1_crossings)
    arm0_contacts_mean = _mean_f(arm0_contacts)
    arm1_contacts_mean = _mean_f(arm1_contacts)
    arm0_active_mean = _mean_f(arm0_active)
    arm1_active_mean = _mean_f(arm1_active)

    threshold = C1_TREATMENT_FRACTION * arm0_median

    if dry_run:
        c1_pass = True
        c2_pass = True
        c3_pass = True
    else:
        c1_pass = arm1_median < threshold
        c2_pass = arm1_contacts_mean > arm0_contacts_mean
        c3_pass = arm1_active_mean > arm0_active_mean

    outcome = "PASS" if c1_pass else "FAIL"

    print("", flush=True)
    print(f"ARM_0 first-crossing median: {arm0_median:.1f}", flush=True)
    print(f"ARM_1 first-crossing median: {arm1_median:.1f}", flush=True)
    print(
        f"C1 threshold (ARM_1 < {C1_TREATMENT_FRACTION:.2f} x ARM_0={arm0_median:.1f}): "
        f"{threshold:.1f}",
        flush=True,
    )
    print(
        f"C1 (ARM_1 median first-crossing < {C1_TREATMENT_FRACTION}x ARM_0): "
        f"{'PASS' if c1_pass else 'FAIL'}",
        flush=True,
    )
    print(
        f"C2 [info] (ARM_1 mean contacts > ARM_0): "
        f"{arm1_contacts_mean:.3f} vs {arm0_contacts_mean:.3f} -> "
        f"{'YES' if c2_pass else 'NO'}",
        flush=True,
    )
    print(
        f"C3 [info] (ARM_1 active fraction > ARM_0): "
        f"{arm1_active_mean:.3f} vs {arm0_active_mean:.3f} -> "
        f"{'YES' if c3_pass else 'NO'}",
        flush=True,
    )
    print(f"Overall outcome: {outcome}", flush=True)

    return {
        "outcome": outcome,
        "c1_pass": c1_pass,
        "c2_pass": c2_pass,
        "c3_pass": c3_pass,
        "arm0_median_first_crossing": arm0_median,
        "arm1_median_first_crossing": arm1_median,
        "c1_threshold": threshold,
        "arm0_crossings_per_seed": arm0_crossings,
        "arm1_crossings_per_seed": arm1_crossings,
        "arm0_mean_contacts_per_ep": arm0_contacts_mean,
        "arm1_mean_contacts_per_ep": arm1_contacts_mean,
        "arm0_mean_active_fraction": arm0_active_mean,
        "arm1_mean_active_fraction": arm1_active_mean,
        "all_seed_results": all_seed_results,
    }


def main(*, dry_run: bool = False) -> Tuple[str, Path]:
    result = run_experiment(dry_run=dry_run)
    outcome = result["outcome"]

    run_id = (
        f"{EXPERIMENT_TYPE}_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}_v3"
    )
    out_dir = (
        REPO_ROOT.parent
        / "REE_assembly"
        / "evidence"
        / "experiments"
        / EXPERIMENT_TYPE
    )
    out_path = out_dir / f"{run_id}.json"

    c1_dir = "supports" if result["c1_pass"] else "does_not_support"

    manifest = {
        "run_id": run_id,
        "queue_id": QUEUE_ID,
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "outcome": outcome,
        "evidence_direction": c1_dir,
        "evidence_direction_per_claim": {
            "MECH-189": c1_dir,
        },
        "timestamp_utc": datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"),
        "dry_run": dry_run,
        "config": {
            "seeds": list(SEEDS if not dry_run else [SEEDS[0]]),
            "n_episodes": N_EPISODES if not dry_run else 2,
            "steps_per_episode": STEPS_PER_EPISODE,
            "z_goal_crossing_threshold": Z_GOAL_CROSSING_THRESHOLD,
            "crossing_sentinel": CROSSING_SENTINEL,
            "c1_treatment_fraction": C1_TREATMENT_FRACTION,
            "arm0_transient_benefit_enabled": False,
            "arm1_transient_benefit_enabled": True,
            "arm1_transient_benefit_multiplier": 3.0,
            "alpha_world": 0.9,
            "z_goal_enabled": True,
            "drive_weight": 2.0,
        },
        "acceptance_criteria": {
            "C1_primary_gate": (
                f"ARM_1 median first-crossing < {C1_TREATMENT_FRACTION}x ARM_0 median "
                f"(sentinel={CROSSING_SENTINEL} when never crosses)"
            ),
            "C2_secondary_info": "ARM_1 mean transient_contacts_per_ep > ARM_0",
            "C3_secondary_info": "ARM_1 active_episode_fraction > ARM_0",
        },
        "criteria_results": {
            "C1_pass": result["c1_pass"],
            "C2_contacts_info": result["c2_pass"],
            "C3_active_fraction_info": result["c3_pass"],
        },
        "metrics": {
            "arm0_median_first_crossing_episode": result["arm0_median_first_crossing"],
            "arm1_median_first_crossing_episode": result["arm1_median_first_crossing"],
            "c1_threshold": result["c1_threshold"],
            "arm0_crossings_per_seed": result["arm0_crossings_per_seed"],
            "arm1_crossings_per_seed": result["arm1_crossings_per_seed"],
            "arm0_mean_transient_contacts_per_ep": result["arm0_mean_contacts_per_ep"],
            "arm1_mean_transient_contacts_per_ep": result["arm1_mean_contacts_per_ep"],
            "arm0_mean_active_fraction": result["arm0_mean_active_fraction"],
            "arm1_mean_active_fraction": result["arm1_mean_active_fraction"],
        },
        "per_seed_results": result["all_seed_results"],
        "notes": (
            "infant_substrate:GAP-11 closure. EXQ-ISEF-002 z_goal seeding rate "
            "comparison: uniform resource placement (ARM_0) vs transient benefit "
            "patches multiplier=3.0 (ARM_1). Primary criterion: ARM_1 median "
            "first z_goal.norm() > 0.4 crossing episode < 0.7x ARM_0 median. "
            "Agent persists across all 1000 episodes per seed (goal state "
            "accumulates -- developmental model). C2/C3 are informational: "
            "C2 confirms patches generate measurable transient contacts; C3 "
            "confirms broader goal-state activation in treatment. PASS enables "
            "enabling transient patches as infant default and proceeding to "
            "7-criterion gate update (GAP-15). FAIL triggers goal-seeding "
            "pipeline diagnostic (MECH-189 write path)."
        ),
    }

    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"Result written to: {out_path}", flush=True)
    else:
        print("[dry-run] manifest not written", flush=True)
        print(json.dumps(manifest, indent=2), flush=True)

    return outcome, out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    _outcome, _out_path = main(dry_run=args.dry_run)

    emit_outcome(
        outcome=_outcome if _outcome in ("PASS", "FAIL") else "FAIL",
        manifest_path=_out_path,
    )
    sys.exit(0)
