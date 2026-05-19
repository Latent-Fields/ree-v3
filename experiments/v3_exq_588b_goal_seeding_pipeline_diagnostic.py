"""
V3-EXQ-588b: Goal-seeding pipeline diagnostic (post EXQ-588 autopsy).

EXPERIMENT_PURPOSE = diagnostic (claim_ids=[]).

Scientific question: In the EXQ-588 developmental regime (persistent agent,
z_goal_enabled, alpha_world=0.9), does SD-012 drive_floor=0.9 restore contact-
time seeding (replicating EXQ-582a), and do transient benefit patches (GAP-3)
add contact density and z_goal norm lift beyond floor alone?

Arms (3 x 3 seeds):
  ARM_OFF:           drive_floor=0.0, transient_benefit_enabled=False
  ARM_FLOOR:         drive_floor=0.9, transient_benefit_enabled=False
  ARM_FLOOR_PATCH:   drive_floor=0.9, transient_benefit_enabled=True, multiplier=3.0

Agent persists across all episodes within a seed x arm (matches EXQ-588).
Instrumentation at every resource / transient-benefit contact step logs
benefit_exposure, drive_level, drive_trace, effective_benefit, seeding_fired,
z_goal.norm().

=== HYPOTHESES UNDER TEST ===

H1: Without drive_floor, effective_benefit stays below benefit_threshold at
    contact (EXQ-588 / 582 OFF-arm collapse).
H2: drive_floor=0.9 alone restores seeding in >=2/3 seeds post-warmup.
H3: floor + transient patches increase post-warmup contacts vs floor-only.

=== INTERPRETATION GRID ===

| Outcome                                      | Diagnosis / next action |
|----------------------------------------------|-------------------------|
| A1-A3 PASS, A4 PASS                          | Pipeline unblocked; optional EXQ-588c retest z_goal.norm>0.4 gate with floor+patches |
| A1 fails (OFF arm fires seeding)             | Regime drift; STOP and reconcile env vs 582a anchor |
| A2 fails (floor alone no seeding)            | Escalate goal_pipeline (lower benefit_threshold or MECH-216) |
| A3 fails but A2 passes                     | Patches not needed for seeding; floor is sufficient |
| A4 fails (patches do not add contacts)       | Env GAP-3 contact rate too low; raise prob/multiplier |

architecture_epoch: ree_hybrid_guardrails_v1
supersedes: V3-EXQ-588
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

EXPERIMENT_TYPE = "v3_exq_588b_goal_seeding_pipeline_diagnostic"
QUEUE_ID = "V3-EXQ-588b"
CLAIM_IDS: List[str] = []
EXPERIMENT_PURPOSE = "diagnostic"

SEEDS = [42, 43, 44]
BODY_OBS_DIM = 12
WORLD_OBS_DIM = 250
ACTION_DIM = 4
N_EPISODES = 300
WARMUP_EPISODES = 50
STEPS_PER_EPISODE = 200
Z_GOAL_CROSSING_THRESHOLD = 0.4
CROSSING_SENTINEL = N_EPISODES + 1

ARM_SPECS = [
    ("ARM_OFF", 0.0, False),
    ("ARM_FLOOR", 0.9, False),
    ("ARM_FLOOR_PATCH", 0.9, True),
]

# Pre-registered acceptance (post-warmup window: ep >= WARMUP_EPISODES).
A2_MIN_SEEDS_WITH_SEEDING = 2
A4_CONTACT_LIFT_RATIO = 1.1


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


def _build_agent(*, drive_floor: float) -> REEAgent:
    cfg = REEConfig.from_dims(
        body_obs_dim=BODY_OBS_DIM,
        world_obs_dim=WORLD_OBS_DIM,
        action_dim=ACTION_DIM,
        z_goal_enabled=True,
        drive_weight=2.0,
        drive_ema_alpha=1.0,
        drive_floor=drive_floor,
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


def _is_contact_step(info: Dict[str, Any]) -> bool:
    ttype = str(info.get("transition_type", "none"))
    if ttype in ("resource", "benefit_approach"):
        return True
    return float(info.get("transient_benefit_contact_this_tick", 0.0)) > 0.0


def _run_arm(
    *,
    seed: int,
    arm_name: str,
    drive_floor: float,
    transient_benefit_enabled: bool,
    dry_run: bool,
) -> Dict[str, Any]:
    torch.manual_seed(seed)

    agent = _build_agent(drive_floor=drive_floor)
    env = _build_env(transient_benefit_enabled=transient_benefit_enabled, seed=seed)

    n_episodes = 2 if dry_run else N_EPISODES
    warmup_eps = 0 if dry_run else WARMUP_EPISODES
    steps_per_episode = 20 if dry_run else STEPS_PER_EPISODE

    benefit_threshold = float(agent.config.goal.benefit_threshold)
    drive_weight = float(agent.config.goal.drive_weight)
    seeding_gain = float(agent.config.goal.z_goal_seeding_gain)

    first_crossing_episode: Optional[int] = None
    z_goal_norms: List[float] = []
    max_z_goal_norm = 0.0

    n_contacts_all = 0
    n_contacts_post_warmup = 0
    n_seedings_all = 0
    n_seedings_post_warmup = 0
    n_transient_contacts_all = 0

    eff_on_contact: List[float] = []
    eff_on_contact_post_warmup: List[float] = []
    contact_log: List[Dict[str, Any]] = []

    for ep in range(n_episodes):
        _flat, obs_dict = env.reset()
        obs_body, obs_world = _extract_obs(obs_dict)
        ep_transient_contacts = 0
        post_warmup = ep >= warmup_eps

        for _step in range(steps_per_episode):
            with torch.no_grad():
                action = agent.act_with_split_obs(
                    obs_body=obs_body, obs_world=obs_world
                )

            _flat, harm_signal, done, info, obs_dict = env.step(action)

            agent.update_residue(float(harm_signal))

            obs_body, obs_world = _extract_obs(obs_dict)
            benefit, drive = _benefit_and_drive(obs_body)
            agent.update_z_goal(benefit_exposure=benefit, drive_level=drive)

            if float(info.get("transient_benefit_contact_this_tick", 0.0)) > 0.0:
                ep_transient_contacts += 1
                n_transient_contacts_all += 1

            if _is_contact_step(info):
                trace = float(agent.goal_state._drive_trace)
                eff = benefit * seeding_gain * (1.0 + drive_weight * trace)
                z_norm = (
                    agent.goal_state.goal_norm()
                    if agent.goal_state is not None
                    else 0.0
                )
                fired = eff > benefit_threshold
                n_contacts_all += 1
                if fired:
                    n_seedings_all += 1
                eff_on_contact.append(eff)
                if post_warmup:
                    n_contacts_post_warmup += 1
                    eff_on_contact_post_warmup.append(eff)
                    if fired:
                        n_seedings_post_warmup += 1
                if len(contact_log) < 200:
                    contact_log.append(
                        {
                            "ep": ep,
                            "step": _step,
                            "post_warmup": post_warmup,
                            "benefit_exposure": benefit,
                            "drive_level": drive,
                            "drive_trace": trace,
                            "effective_benefit": eff,
                            "seeding_fired": fired,
                            "z_goal_norm": z_norm,
                            "transition_type": str(
                                info.get("transition_type", "none")
                            ),
                        }
                    )

            if done:
                _flat, obs_dict = env.reset()
                obs_body, obs_world = _extract_obs(obs_dict)

        z_norm = agent.goal_state.goal_norm() if agent.goal_state is not None else 0.0
        z_goal_norms.append(z_norm)
        max_z_goal_norm = max(max_z_goal_norm, z_norm)

        if first_crossing_episode is None and z_norm >= Z_GOAL_CROSSING_THRESHOLD:
            first_crossing_episode = ep

        if (ep + 1) % 100 == 0 or (ep + 1) == n_episodes:
            print(
                f"  [train] {arm_name} seed={seed} ep {ep + 1}/{n_episodes} "
                f"z_goal_norm={z_norm:.4f} seedings_pw={n_seedings_post_warmup}",
                flush=True,
            )

    if first_crossing_episode is None:
        first_crossing_episode = CROSSING_SENTINEL

    run_ok = n_contacts_all > 0 or dry_run
    print(f"verdict: {'PASS' if run_ok else 'FAIL'}", flush=True)

    return {
        "arm": arm_name,
        "drive_floor": drive_floor,
        "transient_benefit_enabled": transient_benefit_enabled,
        "seed": seed,
        "benefit_threshold": benefit_threshold,
        "first_crossing_episode": first_crossing_episode,
        "final_z_goal_norm": z_goal_norms[-1] if z_goal_norms else 0.0,
        "max_z_goal_norm": max_z_goal_norm,
        "n_contacts_all": n_contacts_all,
        "n_contacts_post_warmup": n_contacts_post_warmup,
        "n_seedings_all": n_seedings_all,
        "n_seedings_post_warmup": n_seedings_post_warmup,
        "n_transient_contacts_all": n_transient_contacts_all,
        "mean_effective_benefit_on_contact": (
            sum(eff_on_contact) / len(eff_on_contact) if eff_on_contact else 0.0
        ),
        "mean_effective_benefit_on_contact_post_warmup": (
            sum(eff_on_contact_post_warmup) / len(eff_on_contact_post_warmup)
            if eff_on_contact_post_warmup
            else 0.0
        ),
        "contact_log_sample": contact_log,
    }


def _arm_aggregate(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    seeds_with_seeding_pw = sum(
        1 for r in runs if r["n_seedings_post_warmup"] > 0
    )
    return {
        "n_seeds": len(runs),
        "seeds_with_seeding_post_warmup": seeds_with_seeding_pw,
        "total_seedings_post_warmup": sum(
            r["n_seedings_post_warmup"] for r in runs
        ),
        "mean_n_contacts_post_warmup": (
            sum(r["n_contacts_post_warmup"] for r in runs) / len(runs)
            if runs
            else 0.0
        ),
        "mean_max_z_goal_norm": (
            sum(r["max_z_goal_norm"] for r in runs) / len(runs) if runs else 0.0
        ),
        "median_first_crossing": sorted(
            r["first_crossing_episode"] for r in runs
        )[len(runs) // 2]
        if runs
        else CROSSING_SENTINEL,
    }


def run_experiment(*, dry_run: bool = False) -> Dict[str, Any]:
    seeds = [SEEDS[0]] if dry_run else SEEDS
    arms = ARM_SPECS if not dry_run else ARM_SPECS

    print(
        f"V3-EXQ-588b: goal-seeding pipeline diagnostic",
        flush=True,
    )
    print(
        f"  dry_run={dry_run} seeds={seeds} n_episodes="
        f"{2 if dry_run else N_EPISODES} warmup={WARMUP_EPISODES}",
        flush=True,
    )

    all_runs: List[Dict[str, Any]] = []
    for arm_name, drive_floor, tb_enabled in arms:
        for seed in seeds:
            print(f"Seed {seed} Condition {arm_name}", flush=True)
            all_runs.append(
                _run_arm(
                    seed=seed,
                    arm_name=arm_name,
                    drive_floor=drive_floor,
                    transient_benefit_enabled=tb_enabled,
                    dry_run=dry_run,
                )
            )

    by_arm: Dict[str, Dict[str, Any]] = {}
    for arm_name, _, _ in ARM_SPECS:
        arm_runs = [r for r in all_runs if r["arm"] == arm_name]
        by_arm[arm_name] = _arm_aggregate(arm_runs)

    off = by_arm["ARM_OFF"]
    floor = by_arm["ARM_FLOOR"]
    patch = by_arm["ARM_FLOOR_PATCH"]

    a1 = off["total_seedings_post_warmup"] == 0
    a2 = floor["seeds_with_seeding_post_warmup"] >= A2_MIN_SEEDS_WITH_SEEDING
    a3 = patch["seeds_with_seeding_post_warmup"] >= A2_MIN_SEEDS_WITH_SEEDING
    floor_contacts = floor["mean_n_contacts_post_warmup"]
    patch_contacts = patch["mean_n_contacts_post_warmup"]
    a4 = patch_contacts >= A4_CONTACT_LIFT_RATIO * max(floor_contacts, 1e-9)

    outcome = "PASS" if (a1 and a2 and a3 and a4) else "FAIL"

    print("", flush=True)
    print(f"A1 OFF zero seedings post-warmup: {'PASS' if a1 else 'FAIL'}", flush=True)
    print(f"A2 FLOOR seeding >=2/3 seeds: {'PASS' if a2 else 'FAIL'}", flush=True)
    print(f"A3 FLOOR_PATCH seeding >=2/3 seeds: {'PASS' if a3 else 'FAIL'}", flush=True)
    print(
        f"A4 PATCH contacts {patch_contacts:.2f} vs FLOOR {floor_contacts:.2f}: "
        f"{'PASS' if a4 else 'FAIL'}",
        flush=True,
    )
    print(f"Overall outcome: {outcome}", flush=True)

    return {
        "outcome": outcome,
        "acceptance": {
            "A1_off_zero_seedings_post_warmup": a1,
            "A2_floor_seeding_ge_2_of_3": a2,
            "A3_floor_patch_seeding_ge_2_of_3": a3,
            "A4_patch_contacts_lift": a4,
        },
        "by_arm": by_arm,
        "all_runs": all_runs,
    }


def main(*, dry_run: bool = False) -> Tuple[str, Optional[Path]]:
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
    out_path: Optional[Path] = None

    manifest = {
        "run_id": run_id,
        "queue_id": QUEUE_ID,
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "supersedes": "V3-EXQ-588",
        "outcome": outcome,
        "evidence_direction": "non_contributory",
        "evidence_direction_note": (
            "Diagnostic (claim_ids=[]). Post EXQ-588 autopsy: tests whether "
            "drive_floor=0.9 restores GoalState seeding at contact and whether "
            "transient benefit patches add post-warmup contact density beyond "
            "floor alone. Excluded from governance claim scoring."
        ),
        "timestamp_utc": datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"),
        "dry_run": dry_run,
        "config": {
            "seeds": list(SEEDS if not dry_run else [SEEDS[0]]),
            "n_episodes": N_EPISODES if not dry_run else 2,
            "warmup_episodes": WARMUP_EPISODES if not dry_run else 0,
            "steps_per_episode": STEPS_PER_EPISODE,
            "arms": [
                {
                    "name": n,
                    "drive_floor": f,
                    "transient_benefit_enabled": t,
                }
                for n, f, t in ARM_SPECS
            ],
            "z_goal_crossing_threshold": Z_GOAL_CROSSING_THRESHOLD,
        },
        "acceptance_results": result["acceptance"],
        "by_arm": result["by_arm"],
        "per_run_results": result["all_runs"],
        "notes": (
            "Follow-on to failure_autopsy_V3-EXQ-588_2026-05-19. MECH-189 not tested. "
            "PASS unblocks optional EXQ-588c retest of infant z_goal.norm>0.4 gate."
        ),
    }

    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{run_id}.json"
        with open(out_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"Result written to: {out_path}", flush=True)
    else:
        print("[dry-run] manifest not written", flush=True)

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
