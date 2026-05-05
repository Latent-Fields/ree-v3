"""
V3-EXQ-517b: MECH-302 relief-completion discriminative pair -- longer episodes.

Supersedes V3-EXQ-517a.  EXQ-517 produced 0 events across all 3 seeds because the
5-step window / 0.05-drop threshold cannot resolve 0.002/step limb-healing dynamics.
The z_harm_a norm change over 5 steps of healing is consistently < 0.05.

Recalibration:
  window_length: 5  -> 30   (captures ~0.06 total healing per limb axis)
  drop_threshold: 0.05 -> 0.005  (10x more sensitive)
  min_initial_norm: 0.02 -> 0.01  (permissive gate; keeps spurious zero-input fires blocked)

Everything else (env, seeds, episodes, ARM_A/ARM_B structure, acceptance criteria) is
identical to EXQ-517.

ARM_A: use_suffering_derivative_comparator=True (MECH-302 ON)
ARM_B: use_suffering_derivative_comparator=False (MECH-302 OFF)

3 seeds x 2 conditions = 6 runs total.
P0 (30 ep): warmup -- agent explores with limb_damage_enabled (clear z_harm_a dynamics).
P1 (40 ep): measurement -- count relief-completion events and valence writes.

Acceptance criteria (both arms must pass for PASS fraction >= 2/3):
  C1: ARM_A p1_events >= C1_MIN_EVENTS_PER_SEED (comparator fires at all)
  C2: ARM_A p1_writes >= C2_MIN_WRITES_PER_SEED (valence write path intact)
  C3: ARM_B p1_events == 0 (no events when comparator disabled)
  C4: ARM_B p1_writes == 0 (no writes when comparator disabled)

PASS = at least 2/3 seeds satisfy all four criteria in ARM_A AND ARM_B.
PASS lifts gate (c) for MECH-304 conditioned-inhibition experiment.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.agent import REEAgent
from ree_core.utils.config import REEConfig

EXPERIMENT_TYPE = "v3_exq_517b_mech302_relief_completion_discriminative_pair"
CLAIM_IDS = ["MECH-302"]
EXPERIMENT_PURPOSE = "evidence"
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"

# Pre-registered thresholds
C1_MIN_EVENTS_PER_SEED = 3
C2_MIN_WRITES_PER_SEED = 2
PASS_FRACTION_REQUIRED = 2.0 / 3.0

SEEDS = [42, 43, 44]
P0_EPISODES = 30
P1_EPISODES = 40
TOTAL_EPISODES = P0_EPISODES + P1_EPISODES  # 70
STEPS_PER_EPISODE = 300
PRINT_INTERVAL = 10

GRID_SIZE = 12
N_HAZARDS = 3
N_RESOURCES = 2
BODY_OBS_DIM = 17   # limb_damage_enabled=True
WORLD_OBS_DIM = 250
HARM_OBS_A_DIM = 7  # limb_damage_enabled=True
ACTION_DIM = 5

# Recalibrated vs EXQ-517: wider window + lower threshold for 0.002/step healing
COMP_WINDOW_LENGTH = 30    # was 5
COMP_DROP_THRESHOLD = 0.005  # was 0.05
COMP_MIN_INITIAL_NORM = 0.01  # was 0.02


def make_config(arm_on: bool) -> REEConfig:
    return REEConfig.from_dims(
        body_obs_dim=BODY_OBS_DIM,
        world_obs_dim=WORLD_OBS_DIM,
        action_dim=ACTION_DIM,
        use_affective_harm_stream=True,
        harm_obs_a_dim=HARM_OBS_A_DIM,
        use_suffering_derivative_comparator=arm_on,
        suffering_window_length=COMP_WINDOW_LENGTH,
        suffering_drop_threshold=COMP_DROP_THRESHOLD,
        suffering_min_initial_norm=COMP_MIN_INITIAL_NORM,
        valence_liking_enabled=arm_on,
        relief_completion_weight=1.0,
    )


def make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        size=GRID_SIZE,
        num_hazards=N_HAZARDS,
        num_resources=N_RESOURCES,
        seed=seed,
        limb_damage_enabled=True,
        damage_increment=0.15,
        failure_prob_scale=0.3,
        heal_rate=0.002,
    )


def step_agent(agent, env, obs_dict):
    """One agent step: sense -> clock -> e1 -> generate -> select_action -> env.step."""
    body = obs_dict["body_state"]
    world = obs_dict["world_state"]
    if body.dim() == 1:
        body = body.unsqueeze(0)
    if world.dim() == 1:
        world = world.unsqueeze(0)

    sense_kwargs = {"obs_body": body, "obs_world": world}
    obs_harm = obs_dict.get("harm_obs")
    if obs_harm is not None:
        if obs_harm.dim() == 1:
            obs_harm = obs_harm.unsqueeze(0)
        sense_kwargs["obs_harm"] = obs_harm
    obs_harm_a = obs_dict.get("harm_obs_a")
    if obs_harm_a is not None:
        if obs_harm_a.dim() == 1:
            obs_harm_a = obs_harm_a.unsqueeze(0)
        sense_kwargs["obs_harm_a"] = obs_harm_a

    valence_write_count = [0]
    orig_uv = None
    if hasattr(agent, "residue_field") and hasattr(agent.residue_field, "update_valence"):
        orig_uv = agent.residue_field.update_valence

        def _patched_uv(*args, **kwargs):
            valence_write_count[0] += 1
            return orig_uv(*args, **kwargs)

        agent.residue_field.update_valence = _patched_uv

    with torch.no_grad():
        latent = agent.sense(**sense_kwargs)
        event_fired = bool(getattr(agent, "_relief_completion_event", False))
        ticks = agent.clock.advance()
        world_dim = agent.config.latent.world_dim
        e1_prior = (
            agent._e1_tick(latent)
            if ticks.get("e1_tick", True)
            else torch.zeros(1, world_dim)
        )
        candidates = agent.generate_trajectories(latent, e1_prior, ticks)
        action = agent.select_action(candidates, ticks)

    if orig_uv is not None:
        agent.residue_field.update_valence = orig_uv

    action_idx = int(action.argmax(dim=-1).item()) if action is not None else 0
    _flat, _harm_signal, done, _info, next_obs = env.step(action_idx)

    return {
        "event_fired": event_fired,
        "valence_writes": valence_write_count[0],
        "done": done,
        "next_obs": next_obs,
    }


def run_arm(seed: int, arm_label: str, arm_on: bool, dry_run: bool) -> dict:
    """Run one arm for one seed. Returns per-seed result dict."""
    print(f"Seed {seed} Condition {arm_label}", flush=True)

    cfg = make_config(arm_on)
    agent = REEAgent(cfg)
    agent.reset()

    p1_events = 0
    p1_writes = 0

    for ep in range(TOTAL_EPISODES):
        ep_seed = seed * 10000 + ep
        env = make_env(seed=ep_seed)
        _flat, obs_dict = env.reset()
        agent.reset()

        in_p1 = ep >= P0_EPISODES

        for _step in range(STEPS_PER_EPISODE):
            diag = step_agent(agent, env, obs_dict)
            if in_p1:
                p1_events += int(diag["event_fired"])
                p1_writes += int(diag["valence_writes"])
            obs_dict = diag["next_obs"]
            if diag["done"]:
                break

        if dry_run or (ep + 1) % PRINT_INTERVAL == 0:
            print(
                f"  [train] seed={seed} arm={arm_label} ep {ep+1}/{TOTAL_EPISODES} "
                f"p1_events={p1_events} p1_writes={p1_writes}",
                flush=True,
            )

        if dry_run and ep >= 2:
            break

    if arm_on:
        c1_pass = p1_events >= C1_MIN_EVENTS_PER_SEED
        c2_pass = p1_writes >= C2_MIN_WRITES_PER_SEED
        c3_pass = True
        c4_pass = True
    else:
        c1_pass = True
        c2_pass = True
        c3_pass = p1_events == 0
        c4_pass = p1_writes == 0

    seed_pass = c1_pass and c2_pass and c3_pass and c4_pass

    return {
        "seed": seed,
        "arm_label": arm_label,
        "arm_on": arm_on,
        "p1_episodes": P1_EPISODES,
        "p1_events": p1_events,
        "p1_writes": p1_writes,
        "c1_pass": c1_pass,
        "c2_pass": c2_pass,
        "c3_pass": c3_pass,
        "c4_pass": c4_pass,
        "seed_pass": seed_pass,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    dry_run = args.dry_run

    t0 = time.time()
    timestamp_utc = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{EXPERIMENT_TYPE}_{timestamp_utc}_v3"

    print(f"EXQ-517b: MECH-302 longer episodes", flush=True)
    print(
        f"window={COMP_WINDOW_LENGTH} threshold={COMP_DROP_THRESHOLD} "
        f"min_norm={COMP_MIN_INITIAL_NORM}",
        flush=True,
    )

    results_arm_a = []
    results_arm_b = []

    for seed in SEEDS:
        results_arm_a.append(run_arm(seed, "ARM_A_mech302_on", True, dry_run))
        results_arm_b.append(run_arm(seed, "ARM_B_mech302_off", False, dry_run))

    n_seeds = len(results_arm_a)
    a_passes = sum(1 for r in results_arm_a if r["seed_pass"])
    b_passes = sum(1 for r in results_arm_b if r["seed_pass"])

    a_pass_rate = a_passes / n_seeds if n_seeds > 0 else 0.0
    b_pass_rate = b_passes / n_seeds if n_seeds > 0 else 0.0

    overall_pass = (
        a_pass_rate >= PASS_FRACTION_REQUIRED
        and b_pass_rate >= PASS_FRACTION_REQUIRED
    )

    outcome = "PASS" if overall_pass else "FAIL"
    evidence_direction = "supports" if overall_pass else "non_contributory"

    mean_a_events = (
        sum(r["p1_events"] for r in results_arm_a) / n_seeds if n_seeds else 0.0
    )
    mean_a_writes = (
        sum(r["p1_writes"] for r in results_arm_a) / n_seeds if n_seeds else 0.0
    )
    mean_b_events = (
        sum(r["p1_events"] for r in results_arm_b) / n_seeds if n_seeds else 0.0
    )
    mean_b_writes = (
        sum(r["p1_writes"] for r in results_arm_b) / n_seeds if n_seeds else 0.0
    )

    elapsed = time.time() - t0

    manifest = {
        "schema_version": "v1",
        "run_id": run_id,
        "experiment_type": EXPERIMENT_TYPE,
        "architecture_epoch": ARCHITECTURE_EPOCH,
        "timestamp_utc": timestamp_utc,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "claim_ids": CLAIM_IDS,
        "supersedes": "v3_exq_517a_mech302_relief_completion_discriminative_pair",
        "outcome": outcome,
        "evidence_direction": evidence_direction,
        "evidence_direction_per_claim": {
            "MECH-302": evidence_direction,
        },
        "registered_thresholds": {
            "C1_MIN_EVENTS_PER_SEED": C1_MIN_EVENTS_PER_SEED,
            "C2_MIN_WRITES_PER_SEED": C2_MIN_WRITES_PER_SEED,
            "PASS_FRACTION_REQUIRED": PASS_FRACTION_REQUIRED,
        },
        "criteria": {
            "n_seeds": n_seeds,
            "a_seeds_pass": a_passes,
            "b_seeds_pass": b_passes,
            "a_pass_rate": a_pass_rate,
            "b_pass_rate": b_pass_rate,
            "a_pass_rate_meets_threshold": a_pass_rate >= PASS_FRACTION_REQUIRED,
            "b_pass_rate_meets_threshold": b_pass_rate >= PASS_FRACTION_REQUIRED,
            "overall_pass": overall_pass,
            "mean_arm_a_p1_events": mean_a_events,
            "mean_arm_a_p1_writes": mean_a_writes,
            "mean_arm_b_p1_events": mean_b_events,
            "mean_arm_b_p1_writes": mean_b_writes,
        },
        "config": {
            "p0_episodes": P0_EPISODES,
            "p1_episodes": P1_EPISODES,
            "steps_per_episode": STEPS_PER_EPISODE,
            "env_size": GRID_SIZE,
            "n_hazards": N_HAZARDS,
            "n_resources": N_RESOURCES,
            "body_obs_dim": BODY_OBS_DIM,
            "world_obs_dim": WORLD_OBS_DIM,
            "harm_obs_a_dim": HARM_OBS_A_DIM,
            "action_dim": ACTION_DIM,
            "comp_window_length": COMP_WINDOW_LENGTH,
            "comp_drop_threshold": COMP_DROP_THRESHOLD,
            "comp_min_initial_norm": COMP_MIN_INITIAL_NORM,
            "seeds": SEEDS,
            "dry_run": dry_run,
        },
        "results_arm_a_mech302_on": results_arm_a,
        "results_arm_b_mech302_off": results_arm_b,
        "elapsed_seconds": elapsed,
        "notes": (
            "MECH-302 relief-completion discriminative pair -- longer episodes. "
            "Supersedes EXQ-517a (mean 0.33 events/seed: episode length 150 steps is "
            "structurally too short relative to heal_rate=0.002/step healing dynamics "
            "requiring ~500 steps for full healing). Fix: steps_per_episode 150->300. "
            "heal_rate=0.002 unchanged (biological timescale claim). "
            "Comparator thresholds unchanged from 517a (window=30, threshold=0.005, min_norm=0.01). "
            "ARM_A (ON): use_suffering_derivative_comparator=True + valence_liking_enabled=True; "
            "ARM_B (OFF): both False. limb_damage_enabled=True (SD-022), heal_rate=0.002/step. "
            "C1: ARM_A events >= 3/seed. C2: ARM_A writes >= 2/seed. "
            "C3: ARM_B events == 0. C4: ARM_B writes == 0. "
            "PASS lifts gate (c) for MECH-304 conditioned-inhibition experiment."
        ),
    }

    out_dir = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "REE_assembly",
        "evidence",
        "experiments",
    )
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{run_id}.json")

    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"outcome: {outcome}", flush=True)
    print(f"Done: {out_path}", flush=True)
    print(
        f"mean arm_a events={mean_a_events:.1f} writes={mean_a_writes:.1f} | "
        f"arm_b events={mean_b_events:.1f} writes={mean_b_writes:.1f}",
        flush=True,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
