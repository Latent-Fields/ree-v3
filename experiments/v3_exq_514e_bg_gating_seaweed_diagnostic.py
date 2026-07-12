"""
V3-EXQ-514e: Schema predictability + BG gating.

ARM_A (baseline): random resource placement, no landmarks -- same as 514d.
ARM_B (seaweed):  n_landmarks_b=2, landmark_b_resource_bias=1.0 -- resources
                  spawn near smelly-seaweed landmarks; agent can smell them
                  from a distance via the 5x5 gradient field view.

Tests whether E1-learnable spatial schema enables BG gate commitment.
514d showed rv_final=0.5000 all seeds, gate never commits under random placement.
If ARM_B committed_frac_last_third > 0.05 AND rv_final < 0.40, schema
predictability is a missing ingredient and MECH-216 (E1 schema readout)
is the architectural lever.

experiment_purpose = "diagnostic"
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import torch
import torch.optim as optim

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.agent import REEAgent
from ree_core.utils.config import REEConfig
from experiments.pack_writer import write_flat_manifest  # noqa: E402
from pathlib import Path  # noqa: E402

EXPERIMENT_TYPE = "v3_exq_514e_bg_gating_seaweed_diagnostic"
CLAIM_IDS: List[str] = []
EXPERIMENT_PURPOSE = "diagnostic"
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"

SEEDS = [42, 43, 44]
N_EPISODES = 90
STEPS_PER_EPISODE = 300
DRY_RUN_EPISODES = 3

GRID_SIZE = 8
N_HAZARDS = 0
N_RESOURCES = 15
N_RESOURCE_TYPES = 3

COMMIT_THRESHOLD = 0.40
CLASSIFIER_LOSS_WEIGHT = 0.1

ARM_LABEL_A = "ARM_A_random"
ARM_LABEL_B = "ARM_B_seaweed"


def make_env(seed: int, seaweed: bool) -> CausalGridWorldV2:
    kwargs: Dict = dict(
        size=GRID_SIZE,
        num_hazards=N_HAZARDS,
        num_resources=N_RESOURCES,
        seed=seed,
        multi_resource_heterogeneity_enabled=True,
        n_resource_types=N_RESOURCE_TYPES,
        per_axis_drive_enabled=True,
    )
    if seaweed:
        kwargs["n_landmarks_b"] = 2
        kwargs["landmark_b_resource_bias"] = 1.0  # resources always near seaweed
    return CausalGridWorldV2(**kwargs)


def make_config(env: CausalGridWorldV2) -> REEConfig:
    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        alpha_world=0.9,
        world_dim=32,
        drive_weight=2.0,
        z_goal_enabled=True,
    )
    cfg.latent.use_resource_encoder = True
    cfg.latent.z_resource_dim = 32
    cfg.latent.use_identity_classifier = True
    cfg.latent.identity_classifier_n_types = N_RESOURCE_TYPES
    cfg.goal.goal_dim = cfg.latent.world_dim
    return cfg


def run_arm(seed: int, n_episodes: int, seaweed: bool,
            device: torch.device) -> Dict:
    env = make_env(seed, seaweed)
    cfg = make_config(env)
    agent = REEAgent(cfg)
    agent.train()

    opt = optim.Adam(agent.parameters(), lr=1e-3)

    ep_committed_frac: List[float] = []
    ep_rv_final: List[float] = []
    ep_resource_contacts: List[int] = []
    ep_rv_mean: List[float] = []

    for ep in range(n_episodes):
        _, obs_dict = env.reset()
        agent.reset()

        step_committed = 0
        step_rv_sum = 0.0
        resource_contacts = 0

        for _step in range(STEPS_PER_EPISODE):
            obs_body = obs_dict["body_state"].to(device)
            obs_world = obs_dict["world_state"].to(device)
            latent = agent.sense(obs_body, obs_world)

            ticks = agent.clock.advance()
            e1_prior = (
                agent._e1_tick(latent) if ticks.get("e1_tick")
                else torch.zeros(1, agent.config.latent.world_dim, device=device)
            )
            candidates = agent.generate_trajectories(latent, e1_prior, ticks)
            action = agent.select_action(candidates, ticks)
            action_idx = int(action.argmax(dim=-1).item())

            is_committed = agent.beta_gate.is_elevated
            rv = agent.e3._running_variance
            if is_committed:
                step_committed += 1
            step_rv_sum += rv

            _, harm_signal, done, info, obs_dict = env.step(action_idx)
            ttype = info.get("transition_type", "none")
            if ttype == "resource":
                resource_contacts += 1

            opt.zero_grad()
            loss = agent.compute_prediction_loss()
            if (
                agent.latent_stack.resource_encoder is not None
                and latent.resource_prox_pred_r is not None
            ):
                prox_val = float(info.get("resource_field_at_agent", 0.0))
                prox_target = torch.tensor([[prox_val]], dtype=torch.float32,
                                           device=device)
                loss = loss + agent.compute_resource_encoder_loss(prox_target, latent)
            if getattr(agent.config.latent, "use_identity_classifier", False):
                target_type = int(info.get("sd049_consumed_type_tag_this_tick", 0))
                loss = loss + CLASSIFIER_LOSS_WEIGHT * agent.compute_resource_identity_loss(
                    target_type, latent)

            if loss.requires_grad:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                opt.step()

            if ttype == "resource":
                drive_lvl = float(REEAgent.compute_drive_level(obs_body))
                agent.update_z_goal(float(harm_signal), drive_level=drive_lvl)

            if done:
                break

        ep_committed_frac.append(step_committed / STEPS_PER_EPISODE)
        ep_rv_final.append(agent.e3._running_variance)
        ep_rv_mean.append(step_rv_sum / STEPS_PER_EPISODE)
        ep_resource_contacts.append(resource_contacts)

    def _mean(lst: List[float]) -> float:
        return sum(lst) / max(1, len(lst))

    n = len(ep_committed_frac)
    third = max(1, n // 3)
    ever_committed = any(f > 0 for f in ep_committed_frac)
    first_commit_ep = next((i for i, f in enumerate(ep_committed_frac) if f > 0), -1)

    return {
        "seed": seed,
        "seaweed": seaweed,
        "world_obs_dim": env.world_obs_dim,
        "ever_committed": ever_committed,
        "first_commit_episode": first_commit_ep,
        "committed_frac_last_third": _mean(ep_committed_frac[n - third:]),
        "rv_final": ep_rv_final[-1] if ep_rv_final else -1.0,
        "rv_mean_last_third": _mean(ep_rv_mean[n - third:]),
        "resource_contacts_total": sum(ep_resource_contacts),
        "resource_contacts_mean": _mean([float(c) for c in ep_resource_contacts]),
        "ep_committed_frac": ep_committed_frac,
        "ep_rv_final": ep_rv_final,
        "ep_resource_contacts": ep_resource_contacts,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    device = torch.device("cpu")
    n_eps = DRY_RUN_EPISODES if args.dry_run else N_EPISODES

    def _mean(lst: List[float]) -> float:
        return sum(lst) / max(1, len(lst))

    all_results: Dict[str, List[Dict]] = {ARM_LABEL_A: [], ARM_LABEL_B: []}

    for seed in SEEDS:
        for arm_label, seaweed in [(ARM_LABEL_A, False), (ARM_LABEL_B, True)]:
            print(f"Seed {seed} {arm_label}", flush=True)
            t0 = time.time()
            result = run_arm(seed, n_eps, seaweed, device)
            elapsed = time.time() - t0
            print(
                f"  [train] seed={seed} arm={arm_label} ep 1/{n_eps}"
                f" ever_committed={result['ever_committed']}"
                f" rv_final={result['rv_final']:.4f}"
                f" contacts={result['resource_contacts_total']}"
                f" world_obs_dim={result['world_obs_dim']}"
                f" elapsed={elapsed:.1f}s",
                flush=True,
            )
            all_results[arm_label].append(result)

    arm_a = all_results[ARM_LABEL_A]
    arm_b = all_results[ARM_LABEL_B]

    mean_committed_a = _mean([r["committed_frac_last_third"] for r in arm_a])
    mean_committed_b = _mean([r["committed_frac_last_third"] for r in arm_b])
    mean_rv_a = _mean([r["rv_final"] for r in arm_a])
    mean_rv_b = _mean([r["rv_final"] for r in arm_b])
    mean_contacts_a = _mean([r["resource_contacts_mean"] for r in arm_a])
    mean_contacts_b = _mean([r["resource_contacts_mean"] for r in arm_b])

    # PASS: seaweed arm shows gating; baseline does not
    schema_enables_gating = (
        mean_committed_b > 0.05
        and mean_rv_b < COMMIT_THRESHOLD
        and mean_committed_b > mean_committed_a + 0.02
    )
    outcome = "PASS" if schema_enables_gating else "FAIL"

    print(f"outcome: {outcome}", flush=True)
    print(
        f"mean arm_a committed_last={mean_committed_a:.3f} rv_final={mean_rv_a:.4f}"
        f" contacts={mean_contacts_a:.2f}",
        flush=True,
    )
    print(
        f"mean arm_b committed_last={mean_committed_b:.3f} rv_final={mean_rv_b:.4f}"
        f" contacts={mean_contacts_b:.2f}",
        flush=True,
    )

    if args.dry_run:
        return

    manifest = {
        "run_id": f"{EXPERIMENT_TYPE}_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}_v3",
        "experiment_type": EXPERIMENT_TYPE,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": ARCHITECTURE_EPOCH,
        "timestamp_utc": datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"),
        "outcome": outcome,
        "evidence_direction": "non_contributory",
        "schema_enables_gating": schema_enables_gating,
        "mean_committed_frac_last_a": mean_committed_a,
        "mean_committed_frac_last_b": mean_committed_b,
        "mean_rv_final_a": mean_rv_a,
        "mean_rv_final_b": mean_rv_b,
        "mean_contacts_per_ep_a": mean_contacts_a,
        "mean_contacts_per_ep_b": mean_contacts_b,
        "commit_threshold": COMMIT_THRESHOLD,
        "per_seed_results": {
            ARM_LABEL_A: [
                {k: v for k, v in r.items()
                 if k not in ("ep_committed_frac", "ep_rv_final",
                              "ep_resource_contacts")}
                for r in arm_a
            ],
            ARM_LABEL_B: [
                {k: v for k, v in r.items()
                 if k not in ("ep_committed_frac", "ep_rv_final",
                              "ep_resource_contacts")}
                for r in arm_b
            ],
        },
        "per_seed_episode_traces": {
            ARM_LABEL_A: [
                {
                    "seed": r["seed"],
                    "ep_committed_frac": r["ep_committed_frac"],
                    "ep_rv_final": r["ep_rv_final"],
                    "ep_resource_contacts": r["ep_resource_contacts"],
                }
                for r in arm_a
            ],
            ARM_LABEL_B: [
                {
                    "seed": r["seed"],
                    "ep_committed_frac": r["ep_committed_frac"],
                    "ep_rv_final": r["ep_rv_final"],
                    "ep_resource_contacts": r["ep_resource_contacts"],
                }
                for r in arm_b
            ],
        },
        "config": {
            "seeds": SEEDS,
            "n_episodes": n_eps,
            "steps_per_episode": STEPS_PER_EPISODE,
            "grid_size": GRID_SIZE,
            "n_hazards": N_HAZARDS,
            "n_resources": N_RESOURCES,
            "n_resource_types": N_RESOURCE_TYPES,
            "commit_threshold": COMMIT_THRESHOLD,
            "arm_b_n_landmarks_b": 2,
            "arm_b_landmark_b_resource_bias": 1.0,
        },
    }

    out_dir = args.output_dir or os.path.join(
        os.path.dirname(__file__), "..", "..", "REE_assembly",
        "evidence", "experiments", EXPERIMENT_TYPE,
    )
    os.makedirs(out_dir, exist_ok=True)
    out_path = write_flat_manifest(
        manifest,
        out_dir,
        dry_run=False,
        config=manifest.get("config"),
        seeds=SEEDS,
        script_path=Path(__file__),
    )
    print(f"Done: {out_path}", flush=True)
    print(
        f"mean arm_a committed={mean_committed_a:.3f} rv={mean_rv_a:.4f}"
        f" | arm_b committed={mean_committed_b:.3f} rv={mean_rv_b:.4f}",
        flush=True,
    )

    from experiment_protocol import emit_outcome
    emit_outcome(outcome=outcome, manifest_path=str(out_path))


if __name__ == "__main__":
    main()
