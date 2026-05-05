"""
V3-EXQ-514d: SD-049 BG gating diagnostic.

Diagnostic motivated by 514b C2b FAIL: user concern that the beta gate
(MECH-090) may never commit with the default 514b config, so the agent
does a fixed initial behavior then idles. Longer episodes would then give
more idle time, not more multi-step chaining.

This experiment uses the exact 514b config (no bistable, no dACC, no
urgency_weight, no explicit commit_threshold override) and tracks per-step
and per-episode BG state to determine:
  (a) Does running_variance ever drop below the commit_threshold (0.40)?
  (b) If so, when -- early or late in training?
  (c) What fraction of steps are committed?
  (d) Does committed state correlate with resource contacts?

experiment_purpose = "diagnostic" -- not scored against evidence criteria.
Results inform whether 514c (reef approach) or BG-config changes are needed.
"""

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime
from typing import Dict, List

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.agent import REEAgent
from ree_core.utils.config import REEConfig

EXPERIMENT_TYPE = "v3_exq_514d_sd049_bg_gating_diagnostic"
CLAIM_IDS: List[str] = []
EXPERIMENT_PURPOSE = "diagnostic"
ARCHITECTURE_EPOCH = "ree_hybrid_guardrails_v1"

SEEDS = [42, 43, 44]
N_EPISODES = 90          # same as 514b P0
STEPS_PER_EPISODE = 300  # same as 514b

GRID_SIZE = 8
N_HAZARDS = 0
N_RESOURCES = 15

COMMIT_THRESHOLD = 0.40   # default from E3Config.commitment_threshold

DRY_RUN_EPISODES = 3


def make_config() -> REEConfig:
    """Exact same config as 514b (no bistable, no dACC)."""
    cfg = REEConfig.from_dims(
        body_obs_dim=20,  # 514b uses default body_obs_dim; will be overridden below
        world_obs_dim=325,  # SD-049 3-type: 250 + 3*25 = 325
        action_dim=5,
        alpha_world=0.9,
        world_dim=32,
        drive_weight=2.0,
        z_goal_enabled=True,
    )
    cfg.latent.use_resource_encoder = True
    cfg.latent.z_resource_dim = 32
    cfg.latent.use_identity_classifier = True
    cfg.latent.identity_classifier_n_types = 3
    cfg.goal.goal_dim = cfg.latent.world_dim
    return cfg


def make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        size=GRID_SIZE,
        num_hazards=N_HAZARDS,
        num_resources=N_RESOURCES,
        seed=seed,
        multi_resource_heterogeneity_enabled=True,
        n_resource_types=3,
        per_axis_drive_enabled=True,
    )


def _entropy(counts: Dict[int, int]) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    ent = 0.0
    for c in counts.values():
        if c > 0:
            p = c / total
            ent -= p * math.log(p + 1e-12)
    return ent


def run_diagnostic(seed: int, n_episodes: int, device: torch.device) -> Dict:
    env = make_env(seed)
    cfg = make_config()

    # Re-build with correct dims from env
    cfg2 = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        alpha_world=0.9,
        world_dim=32,
        drive_weight=2.0,
        z_goal_enabled=True,
    )
    cfg2.latent.use_resource_encoder = True
    cfg2.latent.z_resource_dim = 32
    cfg2.latent.use_identity_classifier = True
    cfg2.latent.identity_classifier_n_types = 3
    cfg2.goal.goal_dim = cfg2.latent.world_dim

    agent = REEAgent(cfg2)
    agent.train()

    import torch.optim as optim
    classifier_loss_weight = 0.1
    opt = optim.Adam(agent.parameters(), lr=1e-3)

    # Per-episode tracking
    ep_committed_frac: List[float] = []
    ep_rv_mean: List[float] = []
    ep_rv_final: List[float] = []
    ep_action_entropy: List[float] = []
    ep_resource_contacts: List[int] = []

    for ep in range(n_episodes):
        _, obs_dict = env.reset()
        agent.reset()

        step_committed = 0
        step_rv_sum = 0.0
        step_rv_n = 0
        action_counts: Dict[int, int] = {}
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

            # BG state read
            is_committed = agent.beta_gate.is_elevated
            rv = agent.e3._running_variance
            if is_committed:
                step_committed += 1
            step_rv_sum += rv
            step_rv_n += 1
            action_counts[action_idx] = action_counts.get(action_idx, 0) + 1

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
                prox_target_val = float(info.get("resource_field_at_agent", 0.0))
                prox_target = torch.tensor([[prox_target_val]], dtype=torch.float32, device=device)
                res_loss = agent.compute_resource_encoder_loss(prox_target, latent)
                loss = loss + res_loss
            if getattr(agent.config.latent, "use_identity_classifier", False):
                target_type = int(info.get("sd049_consumed_type_tag_this_tick", 0))
                id_loss = agent.compute_resource_identity_loss(target_type, latent)
                loss = loss + classifier_loss_weight * id_loss

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
        ep_rv_mean.append(step_rv_sum / max(1, step_rv_n))
        ep_rv_final.append(agent.e3._running_variance)
        ep_action_entropy.append(_entropy(action_counts))
        ep_resource_contacts.append(resource_contacts)

    def _mean(lst: List[float]) -> float:
        return sum(lst) / max(1, len(lst))

    # Summarise over first / middle / last third of training
    n = len(ep_committed_frac)
    third = max(1, n // 3)
    first_frac = ep_committed_frac[:third]
    last_frac = ep_committed_frac[n - third:]
    first_rv = ep_rv_mean[:third]
    last_rv = ep_rv_mean[n - third:]

    ever_committed = any(f > 0 for f in ep_committed_frac)
    first_commit_ep = next((i for i, f in enumerate(ep_committed_frac) if f > 0), -1)

    return {
        "seed": seed,
        "commit_threshold": COMMIT_THRESHOLD,
        "ever_committed": ever_committed,
        "first_commit_episode": first_commit_ep,
        "committed_frac_first_third": _mean(first_frac),
        "committed_frac_last_third": _mean(last_frac),
        "rv_mean_first_third": _mean(first_rv),
        "rv_mean_last_third": _mean(last_rv),
        "rv_final": ep_rv_final[-1] if ep_rv_final else -1.0,
        "action_entropy_mean": _mean(ep_action_entropy),
        "action_entropy_final": ep_action_entropy[-1] if ep_action_entropy else -1.0,
        "resource_contacts_mean": _mean([float(c) for c in ep_resource_contacts]),
        "resource_contacts_total": sum(ep_resource_contacts),
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

    per_seed_results = []
    for seed in SEEDS:
        print(f"[514d] seed={seed} episodes={n_eps}")
        t0 = time.time()
        result = run_diagnostic(seed, n_eps, device)
        elapsed = time.time() - t0
        print(
            f"  ever_committed={result['ever_committed']}  "
            f"first_commit_ep={result['first_commit_episode']}  "
            f"rv_final={result['rv_final']:.4f}  "
            f"committed_frac_last_third={result['committed_frac_last_third']:.3f}  "
            f"contacts={result['resource_contacts_total']}  "
            f"elapsed={elapsed:.1f}s"
        )
        per_seed_results.append(result)

    # Aggregate
    n_ever_committed = sum(1 for r in per_seed_results if r["ever_committed"])
    mean_rv_final = sum(r["rv_final"] for r in per_seed_results) / len(per_seed_results)
    mean_committed_frac_last = sum(
        r["committed_frac_last_third"] for r in per_seed_results
    ) / len(per_seed_results)

    # Outcome assessment
    bg_functional = n_ever_committed >= 2 and mean_committed_frac_last >= 0.05

    outcome = "PASS" if bg_functional else "FAIL"

    manifest = {
        "run_id": (
            f"{EXPERIMENT_TYPE}_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}_v3"
        ),
        "experiment_type": EXPERIMENT_TYPE,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "architecture_epoch": ARCHITECTURE_EPOCH,
        "timestamp_utc": datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"),
        "outcome": outcome,
        "evidence_direction": "non_contributory",
        "n_seeds_ever_committed": n_ever_committed,
        "mean_rv_final": mean_rv_final,
        "mean_committed_frac_last_third": mean_committed_frac_last,
        "commit_threshold": COMMIT_THRESHOLD,
        "bg_functional_verdict": bg_functional,
        "per_seed_results": [
            {
                k: v for k, v in r.items()
                if k not in ("ep_committed_frac", "ep_rv_final", "ep_resource_contacts")
            }
            for r in per_seed_results
        ],
        "per_seed_episode_traces": [
            {
                "seed": r["seed"],
                "ep_committed_frac": r["ep_committed_frac"],
                "ep_rv_final": r["ep_rv_final"],
                "ep_resource_contacts": r["ep_resource_contacts"],
            }
            for r in per_seed_results
        ],
        "config": {
            "seeds": SEEDS,
            "n_episodes": n_eps,
            "steps_per_episode": STEPS_PER_EPISODE,
            "grid_size": GRID_SIZE,
            "n_hazards": N_HAZARDS,
            "n_resources": N_RESOURCES,
            "commit_threshold": COMMIT_THRESHOLD,
            "beta_gate_bistable": False,
            "use_dacc": False,
            "urgency_weight": 0.0,
        },
    }

    if args.dry_run:
        print(f"[dry-run] outcome={outcome}  bg_functional={bg_functional}")
        print(f"[dry-run] mean_rv_final={mean_rv_final:.4f}  "
              f"commit_threshold={COMMIT_THRESHOLD}")
        return

    # Write output
    out_dir = args.output_dir or os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "REE_assembly",
        "evidence",
        "experiments",
        EXPERIMENT_TYPE,
    )
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{manifest['run_id']}.json")
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[514d] written -> {out_path}")
    print(f"[514d] outcome={outcome}  bg_functional={bg_functional}  "
          f"mean_rv_final={mean_rv_final:.4f}  "
          f"mean_committed_frac_last={mean_committed_frac_last:.3f}")


if __name__ == "__main__":
    main()
