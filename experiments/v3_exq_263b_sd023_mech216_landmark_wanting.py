"""
V3-EXQ-263b: SD-023 Validation -- Landmark Gradient Texture for MECH-216 Anticipatory Wanting

Supersedes: V3-EXQ-263a

Root cause of EXQ-263a (and EXQ-263) failure or artifactual salience:
  Even after API fixes (263a), MECH-216 will likely produce artifactual salience because E1
  has no leading-indicator signal to predict. The only world feature correlated with resource
  proximity is the resource proximity field itself. E1 learns "resource_prox is high" not
  "pattern X predicts upcoming resource contact." There is no temporal predictive structure.

This experiment tests whether:
  (1) SD-023 landmarks provide a leading indicator that E1's LSTM can learn
  (2) schema_salience rises near Landmark B BEFORE resource proximity rises
  (3) MECH-216 produces genuinely anticipatory (not reactive) wanting

MECHANISM UNDER TEST: SD-023 (environment.gradient_texture) + MECH-216 (e1_predictive_wanting)

EXPERIMENT DESIGN:
  Two conditions:
    LANDMARK_ENABLED: n_landmarks_a=2, n_landmarks_b=2, world_obs_dim=300
    LANDMARK_ABLATED: n_landmarks_a=0, n_landmarks_b=0, world_obs_dim=250
  Both conditions: schema_wanting_enabled=True, use_resource_proximity_head=True
  3 seeds per condition (6 total runs)
  Episodes: 200 per seed (enough to learn landmark-resource correlation)
  Steps/ep: 300

Phased training:
  P0 (50 episodes): encoder + resource_prox warmup
  P1 (150 episodes): schema_readout training on frozen encoder (E1 trains on z_world)

Key metrics:
  schema_salience_mean: average schema salience over P1
  salience_at_landmark_b: mean salience when agent is within 2 cells of any landmark_B
  salience_at_approach: mean salience in leading-indicator zone (within 3 cells but
                        outside 1 cell of resource)
  resource_prox_correlation: correlation(schema_salience, max_resource_field_view)
  landmark_prox_correlation: correlation(schema_salience, max_landmark_b_field_view)
    (should be LOWER in LANDMARK_ENABLED -- salience fires on landmark not resource)

ACCEPTANCE CRITERIA:
  C1: LANDMARK_ENABLED shows salience_at_landmark_b > LANDMARK_ABLATED salience_at_landmark_b
      (landmark B proximity elevates schema salience)
  C2: LANDMARK_ENABLED shows landmark_prox_correlation > resource_prox_correlation
      (schema salience tracks landmark_B better than direct resource proximity)

EXPERIMENT_PURPOSE: diagnostic  (substrate readiness validation, not primary evidence)
"""

import os
import sys
import json
import time
import numpy as np
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig

EXPERIMENT_PURPOSE = "diagnostic"
EXPERIMENT_TYPE = "v3_exq_263b_sd023_mech216_landmark_wanting"
CLAIM_IDS = ["SD-023", "MECH-216"]

GRID_SIZE = 10
ACTION_DIM = 4
N_HAZARDS = 2
N_RESOURCES = 3

NUM_SEEDS = 3
P0_EPISODES = 50     # encoder + resource_prox warmup
P1_EPISODES = 150    # schema_readout training (frozen encoder after P0)
STEPS_PER_EPISODE = 300
TOTAL_EPISODES = P0_EPISODES + P1_EPISODES

LANDMARK_RADIUS = 2   # proximity radius for salience_at_landmark_b metric
APPROACH_OUTER = 3    # outer boundary of leading-indicator zone
APPROACH_INNER = 1    # inner boundary (inside = at resource)

CONDITIONS = {
    "LANDMARK_ENABLED": {
        "n_landmarks_a": 2,
        "n_landmarks_b": 2,
        "landmark_b_resource_bias": 0.7,
    },
    "LANDMARK_ABLATED": {
        "n_landmarks_a": 0,
        "n_landmarks_b": 0,
        "landmark_b_resource_bias": 0.7,  # no effect when n_landmarks_b=0
    },
}


def _dist(ax, ay, pos):
    """Euclidean distance from agent to (x, y) position."""
    return ((ax - pos[0]) ** 2 + (ay - pos[1]) ** 2) ** 0.5


def run_condition(condition_name, env_cfg, seed):
    """Run one condition x seed. Returns per-seed metrics."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = CausalGridWorldV2(
        size=GRID_SIZE,
        num_hazards=N_HAZARDS,
        num_resources=N_RESOURCES,
        use_proxy_fields=True,
        resource_respawn_on_consume=True,
        seed=seed,
        **env_cfg,
    )

    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        alpha_world=0.9,
        use_event_classifier=True,
        use_resource_proximity_head=True,
        resource_proximity_weight=0.5,
        use_harm_stream=True,
        use_affective_harm_stream=True,
        z_goal_enabled=True,
        drive_weight=2.0,
        tonic_5ht_enabled=True,
        wanting_weight=0.4,
        schema_wanting_enabled=True,
        schema_wanting_threshold=0.3,
        schema_wanting_gain=0.5,
    )

    agent = REEAgent(cfg)
    device = agent.device

    e1_optimizer = torch.optim.Adam(agent.e1.parameters(), lr=1e-4)
    encoder_optimizer = torch.optim.Adam(agent.latent_stack.parameters(), lr=1e-4)

    metrics = {
        "p1_schema_salience": [],
        "p1_salience_at_landmark_b": [],
        "p1_salience_at_approach": [],
        "p1_resource_prox_values": [],
        "p1_landmark_prox_values": [],
    }

    has_landmarks = env_cfg["n_landmarks_b"] > 0

    for ep in range(TOTAL_EPISODES):
        _, obs_dict = env.reset()
        agent.reset()
        agent.e1.reset_hidden_state()

        # Freeze encoder after P0
        encoder_frozen = (ep >= P0_EPISODES)
        if encoder_frozen:
            for param in agent.latent_stack.parameters():
                param.requires_grad_(False)
        else:
            for param in agent.latent_stack.parameters():
                param.requires_grad_(True)

        for step in range(STEPS_PER_EPISODE):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            obs_harm  = obs_dict.get("harm_obs", None)

            latent = agent.sense(obs_body, obs_world, obs_harm=obs_harm)
            ticks  = agent.clock.advance()

            e1_prior = agent._e1_tick(latent) if ticks.get("e1_tick", False) else torch.zeros(
                1, cfg.latent.world_dim, device=device
            )

            candidates = agent.generate_trajectories(latent, e1_prior, ticks)
            action = agent.select_action(candidates, ticks)

            _, harm_signal, done, info, obs_dict = env.step(action)

            benefit = float(info.get("benefit_exposure", 0.0))

            drive_level = agent.compute_drive_level(obs_body)
            agent.update_z_goal(benefit, drive_level=drive_level)
            agent.serotonin_step(benefit_exposure=benefit)
            if benefit > 0:
                agent.update_benefit_salience(benefit)

            agent.update_schema_wanting(drive_level=float(drive_level))

            # Training (skip during frozen P1 encoder phase, but still train E1)
            if ep < P0_EPISODES:
                # P0: train encoder + resource proximity head + E1 schema readout
                e1_optimizer.zero_grad()
                encoder_optimizer.zero_grad()

                pred_loss = agent.compute_prediction_loss()
                rfv = obs_dict.get("resource_field_view", None)
                if rfv is not None:
                    resource_prox = float(rfv.max())
                    schema_loss = agent.compute_schema_readout_loss(resource_prox)
                    rp_loss = agent.compute_resource_proximity_loss(resource_prox, latent)
                else:
                    resource_prox = 0.0
                    schema_loss = agent.compute_schema_readout_loss(0.0)
                    rp_loss = torch.tensor(0.0)

                total_loss = pred_loss + schema_loss + 0.5 * rp_loss
                if total_loss.requires_grad:
                    total_loss.backward()
                    e1_optimizer.step()
                    encoder_optimizer.step()

            elif ep >= P0_EPISODES:
                # P1: train E1 schema readout only (encoder frozen)
                e1_optimizer.zero_grad()
                rfv = obs_dict.get("resource_field_view", None)
                if rfv is not None:
                    resource_prox = float(rfv.max())
                    schema_loss = agent.compute_schema_readout_loss(resource_prox)
                    if schema_loss.requires_grad:
                        schema_loss.backward()
                        e1_optimizer.step()

            # P1 metrics collection
            if ep >= P0_EPISODES and agent._schema_salience is not None:
                sal = float(agent._schema_salience.squeeze())
                ax_pos = int(env.agent_x)
                ay_pos = int(env.agent_y)

                # Resource field view max
                rfv = obs_dict.get("resource_field_view", None)
                r_prox = float(rfv.max()) if rfv is not None else 0.0

                # Landmark B field view max
                lbfv = obs_dict.get("landmark_b_field_view", None)
                lb_prox = float(lbfv.max()) if lbfv is not None else 0.0

                metrics["p1_schema_salience"].append(sal)
                metrics["p1_resource_prox_values"].append(r_prox)
                metrics["p1_landmark_prox_values"].append(lb_prox)

                # salience_at_landmark_b: within LANDMARK_RADIUS cells of any landmark_B
                if has_landmarks and env.landmark_b_positions:
                    near_lb = any(
                        _dist(ax_pos, ay_pos, lbp) <= LANDMARK_RADIUS
                        for lbp in env.landmark_b_positions
                    )
                    if near_lb:
                        metrics["p1_salience_at_landmark_b"].append(sal)

                # salience_at_approach: within APPROACH_OUTER but outside APPROACH_INNER of resource
                near_resource = any(
                    _dist(ax_pos, ay_pos, (r[0], r[1])) <= APPROACH_INNER
                    for r in env.resources
                )
                in_approach_zone = any(
                    APPROACH_INNER < _dist(ax_pos, ay_pos, (r[0], r[1])) <= APPROACH_OUTER
                    for r in env.resources
                )
                if in_approach_zone and not near_resource:
                    metrics["p1_salience_at_approach"].append(sal)

            if done:
                break

    return metrics


def compute_correlation(xs, ys):
    """Pearson correlation between two lists. Returns 0 if insufficient data."""
    if len(xs) < 2 or len(ys) < 2:
        return 0.0
    xa = np.array(xs, dtype=np.float32)
    ya = np.array(ys, dtype=np.float32)
    n = min(len(xa), len(ya))
    xa, ya = xa[:n], ya[:n]
    if xa.std() < 1e-8 or ya.std() < 1e-8:
        return 0.0
    return float(np.corrcoef(xa, ya)[0, 1])


def main():
    results = {}
    for cond_name, env_cfg in CONDITIONS.items():
        results[cond_name] = {}
        for seed_idx in range(NUM_SEEDS):
            seed = 42 + seed_idx
            print(f"Running {cond_name} seed={seed_idx} (seed={seed})...")
            t0 = time.time()
            m = run_condition(cond_name, env_cfg, seed=seed)
            elapsed = time.time() - t0
            print(f"  Done in {elapsed:.1f}s -- salience_mean={float(np.mean(m['p1_schema_salience'])) if m['p1_schema_salience'] else 0.0:.4f}")
            results[cond_name][f"seed_{seed_idx}"] = m

    # Aggregate per condition
    agg = {}
    for cond_name in CONDITIONS:
        cond_results = results[cond_name]
        all_sal = []
        all_sal_lb = []
        all_sal_ap = []
        all_r_prox = []
        all_lb_prox = []

        for m in cond_results.values():
            all_sal.extend(m["p1_schema_salience"])
            all_sal_lb.extend(m["p1_salience_at_landmark_b"])
            all_sal_ap.extend(m["p1_salience_at_approach"])
            all_r_prox.extend(m["p1_resource_prox_values"])
            all_lb_prox.extend(m["p1_landmark_prox_values"])

        resource_prox_corr = compute_correlation(all_sal, all_r_prox)
        landmark_prox_corr = compute_correlation(all_sal, all_lb_prox)

        agg[cond_name] = {
            "schema_salience_mean": float(np.mean(all_sal)) if all_sal else 0.0,
            "schema_salience_std": float(np.std(all_sal)) if all_sal else 0.0,
            "salience_at_landmark_b": float(np.mean(all_sal_lb)) if all_sal_lb else 0.0,
            "salience_at_landmark_b_n": len(all_sal_lb),
            "salience_at_approach": float(np.mean(all_sal_ap)) if all_sal_ap else 0.0,
            "salience_at_approach_n": len(all_sal_ap),
            "resource_prox_correlation": resource_prox_corr,
            "landmark_prox_correlation": landmark_prox_corr,
            "n_p1_steps": len(all_sal),
        }

    # Acceptance checks
    le_agg = agg["LANDMARK_ENABLED"]
    la_agg = agg["LANDMARK_ABLATED"]

    c1_pass = le_agg["salience_at_landmark_b"] > la_agg["salience_at_landmark_b"]
    c2_pass = le_agg["landmark_prox_correlation"] > le_agg["resource_prox_correlation"]

    outcome = "PASS" if (c1_pass and c2_pass) else "FAIL"

    output = {
        "experiment_type": EXPERIMENT_TYPE,
        "run_id": f"{EXPERIMENT_TYPE}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_v3",
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "supersedes": "V3-EXQ-263a",
        "outcome": outcome,
        "conditions": list(CONDITIONS.keys()),
        "aggregated": agg,
        "acceptance_checks": {
            "C1_salience_at_landmark_b_higher": c1_pass,
            "C2_landmark_prox_corr_gt_resource_prox_corr": c2_pass,
        },
        "params": {
            "grid_size": GRID_SIZE,
            "n_hazards": N_HAZARDS,
            "n_resources": N_RESOURCES,
            "p0_episodes": P0_EPISODES,
            "p1_episodes": P1_EPISODES,
            "steps_per_episode": STEPS_PER_EPISODE,
            "num_seeds": NUM_SEEDS,
            "landmark_radius": LANDMARK_RADIUS,
            "approach_outer": APPROACH_OUTER,
            "approach_inner": APPROACH_INNER,
        },
        "evidence_direction": "supports" if outcome == "PASS" else "does_not_support",
        "evidence_direction_per_claim": {
            "SD-023": "supports" if c1_pass else "does_not_support",
            "MECH-216": "supports" if (c1_pass and c2_pass) else "does_not_support",
        },
    }

    # Write output
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "..", "REE_assembly", "evidence", "experiments"
    )
    out_path = os.path.join(out_dir, f"{EXPERIMENT_TYPE}_{ts}_v3.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults written to {out_path}")
    print(f"Outcome: {outcome}")
    print(f"C1 (salience_at_lb higher in ENABLED): {c1_pass}")
    print(f"  ENABLED  salience_at_lb = {le_agg['salience_at_landmark_b']:.4f} (n={le_agg['salience_at_landmark_b_n']})")
    print(f"  ABLATED  salience_at_lb = {la_agg['salience_at_landmark_b']:.4f} (n={la_agg['salience_at_landmark_b_n']})")
    print(f"C2 (landmark_prox_corr > resource_prox_corr in ENABLED): {c2_pass}")
    print(f"  ENABLED  landmark_prox_corr={le_agg['landmark_prox_correlation']:.4f}  resource_prox_corr={le_agg['resource_prox_correlation']:.4f}")

    return output


if __name__ == "__main__":
    main()
