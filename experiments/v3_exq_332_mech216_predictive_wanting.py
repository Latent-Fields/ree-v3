"""
V3-EXQ-332: MECH-216 Genuinely Predictive Wanting -- Future-Target Supervision

Supersedes: V3-EXQ-263b

=== Design note: why EXQ-263b was insufficient ===

EXQ-263b trained schema_readout_head on CURRENT resource_proximity_target
(max(resource_field_view) at the same timestep). Four structural problems:

  (1) Reactive supervision: the head learns to detect nearby resources NOW, not
      predict upcoming contact. A high proximity score at step t is already
      directly available in world_state as the resource gradient -- E1 just
      needs to route that field through its LSTM to reconstruct it. This is
      reactive proximity detection, not anticipatory wanting.

  (2) Shortcut available: the resource gradient field in world_state is 25 dims
      of direct proximity information. With resource_obs_scale=1.0, the agent
      (and z_world encoder) can always see the resource gradient and achieve low
      proximity loss without needing landmark B at all.

  (3) Acceptance criteria were too weak: C1 (salience higher near landmark B in
      ENABLED) and C2 (landmark_prox_corr > resource_prox_corr) can be met if
      landmark B happens to be more spatially coherent than the resource gradient,
      regardless of whether the signal is genuinely predictive or just reflects a
      smoother correlated field. Neither criterion confirms that schema_salience
      predicts something the agent could not already see directly.

  (4) No test of anticipation: the experiment never asked whether schema_salience
      at time t predicts reward OPPORTUNITY at time t+N before that opportunity
      is directly visible. That is the defining test of anticipatory wanting
      (Berridge 2012; Zhang et al. 2009: W_m = kappa x V_hat requires V_hat to
      encode FUTURE expected value, not current proximity).

=== This redesign fixes all four problems ===

  (1) Supervision changes to FUTURE BINARY TARGET: contact_within_next_N_steps.
      Schema readout must learn to fire at timesteps where upcoming reward
      contact is likely -- not timesteps where the resource is currently close.

  (2) resource_obs_scale=0.3 strongly attenuates the direct resource gradient in
      world_state (observation only; benefit calculation unchanged). The agent
      cannot see resources clearly until very close; landmark B becomes the
      dominant useful cue for predicting upcoming contact.

  (3) Acceptance criteria directly separate predictive from reactive:
      - PRIMARY: corr(schema_salience_t, future_target_t) > 0.10 (predictive signal)
      - VERSUS:  corr(schema_salience_t, current_resource_prox_t) (reactive baseline)
      - PASS requires predictive > reactive AND predictive stronger in CUE_ON.

  (4) Episode-level replay training on cached E1 LSTM hidden states with deferred
      future_target labels: the readout head gradient cannot flow through the LSTM
      (h_top detached); only the head's own Linear(hidden_dim,1)+Sigmoid weights
      receive gradient. This forces the head to exploit the LSTM's own implicit
      temporal predictions, not reconstruct the proximity gradient.

MECHANISM UNDER TEST:
  MECH-216 (e1_predictive_wanting) + SD-023 (environment.gradient_texture)

EXPERIMENT DESIGN:
  Two conditions -- both with resource_obs_scale=0.3:
    PREDICTIVE_CUE_ON:      n_landmarks_b=2, landmark_b_resource_bias=0.8
                            Landmark B fields are unattenuated; high spatial bias
                            toward resources; serves as leading indicator.
    PREDICTIVE_CUE_ABLATED: n_landmarks_a=0, n_landmarks_b=0
                            No landmarks; direct resource signal attenuated;
                            no useful leading indicator in world_state.

  3 seeds per condition (6 total runs)
  grid_size=12, n_hazards=2, n_resources=2, resource_respawn=True
  FUTURE_WINDOW=10 steps (binary: any resource contact in t+1..t+10)

Training phases:
  P0 (60 episodes): encoder warmup + resource_proximity_head
    - Trains LatentStack + resource_proximity_head on attenuated resource field
    - Also trains E1 LSTM on prediction loss (z_self+z_world continuity)
    - Schema readout NOT trained in P0 (head exists but receives no gradient)
  P1 (180 episodes): schema_readout_head trained on future_target
    - Encoder frozen (LatentStack.parameters() no_grad)
    - At EACH STEP: cache h_top = agent.e1._hidden_state[0][-1].detach()
                    cache resource_contact flag (transition_type == "resource")
    - At EPISODE END: compute future_target_t for each step (retrospective),
                      run schema_readout_head(h_top) for each step,
                      backprop MSE(salience, future_target) through head only
    - Collect metrics: salience, future_target, current_resource_prox, landmark_prox

Primary metrics:
  future_target_corr:    corr(schema_salience_t, future_target_t) -- PRIMARY
  resource_prox_corr:    corr(schema_salience_t, current_resource_prox_t)
  landmark_prox_corr:    corr(schema_salience_t, landmark_b_prox_t) [CUE_ON only]
  salience_in_cue_zone:  mean salience when near LB (<=2 cells) but far from resource (>2 cells)

ACCEPTANCE CRITERIA (PASS requires all 3):
  C1: PREDICTIVE_CUE_ON future_target_corr > 0.10 (signal above noise threshold)
  C2: CUE_ON future_target_corr > ABLATED future_target_corr (cue enables prediction)
  C3: CUE_ON future_target_corr > CUE_ON resource_prox_corr (predictive not reactive)
"""

import os
import sys
import json
import time
import numpy as np
from collections import deque
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig

EXPERIMENT_PURPOSE = "evidence"
EXPERIMENT_TYPE = "v3_exq_332_mech216_predictive_wanting"
CLAIM_IDS = ["MECH-216", "SD-023"]

GRID_SIZE = 12
ACTION_DIM = 4
N_HAZARDS = 1   # low hazard regime: 1 hazard keeps world interesting without causing
                # rapid ABLATED health depletion (random walk dies at ~7 steps with 2 hazards)
N_RESOURCES = 2

NUM_SEEDS = 3
P0_EPISODES = 60     # encoder + resource_prox head warmup
P1_EPISODES = 180    # schema_readout future-target training + metric collection
STEPS_PER_EPISODE = 300
TOTAL_EPISODES = P0_EPISODES + P1_EPISODES

FUTURE_WINDOW = 10      # binary: any resource contact in next N steps
RESOURCE_OBS_SCALE = 0.3  # attenuate direct resource gradient in observation

LANDMARK_NEAR_RADIUS = 2   # cells -- "near landmark B" zone
RESOURCE_FAR_RADIUS = 2    # cells -- "far from resource" threshold for cue-zone metric

CONDITIONS = {
    "PREDICTIVE_CUE_ON": {
        "n_landmarks_a": 1,
        "n_landmarks_b": 2,
        "landmark_b_resource_bias": 0.8,
        "landmark_b_sigma": 3.0,   # slightly wider -- visible from further away
        "landmark_b_scale": 0.7,
    },
    "PREDICTIVE_CUE_ABLATED": {
        "n_landmarks_a": 0,
        "n_landmarks_b": 0,
        "landmark_b_resource_bias": 0.8,  # no effect
        "landmark_b_sigma": 2.5,
        "landmark_b_scale": 0.6,
    },
}


def _dist(ax, ay, pos):
    """Euclidean distance from agent (ax, ay) to position (x, y)."""
    return ((ax - pos[0]) ** 2 + (ay - pos[1]) ** 2) ** 0.5


def compute_future_targets(contact_flags, window=FUTURE_WINDOW):
    """
    For each step t, compute binary future target:
    1 if any resource contact occurs in steps t+1..t+window, else 0.

    Args:
        contact_flags: list of bool (length = episode steps)
        window:        look-ahead window

    Returns:
        list of float (same length as contact_flags)
    """
    n = len(contact_flags)
    targets = []
    for t in range(n):
        future_any = any(
            contact_flags[t2]
            for t2 in range(t + 1, min(t + 1 + window, n))
        )
        targets.append(1.0 if future_any else 0.0)
    return targets


def compute_correlation(xs, ys):
    """Pearson correlation. Returns 0.0 if insufficient variance."""
    if len(xs) < 2:
        return 0.0
    xa = np.array(xs, dtype=np.float32)
    ya = np.array(ys, dtype=np.float32)
    n = min(len(xa), len(ya))
    xa, ya = xa[:n], ya[:n]
    if xa.std() < 1e-8 or ya.std() < 1e-8:
        return 0.0
    return float(np.corrcoef(xa, ya)[0, 1])


def run_condition(condition_name, env_cfg, seed):
    """Run one condition x seed. Returns per-seed metrics dict."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = CausalGridWorldV2(
        size=GRID_SIZE,
        num_hazards=N_HAZARDS,
        num_resources=N_RESOURCES,
        use_proxy_fields=True,
        resource_respawn_on_consume=True,
        seed=seed,
        resource_obs_scale=RESOURCE_OBS_SCALE,
        **env_cfg,
    )

    has_landmarks = env_cfg.get("n_landmarks_b", 0) > 0

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

    # P0: optimizer over encoder + E1 (all warms up together)
    encoder_optimizer = torch.optim.Adam(agent.latent_stack.parameters(), lr=1e-4)
    e1_optimizer_p0 = torch.optim.Adam(agent.e1.parameters(), lr=1e-4)

    # P1: optimizer only over schema_readout_head (encoder frozen)
    schema_head_optimizer = torch.optim.Adam(
        agent.e1.schema_readout_head.parameters(), lr=5e-4
    )

    # Metric accumulators (P1 only)
    p1_salience = []
    p1_future_targets = []
    p1_resource_prox = []
    p1_landmark_prox = []
    p1_salience_in_cue_zone = []

    for ep in range(TOTAL_EPISODES):
        _, obs_dict = env.reset()
        agent.reset()
        agent.e1.reset_hidden_state()

        in_p1 = (ep >= P0_EPISODES)

        # Freeze / unfreeze encoder
        for param in agent.latent_stack.parameters():
            param.requires_grad_(not in_p1)

        # Per-episode step data for P1 future-target training
        ep_h_tops = []        # cached E1 hidden states (detached)
        ep_contacts = []      # bool per step

        for step in range(STEPS_PER_EPISODE):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            obs_harm  = obs_dict.get("harm_obs", None)

            if in_p1:
                # P1 waking: inference only -- no computation graph needed.
                # All heavy agent operations run under no_grad to avoid graph buildup
                # across 300-step episodes (would cause severe memory/speed degradation).
                with torch.no_grad():
                    latent = agent.sense(obs_body, obs_world, obs_harm=obs_harm)
                    ticks  = agent.clock.advance()
                    if ticks.get("e1_tick", False):
                        e1_prior = agent._e1_tick(latent)
                    else:
                        e1_prior = torch.zeros(1, cfg.latent.world_dim, device=device)

                    # Cache h_top for episode-end replay training.
                    if agent.e1._hidden_state is not None:
                        h_top = agent.e1._hidden_state[0][-1].clone()  # [1, hidden_dim]
                    else:
                        h_top = torch.zeros(1, cfg.e1.hidden_dim, device=device)

                    candidates = agent.generate_trajectories(latent, e1_prior, ticks)
                    action = agent.select_action(candidates, ticks)

            else:
                # P0: compute graph needed for training
                latent = agent.sense(obs_body, obs_world, obs_harm=obs_harm)
                ticks  = agent.clock.advance()
                if ticks.get("e1_tick", False):
                    e1_prior = agent._e1_tick(latent)
                else:
                    e1_prior = torch.zeros(1, cfg.latent.world_dim, device=device)

                if agent.e1._hidden_state is not None:
                    h_top = agent.e1._hidden_state[0][-1].detach().clone()
                else:
                    h_top = torch.zeros(1, cfg.e1.hidden_dim, device=device)

                candidates = agent.generate_trajectories(latent, e1_prior, ticks)
                action = agent.select_action(candidates, ticks)

            _, harm_signal, done, info, obs_dict = env.step(action)

            benefit = float(info.get("benefit_exposure", 0.0))
            resource_contact = (info.get("transition_type", "") == "resource")

            drive_level = agent.compute_drive_level(obs_body)
            agent.update_z_goal(benefit, drive_level=drive_level)
            agent.serotonin_step(benefit_exposure=benefit)
            if benefit > 0:
                agent.update_benefit_salience(benefit)
            agent.update_schema_wanting(drive_level=float(drive_level))

            # --- P0 training: encoder + resource_proximity_head + E1 prediction ---
            if not in_p1:
                encoder_optimizer.zero_grad()
                e1_optimizer_p0.zero_grad()

                pred_loss = agent.compute_prediction_loss()
                rfv = obs_dict.get("resource_field_view", None)
                if rfv is not None:
                    resource_prox = float(rfv.max()) / max(RESOURCE_OBS_SCALE, 1e-6)
                    rp_loss = agent.compute_resource_proximity_loss(
                        min(resource_prox, 1.0), latent
                    )
                else:
                    rp_loss = torch.tensor(0.0)

                total_loss = pred_loss + 0.5 * rp_loss
                if total_loss.requires_grad:
                    total_loss.backward()
                    encoder_optimizer.step()
                    e1_optimizer_p0.step()

            # --- P1: collect step data ---
            if in_p1:
                ep_h_tops.append(h_top)
                ep_contacts.append(resource_contact)

                sal = float(agent._schema_salience.squeeze()) if agent._schema_salience is not None else 0.0

                rfv = obs_dict.get("resource_field_view", None)
                r_prox = float(rfv.max()) if rfv is not None else 0.0

                lbfv = obs_dict.get("landmark_b_field_view", None)
                lb_prox = float(lbfv.max()) if lbfv is not None else 0.0

                ax_pos = int(env.agent_x)
                ay_pos = int(env.agent_y)

                p1_salience.append(sal)
                p1_resource_prox.append(r_prox)
                p1_landmark_prox.append(lb_prox)

                # Cue zone: near landmark B AND far from any resource
                if has_landmarks and env.landmark_b_positions:
                    near_lb = any(
                        _dist(ax_pos, ay_pos, lbp) <= LANDMARK_NEAR_RADIUS
                        for lbp in env.landmark_b_positions
                    )
                    far_from_resource = all(
                        _dist(ax_pos, ay_pos, (r[0], r[1])) > RESOURCE_FAR_RADIUS
                        for r in env.resources
                    ) if env.resources else True

                    if near_lb and far_from_resource:
                        p1_salience_in_cue_zone.append(sal)

            if done:
                break

        # --- P1 end-of-episode: train schema_readout_head on future targets ---
        # Batch all steps in one forward pass (fast; h_tops are already detached).
        if in_p1 and ep_h_tops:
            future_targets = compute_future_targets(ep_contacts, window=FUTURE_WINDOW)

            h_batch  = torch.cat(ep_h_tops, dim=0)          # [N, hidden_dim]
            ft_batch = torch.tensor(
                future_targets, dtype=torch.float32, device=device
            ).unsqueeze(1)                                    # [N, 1]

            schema_head_optimizer.zero_grad()
            sal_batch = agent.e1.schema_readout_head(h_batch)  # [N, 1]
            ep_loss   = F.mse_loss(sal_batch, ft_batch)
            ep_loss.backward()
            schema_head_optimizer.step()

            # Store future targets aligned to p1_salience (for correlation metrics)
            p1_future_targets.extend(future_targets)

    return {
        "p1_salience": p1_salience,
        "p1_future_targets": p1_future_targets,
        "p1_resource_prox": p1_resource_prox,
        "p1_landmark_prox": p1_landmark_prox,
        "p1_salience_in_cue_zone": p1_salience_in_cue_zone,
        "n_p1_steps": len(p1_salience),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Quick smoke test: 1 seed, 3+10 episodes")
    args = parser.parse_args()

    if args.dry_run:
        global NUM_SEEDS, P0_EPISODES, P1_EPISODES, TOTAL_EPISODES, STEPS_PER_EPISODE
        NUM_SEEDS = 1
        P0_EPISODES = 3
        P1_EPISODES = 10
        TOTAL_EPISODES = P0_EPISODES + P1_EPISODES
        STEPS_PER_EPISODE = 60
        print("[DRY RUN] 1 seed, 3+10 eps, 60 steps/ep")

    results = {}
    for cond_name, env_cfg in CONDITIONS.items():
        results[cond_name] = {}
        for seed_idx in range(NUM_SEEDS):
            seed = 42 + seed_idx
            print(f"Running {cond_name} seed={seed_idx} (seed={seed})...")
            t0 = time.time()
            m = run_condition(cond_name, env_cfg, seed=seed)
            elapsed = time.time() - t0
            ft_corr = compute_correlation(m["p1_salience"], m["p1_future_targets"])
            rp_corr = compute_correlation(m["p1_salience"], m["p1_resource_prox"])
            print(f"  Done in {elapsed:.1f}s -- future_target_corr={ft_corr:.4f}  resource_prox_corr={rp_corr:.4f}  n_p1={m['n_p1_steps']}")
            results[cond_name][f"seed_{seed_idx}"] = m

    # Aggregate per condition
    agg = {}
    for cond_name in CONDITIONS:
        cond_results = results[cond_name]
        all_sal = []
        all_ft = []
        all_rp = []
        all_lp = []
        all_cue = []

        for m in cond_results.values():
            n = len(m["p1_salience"])
            # future_targets may be slightly longer if step count differs -- align
            ft = m["p1_future_targets"][:n]
            all_sal.extend(m["p1_salience"])
            all_ft.extend(ft)
            all_rp.extend(m["p1_resource_prox"])
            all_lp.extend(m["p1_landmark_prox"])
            all_cue.extend(m["p1_salience_in_cue_zone"])

        future_target_corr  = compute_correlation(all_sal, all_ft)
        resource_prox_corr  = compute_correlation(all_sal, all_rp)
        landmark_prox_corr  = compute_correlation(all_sal, all_lp)

        agg[cond_name] = {
            "future_target_corr":       future_target_corr,
            "resource_prox_corr":       resource_prox_corr,
            "landmark_prox_corr":       landmark_prox_corr,
            "salience_in_cue_zone_mean":float(np.mean(all_cue)) if all_cue else 0.0,
            "salience_in_cue_zone_n":   len(all_cue),
            "schema_salience_mean":     float(np.mean(all_sal)) if all_sal else 0.0,
            "future_target_mean":       float(np.mean(all_ft)) if all_ft else 0.0,
            "n_p1_steps":               len(all_sal),
        }

    on_agg  = agg["PREDICTIVE_CUE_ON"]
    abl_agg = agg["PREDICTIVE_CUE_ABLATED"]

    # Acceptance criteria
    c1_pass = on_agg["future_target_corr"] > 0.10
    c2_pass = on_agg["future_target_corr"] > abl_agg["future_target_corr"]
    c3_pass = on_agg["future_target_corr"] > on_agg["resource_prox_corr"]

    outcome = "PASS" if (c1_pass and c2_pass and c3_pass) else "FAIL"

    output = {
        "experiment_type":      EXPERIMENT_TYPE,
        "run_id":               f"{EXPERIMENT_TYPE}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_v3",
        "architecture_epoch":   "ree_hybrid_guardrails_v1",
        "claim_ids":            CLAIM_IDS,
        "experiment_purpose":   EXPERIMENT_PURPOSE,
        "supersedes":           "V3-EXQ-263b",
        "outcome":              outcome,
        "conditions":           list(CONDITIONS.keys()),
        "aggregated":           agg,
        "acceptance_checks": {
            "C1_CUE_ON_future_target_corr_gt_0.10":        c1_pass,
            "C2_CUE_ON_future_target_corr_gt_ABLATED":     c2_pass,
            "C3_CUE_ON_future_target_corr_gt_resource_prox_corr": c3_pass,
        },
        "params": {
            "grid_size":            GRID_SIZE,
            "n_hazards":            N_HAZARDS,
            "n_resources":          N_RESOURCES,
            "p0_episodes":          P0_EPISODES,
            "p1_episodes":          P1_EPISODES,
            "steps_per_episode":    STEPS_PER_EPISODE,
            "num_seeds":            NUM_SEEDS,
            "future_window":        FUTURE_WINDOW,
            "resource_obs_scale":   RESOURCE_OBS_SCALE,
            "landmark_near_radius": LANDMARK_NEAR_RADIUS,
            "resource_far_radius":  RESOURCE_FAR_RADIUS,
        },
        "evidence_direction": "supports" if outcome == "PASS" else "does_not_support",
        "evidence_direction_per_claim": {
            "MECH-216": "supports" if (c1_pass and c2_pass and c3_pass) else "does_not_support",
            "SD-023":   "supports" if (c1_pass and c2_pass) else "does_not_support",
        },
    }

    if not args.dry_run:
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

    print(f"\nOutcome: {outcome}")
    print(f"C1 (CUE_ON future_target_corr > 0.10):        {c1_pass}  [{on_agg['future_target_corr']:.4f}]")
    print(f"C2 (CUE_ON > ABLATED future_target_corr):     {c2_pass}  [{on_agg['future_target_corr']:.4f} vs {abl_agg['future_target_corr']:.4f}]")
    print(f"C3 (future_target_corr > resource_prox_corr): {c3_pass}  [{on_agg['future_target_corr']:.4f} vs {on_agg['resource_prox_corr']:.4f}]")
    print(f"CUE_ON  landmark_prox_corr:    {on_agg['landmark_prox_corr']:.4f}")
    print(f"CUE_ON  salience_in_cue_zone:  {on_agg['salience_in_cue_zone_mean']:.4f} (n={on_agg['salience_in_cue_zone_n']})")
    print(f"ABLATED future_target_corr:    {abl_agg['future_target_corr']:.4f}")
    print(f"ABLATED resource_prox_corr:    {abl_agg['resource_prox_corr']:.4f}")

    return output


if __name__ == "__main__":
    main()
