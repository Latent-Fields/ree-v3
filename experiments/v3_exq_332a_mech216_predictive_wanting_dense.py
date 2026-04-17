"""
V3-EXQ-332a: MECH-216 Genuinely Predictive Wanting -- Adjudication Run

Supersedes: V3-EXQ-332

=== Implementation gap diagnosed in EXQ-332 ===

EXQ-332 produced future_target_mean=0.0006 across 162,000 P1 steps --
nearly zero positive labels. Root cause: 2 resources on a 12x12 grid with
purely random P1 actions yields ~10 actual resource contacts over 540
episodes. MSE loss is then ~100% zero-class and the schema_readout_head
learns to output constant ~0. correlation(salience, future_target) ~= 0 is
a ceiling artifact of sparse supervision, NOT evidence that E1 cannot form
anticipatory predictions.

Supporting context from lit-pull SD-023 (2026-04-17):
  - Yaghoubi 2026 (Nature): hippocampal temporal coding shift requires
    hundreds of reward exposures. 10 contacts over 540 P1 episodes is far
    below any plausible learning threshold.
  - Schultz 1997 (Science): DA prediction transfer to distal cues requires
    consistent Pavlovian pairing (many CSx-US co-occurrences).

The fix targets supervision density, not the architectural question.
The scientific question (does Landmark B enable anticipatory prediction?)
is unchanged.

=== Changes from EXQ-332 ===

1. Grid 12x12 -> 8x8; resources 2 -> 4 (positive label density ~10x higher)
2. FUTURE_WINDOW 10 -> 30 (positive label per contact: 30 instead of 10 steps)
3. P1 episodes 180 -> 240 (more training budget)
4. Add total_resource_contacts diagnostic to output
5. resource_obs_scale stays 0.3 (keeps landmark B as the useful cue signal)
6. Evidence_direction_per_claim split: SD-023 (substrate) vs MECH-216 (mechanism)

Expected resource contact rate on 8x8 with 4 resources, random walk:
  P(contact per step) ~ 4/64 = 6.25%
  Expected contacts per episode: 0.0625 * 300 = ~19
  P1 episodes: 3 seeds * 240 eps = 720 eps
  Expected total contacts: ~13,680
  future_target positive rate: ~min(30*0.0625, 1) = ~100% near resources
  Overall positive rate: ~19*30 / 300 = ~190/300 = ~63% of steps

This is sufficient for the schema_readout_head to learn a meaningful
correlation signal.

MECHANISM UNDER TEST:
  MECH-216 (e1_predictive_wanting) + SD-023 (environment.gradient_texture)

EXPERIMENT DESIGN:
  Two conditions -- both with resource_obs_scale=0.3:
    PREDICTIVE_CUE_ON:      n_landmarks_b=2, landmark_b_resource_bias=0.8
                            Landmark B fields unattenuated; strong spatial
                            bias toward resources; serves as leading indicator.
    PREDICTIVE_CUE_ABLATED: n_landmarks_b=0
                            No Landmark B; direct resource signal attenuated;
                            no useful leading indicator in world_state.

  Landmark A (n=1) present in BOTH conditions as navigation texture that is
  NOT resource-predictive. Controls for generic texture benefit.

  3 seeds, grid_size=8, n_hazards=1, n_resources=4, FUTURE_WINDOW=30

Training phases:
  P0 (60 episodes): encoder + resource_prox head warmup
    - LatentStack + resource_proximity_head trained on attenuated resource field
    - E1 LSTM trained on prediction continuity loss
    - schema_readout head NOT trained
  P1 (240 episodes): schema_readout_head trained on future_target
    - Encoder frozen
    - Each step: cache h_top (detached), cache resource_contact bool
    - Each episode end: compute future_target_t retrospectively, backprop MSE
      through schema_readout_head only

Primary metrics:
  future_target_corr:        corr(schema_salience_t, future_target_t) -- PRIMARY
  resource_prox_corr:        corr(schema_salience_t, current_resource_prox_t)
  landmark_prox_corr:        corr(schema_salience_t, landmark_b_prox_t) [CUE_ON]
  salience_in_cue_zone_mean: mean salience near LB but far from resource
  total_resource_contacts:   DIAGNOSTIC -- confirms supervision density was adequate
  positive_label_rate:       fraction of P1 steps where future_target=1

ACCEPTANCE CRITERIA (PASS requires C1 + C2; C3 is informative):
  C1: PREDICTIVE_CUE_ON future_target_corr > 0.10 (signal above noise)
  C2: CUE_ON future_target_corr > ABLATED future_target_corr (cue enables prediction)
  C3: positive_label_rate > 0.10 (supervision was dense enough to be informative)
      [C3 is a quality gate, not a scientific criterion -- if C3 fails, result is
       non_contributory due to same sparse-label artifact as EXQ-332]

Evidence direction per claim:
  SD-023 (substrate): supports if C1+C2 pass; does_not_support otherwise
  MECH-216 (mechanism): supports if C1+C2 pass; non_contributory if C3 fails
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

EXPERIMENT_PURPOSE = "diagnostic"
EXPERIMENT_TYPE = "v3_exq_332a_mech216_predictive_wanting_dense"
CLAIM_IDS = ["MECH-216", "SD-023"]

GRID_SIZE = 8
ACTION_DIM = 4
N_HAZARDS = 1
N_RESOURCES = 4

NUM_SEEDS = 3
P0_EPISODES = 60
P1_EPISODES = 240
STEPS_PER_EPISODE = 300
TOTAL_EPISODES = P0_EPISODES + P1_EPISODES

FUTURE_WINDOW = 30         # binary: any resource contact in next N steps
RESOURCE_OBS_SCALE = 0.3   # keep landmark B as dominant useful cue

LANDMARK_NEAR_RADIUS = 2   # cells -- "near landmark B" zone
RESOURCE_FAR_RADIUS = 2    # cells -- "far from resource" threshold


CONDITIONS = {
    "PREDICTIVE_CUE_ON": {
        "n_landmarks_a": 1,    # neutral texture, NOT resource-predictive
        "n_landmarks_b": 2,    # predictive cue biased near resources
        "landmark_b_resource_bias": 0.8,
        "landmark_b_sigma": 3.0,
        "landmark_b_scale": 0.7,
    },
    "PREDICTIVE_CUE_ABLATED": {
        "n_landmarks_a": 1,    # same neutral texture as CUE_ON (controls for generic texture)
        "n_landmarks_b": 0,    # no predictive cue
        "landmark_b_resource_bias": 0.8,
        "landmark_b_sigma": 2.5,
        "landmark_b_scale": 0.6,
    },
}


def _dist(ax, ay, pos):
    return ((ax - pos[0]) ** 2 + (ay - pos[1]) ** 2) ** 0.5


def compute_future_targets(contact_flags, window=FUTURE_WINDOW):
    """For each step t, binary 1 if any resource contact in t+1..t+window."""
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
    """Pearson correlation; returns 0.0 on insufficient variance."""
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
        hazard_harm=0.1,
        use_proxy_fields=True,
        resource_respawn_on_consume=True,
        seed=seed,
        resource_obs_scale=RESOURCE_OBS_SCALE,
        **env_cfg,
    )

    has_landmarks_b = env_cfg.get("n_landmarks_b", 0) > 0

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

    encoder_optimizer = torch.optim.Adam(agent.latent_stack.parameters(), lr=1e-4)
    e1_optimizer_p0 = torch.optim.Adam(agent.e1.parameters(), lr=1e-4)

    schema_head_optimizer = torch.optim.Adam(
        agent.e1.schema_readout_head.parameters(), lr=5e-4
    )

    p1_salience = []
    p1_future_targets = []
    p1_resource_prox = []
    p1_landmark_prox = []
    p1_salience_in_cue_zone = []
    total_resource_contacts = 0

    for ep in range(TOTAL_EPISODES):
        _, obs_dict = env.reset()
        agent.reset()
        agent.e1.reset_hidden_state()

        in_p1 = (ep >= P0_EPISODES)

        for param in agent.latent_stack.parameters():
            param.requires_grad_(not in_p1)

        ep_h_tops = []
        ep_contacts = []

        for step in range(STEPS_PER_EPISODE):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            obs_harm  = obs_dict.get("harm_obs", None)

            if in_p1:
                with torch.no_grad():
                    latent = agent.sense(obs_body, obs_world, obs_harm=obs_harm)
                    ticks  = agent.clock.advance()
                    if ticks.get("e1_tick", False):
                        agent._e1_tick(latent)

                    if agent.e1._hidden_state is not None:
                        h_top = agent.e1._hidden_state[0][-1].clone()
                    else:
                        h_top = torch.zeros(1, cfg.e1.hidden_dim, device=device)

                    action = torch.randint(0, ACTION_DIM, (1,))

            else:
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

            if resource_contact and in_p1:
                total_resource_contacts += 1

            drive_level = agent.compute_drive_level(obs_body)
            agent.update_z_goal(benefit, drive_level=drive_level)
            agent.serotonin_step(benefit_exposure=benefit)
            if benefit > 0:
                agent.update_benefit_salience(benefit)
            agent.update_schema_wanting(drive_level=float(drive_level))

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

                if has_landmarks_b and env.landmark_b_positions:
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

        if in_p1 and ep_h_tops:
            future_targets = compute_future_targets(ep_contacts, window=FUTURE_WINDOW)

            h_batch  = torch.cat(ep_h_tops, dim=0)
            ft_batch = torch.tensor(
                future_targets, dtype=torch.float32, device=device
            ).unsqueeze(1)

            schema_head_optimizer.zero_grad()
            sal_batch = agent.e1.schema_readout_head(h_batch)
            ep_loss   = F.mse_loss(sal_batch, ft_batch)
            ep_loss.backward()
            schema_head_optimizer.step()

            p1_future_targets.extend(future_targets)

    return {
        "p1_salience": p1_salience,
        "p1_future_targets": p1_future_targets,
        "p1_resource_prox": p1_resource_prox,
        "p1_landmark_prox": p1_landmark_prox,
        "p1_salience_in_cue_zone": p1_salience_in_cue_zone,
        "n_p1_steps": len(p1_salience),
        "total_resource_contacts": total_resource_contacts,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Quick smoke test: 1 seed, 3+10 episodes, 60 steps")
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
            pos_rate = float(np.mean(m["p1_future_targets"])) if m["p1_future_targets"] else 0.0
            print(f"  Done in {elapsed:.1f}s -- future_target_corr={ft_corr:.4f}  "
                  f"resource_prox_corr={rp_corr:.4f}  "
                  f"contacts={m['total_resource_contacts']}  "
                  f"pos_rate={pos_rate:.3f}  n_p1={m['n_p1_steps']}")
            results[cond_name][f"seed_{seed_idx}"] = m

    agg = {}
    for cond_name in CONDITIONS:
        cond_results = results[cond_name]
        all_sal, all_ft, all_rp, all_lp, all_cue = [], [], [], [], []
        total_contacts = 0

        for m in cond_results.values():
            n = len(m["p1_salience"])
            ft = m["p1_future_targets"][:n]
            all_sal.extend(m["p1_salience"])
            all_ft.extend(ft)
            all_rp.extend(m["p1_resource_prox"])
            all_lp.extend(m["p1_landmark_prox"])
            all_cue.extend(m["p1_salience_in_cue_zone"])
            total_contacts += m["total_resource_contacts"]

        future_target_corr = compute_correlation(all_sal, all_ft)
        resource_prox_corr = compute_correlation(all_sal, all_rp)
        landmark_prox_corr = compute_correlation(all_sal, all_lp)
        positive_label_rate = float(np.mean(all_ft)) if all_ft else 0.0

        agg[cond_name] = {
            "future_target_corr":       future_target_corr,
            "resource_prox_corr":       resource_prox_corr,
            "landmark_prox_corr":       landmark_prox_corr,
            "salience_in_cue_zone_mean":float(np.mean(all_cue)) if all_cue else 0.0,
            "salience_in_cue_zone_n":   len(all_cue),
            "schema_salience_mean":     float(np.mean(all_sal)) if all_sal else 0.0,
            "future_target_mean":       float(np.mean(all_ft)) if all_ft else 0.0,
            "positive_label_rate":      positive_label_rate,
            "total_resource_contacts":  total_contacts,
            "n_p1_steps":               len(all_sal),
        }

    on_agg  = agg["PREDICTIVE_CUE_ON"]
    abl_agg = agg["PREDICTIVE_CUE_ABLATED"]

    c1_pass = on_agg["future_target_corr"] > 0.10
    c2_pass = on_agg["future_target_corr"] > abl_agg["future_target_corr"]
    c3_pass = on_agg["positive_label_rate"] > 0.10   # quality gate

    # If C3 fails (sparse labels again), result is non_contributory
    if not c3_pass:
        sd023_dir = "non_contributory"
        mech216_dir = "non_contributory"
        outcome = "FAIL"
    else:
        science_pass = c1_pass and c2_pass
        outcome = "PASS" if science_pass else "FAIL"
        sd023_dir   = "supports" if science_pass else "does_not_support"
        mech216_dir = "supports" if science_pass else "does_not_support"

    output = {
        "experiment_type":    EXPERIMENT_TYPE,
        "run_id":             f"{EXPERIMENT_TYPE}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_v3",
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids":          CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "supersedes":         "V3-EXQ-332",
        "outcome":            outcome,
        "conditions":         list(CONDITIONS.keys()),
        "aggregated":         agg,
        "acceptance_checks": {
            "C1_CUE_ON_future_target_corr_gt_0.10":    c1_pass,
            "C2_CUE_ON_future_target_corr_gt_ABLATED": c2_pass,
            "C3_positive_label_rate_gt_0.10":          c3_pass,
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
        "evidence_direction": sd023_dir if (sd023_dir != "non_contributory") else "non_contributory",
        "evidence_direction_per_claim": {
            "MECH-216": mech216_dir,
            "SD-023":   sd023_dir,
        },
        "implementation_gap_note": (
            "EXQ-332 failed due to sparse positive labels (future_target_mean=0.0006, "
            "~10 contacts across 162k P1 steps). Adjudication run increases resource "
            "density (4 resources, 8x8 grid) and future_window (30 steps). "
            "C3 quality gate confirms supervision density was adequate before "
            "interpreting C1/C2 as scientific results."
        ),
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
    print(f"C1 (CUE_ON future_target_corr > 0.10): {c1_pass}  [{on_agg['future_target_corr']:.4f}]")
    print(f"C2 (CUE_ON > ABLATED):                 {c2_pass}  [{on_agg['future_target_corr']:.4f} vs {abl_agg['future_target_corr']:.4f}]")
    print(f"C3 quality gate (pos_rate > 0.10):      {c3_pass}  [{on_agg['positive_label_rate']:.3f}]")
    print(f"CUE_ON  total_resource_contacts: {on_agg['total_resource_contacts']}")
    print(f"ABLATED total_resource_contacts: {abl_agg['total_resource_contacts']}")
    print(f"CUE_ON  landmark_prox_corr:  {on_agg['landmark_prox_corr']:.4f}")
    print(f"CUE_ON  salience_in_cue_zone: {on_agg['salience_in_cue_zone_mean']:.4f} (n={on_agg['salience_in_cue_zone_n']})")

    return output


if __name__ == "__main__":
    main()
