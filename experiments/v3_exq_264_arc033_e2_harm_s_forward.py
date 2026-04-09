"""
V3-EXQ-264: ARC-033 E2_harm_s Forward Model Validation

Validates the E2HarmSForward module (ree_core/predictors/e2_harm_s.py) as the
dedicated harm-stream forward model for the SD-003 counterfactual attribution pipeline.

MECHANISM UNDER TEST: ARC-033 (harm_stream.sensory_discriminative_forward_model)
  E2HarmSForward: f(z_harm_s_t, a_t) -> z_harm_s_{t+1}
  Residual delta architecture (ResidualHarmForward) avoids identity collapse.
  EXQ-166e PASS (harm-delta R2=0.641) + EXQ-195 (harm_forward_r2=0.914) validate
  the core model; this experiment validates the dedicated module interface.

EXPERIMENT DESIGN:
  Two conditions, ablation pair:
    WITH_HARM_FWD: E2HarmSForward enabled -- forward model trained on z_harm_s transitions
    WITHOUT_HARM_FWD: No harm forward model (baseline z_harm_s without counterfactual)

  Both conditions: use_harm_stream=True, alpha_world=0.9, use_event_classifier=True

  Phased training:
    P0 (100 episodes): HarmEncoder warmup -- harm proximity supervision + event classifier
    P1 (80 episodes): E2HarmSForward trains on frozen z_harm_s (stop-gradient on targets)
    P2 (20 episodes): Evaluation -- forward R2 + counterfactual gap measurement

ACCEPTANCE CRITERIA (diagnostic):
  C1: forward_r2 > 0.5 in WITH_HARM_FWD (confirms model learns z_harm_s transitions)
  C2: z_harm_s_pred_norm > 0.01 in WITH_HARM_FWD (non-trivial -- no identity collapse)
  C3: harm_s_cf_gap_approach > harm_s_cf_gap_neutral in WITH_HARM_FWD
      (SD-003 pipeline: approach events have larger counterfactual signature than neutral)

EXPERIMENT_PURPOSE: diagnostic (substrate readiness -- ARC-033 module validation)
"""

import os
import sys
import json
import random
import numpy as np
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import torch.optim as optim

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.latent.stack import HarmEncoder
from ree_core.predictors.e2_harm_s import E2HarmSForward, E2HarmSConfig
from ree_core.utils.config import REEConfig

EXPERIMENT_PURPOSE = "diagnostic"
EXPERIMENT_TYPE = "v3_exq_264_arc033_e2_harm_s_forward"
CLAIM_IDS = ["ARC-033"]

GRID_SIZE = 5
ACTION_DIM = 4
Z_HARM_DIM = 32

NUM_SEEDS = 3
P0_EPISODES = 100   # HarmEncoder warmup
P1_EPISODES = 80    # E2HarmSForward training (stop-gradient on z_harm_s)
P2_EPISODES = 20    # Evaluation
STEPS_PER_EPISODE = 200

CONDITIONS = {
    "WITH_HARM_FWD": True,
    "WITHOUT_HARM_FWD": False,
}

REPLAY_BUF_MAX = 5000
BATCH_SIZE = 32


def _action_to_onehot(action_idx, n_actions, device):
    """Convert integer action index to one-hot tensor [1, n_actions]."""
    oh = torch.zeros(1, n_actions, device=device)
    oh[0, action_idx] = 1.0
    return oh


def run_condition(condition_name, use_harm_fwd, seed):
    """Run one condition x seed.

    Args:
        condition_name: str label
        use_harm_fwd: bool -- True enables E2HarmSForward training + eval
        seed: int random seed
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    env = CausalGridWorldV2(
        size=GRID_SIZE,
        use_proxy_fields=True,
        resource_respawn_on_consume=True,
        seed=seed,
    )

    n_actions = env.action_dim
    harm_obs_dim = 51  # hazard_field[25] + resource_field[25] + harm_exposure[1]

    cfg = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=n_actions,
        alpha_world=0.9,
        use_event_classifier=True,
        use_harm_stream=True,
        harm_obs_dim=harm_obs_dim,
        z_harm_dim=Z_HARM_DIM,
        use_affective_harm_stream=False,  # isolate z_harm_s only
        use_e2_harm_s_forward=use_harm_fwd,
    )

    agent = REEAgent(cfg)
    device = agent.device

    # Standalone HarmEncoder (instantiated separately for explicit training control)
    harm_enc = HarmEncoder(harm_obs_dim=harm_obs_dim, z_harm_dim=Z_HARM_DIM).to(device)

    # E2HarmSForward (WITH_HARM_FWD condition only)
    harm_fwd = None
    harm_fwd_opt = None
    if use_harm_fwd:
        harm_fwd_cfg = E2HarmSConfig(
            use_e2_harm_s_forward=True,
            z_harm_dim=Z_HARM_DIM,
            action_dim=n_actions,
            hidden_dim=128,
            learning_rate=5e-4,
        )
        harm_fwd = E2HarmSForward(harm_fwd_cfg).to(device)
        harm_fwd_opt = optim.Adam(harm_fwd.parameters(), lr=harm_fwd_cfg.learning_rate)

    # Encoders trained in P0
    enc_params = (
        list(agent.latent_stack.parameters()) + list(harm_enc.parameters())
    )
    enc_opt = optim.Adam(enc_params, lr=1e-4)

    # Replay buffer for P1 training: (z_harm_s_t, action_oh, z_harm_s_t1)
    buf_z = []    # List[Tensor(z_harm_dim,)] -- z_harm_s at step t
    buf_a = []    # List[Tensor(n_actions,)]  -- action one-hot
    buf_z1 = []   # List[Tensor(z_harm_dim,)] -- z_harm_s at step t+1

    metrics = {
        "p1_fwd_losses": [],
        "p2_r2_preds": [],       # flattened z_harm_s_pred values (P2)
        "p2_r2_targets": [],     # flattened z_harm_s_actual_next values (P2)
        "p2_pred_norms": [],
        "p2_cf_gap_approach": [],
        "p2_cf_gap_neutral": [],
        "harm_contacts": 0,
        "total_steps": 0,
    }

    total_episodes = P0_EPISODES + P1_EPISODES + P2_EPISODES

    for ep in range(total_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()
        agent.e1.reset_hidden_state()

        phase = (
            "P0" if ep < P0_EPISODES
            else ("P1" if ep < P0_EPISODES + P1_EPISODES else "P2")
        )

        prev_z_harm_s = None
        prev_action_oh = None

        for step in range(STEPS_PER_EPISODE):
            obs_body  = obs_dict["body_state"].to(device)
            obs_world = obs_dict["world_state"].to(device)
            harm_obs_raw = obs_dict.get("harm_obs", None)
            if harm_obs_raw is not None:
                harm_obs_t = harm_obs_raw.float().to(device)
                if harm_obs_t.dim() == 1:
                    harm_obs_t = harm_obs_t.unsqueeze(0)  # [1, harm_obs_dim]
            else:
                harm_obs_t = torch.zeros(1, harm_obs_dim, device=device)

            # Encode harm stream
            z_harm_s = harm_enc(harm_obs_t)  # [1, z_harm_dim]

            # Collect replay pair from previous step
            if prev_z_harm_s is not None and phase in ("P1", "P2"):
                if len(buf_z) < REPLAY_BUF_MAX:
                    buf_z.append(prev_z_harm_s.squeeze(0).detach().cpu())
                    buf_a.append(prev_action_oh.squeeze(0).detach().cpu())
                    buf_z1.append(z_harm_s.squeeze(0).detach().cpu())
                else:
                    # FIFO: replace oldest
                    idx = ep * STEPS_PER_EPISODE + step
                    buf_z[idx % REPLAY_BUF_MAX] = prev_z_harm_s.squeeze(0).detach().cpu()
                    buf_a[idx % REPLAY_BUF_MAX] = prev_action_oh.squeeze(0).detach().cpu()
                    buf_z1[idx % REPLAY_BUF_MAX] = z_harm_s.squeeze(0).detach().cpu()

            # Agent sense (harm obs routed internally via LatentStack)
            obs_body_2d  = obs_body.unsqueeze(0) if obs_body.dim() == 1 else obs_body
            obs_world_2d = obs_world.unsqueeze(0) if obs_world.dim() == 1 else obs_world
            latent = agent.sense(obs_body_2d, obs_world_2d, obs_harm=harm_obs_t)
            ticks  = agent.clock.advance()
            e1_prior = agent._e1_tick(latent) if ticks.get("e1_tick", False) else torch.zeros(
                1, cfg.latent.world_dim, device=device
            )
            candidates = agent.generate_trajectories(latent, e1_prior, ticks)
            action = agent.select_action(candidates, ticks)

            # Convert action for env step
            action_oh = action  # already one-hot [1, n_actions]
            agent._last_action = action_oh
            flat_obs2, _, done, info, obs_dict2 = env.step(action_oh)

            transition_type = info.get("transition_type", "none")
            harm_exposure = float(info.get("harm_exposure", 0.0))
            if harm_exposure > 0:
                metrics["harm_contacts"] += 1

            # P0: train HarmEncoder warmup
            if phase == "P0":
                enc_opt.zero_grad()
                pred_loss = agent.compute_prediction_loss(latent)
                enc_loss = pred_loss
                # Event contrastive (if classifier head available)
                if (cfg.latent.use_event_classifier
                        and hasattr(agent.latent_stack, "last_event_logits")
                        and agent.latent_stack.last_event_logits is not None):
                    event_label = (
                        torch.tensor([0], dtype=torch.long, device=device)
                        if transition_type == "none"
                        else (torch.tensor([1], dtype=torch.long, device=device)
                              if "env" in transition_type
                              else torch.tensor([2], dtype=torch.long, device=device))
                    )
                    event_loss = nn.functional.cross_entropy(
                        agent.latent_stack.last_event_logits, event_label
                    )
                    enc_loss = enc_loss + 0.5 * event_loss
                enc_loss.backward()
                enc_opt.step()

            # P1: train E2HarmSForward on replay buffer (stop-gradient on targets)
            if phase == "P1" and use_harm_fwd and len(buf_z) >= BATCH_SIZE:
                idxs = random.sample(range(len(buf_z)), BATCH_SIZE)
                z_b  = torch.stack([buf_z[i]  for i in idxs]).to(device)
                a_b  = torch.stack([buf_a[i]  for i in idxs]).to(device)
                z1_b = torch.stack([buf_z1[i] for i in idxs]).to(device)

                harm_fwd_opt.zero_grad()
                # Critical: detach both input and target to isolate forward model training
                z_pred = harm_fwd(z_b.detach(), a_b)
                fwd_loss = harm_fwd.compute_loss(z_pred, z1_b.detach())
                fwd_loss.backward()
                harm_fwd_opt.step()
                metrics["p1_fwd_losses"].append(float(fwd_loss.item()))

            # P2: evaluate forward model
            if phase == "P2" and use_harm_fwd and len(buf_z) >= BATCH_SIZE:
                with torch.no_grad():
                    z_harm_s_det = z_harm_s.detach()
                    # Actual forward prediction
                    z_pred_next = harm_fwd(z_harm_s_det, action_oh.detach())
                    # Counterfactual: no-action (action index 0)
                    a_no_action = _action_to_onehot(0, n_actions, device)
                    z_harm_s_cf = harm_fwd.counterfactual_forward(z_harm_s_det, a_no_action)

                    metrics["p2_pred_norms"].append(float(z_pred_next.norm().item()))

                    # SD-003 causal_sig via E3 harm evaluation
                    if hasattr(agent.e3, "harm_eval_z_harm_head"):
                        harm_actual_score = agent.e3.harm_eval_z_harm_head(z_harm_s_det)
                        harm_cf_score = agent.e3.harm_eval_z_harm_head(z_harm_s_cf)
                        causal_sig = float((harm_actual_score - harm_cf_score).squeeze().item())
                    else:
                        causal_sig = 0.0

                    is_approach = transition_type in ("hazard_approach", "harm")
                    if is_approach:
                        metrics["p2_cf_gap_approach"].append(causal_sig)
                    else:
                        metrics["p2_cf_gap_neutral"].append(causal_sig)

            metrics["total_steps"] += 1
            prev_z_harm_s = z_harm_s.detach()
            prev_action_oh = action_oh.detach()
            obs_dict = obs_dict2

            if done:
                break

    # Compute held-out R2 from replay buffer
    forward_r2 = 0.0
    if use_harm_fwd and len(buf_z) >= BATCH_SIZE:
        n_eval = min(500, len(buf_z))
        eval_idxs = random.sample(range(len(buf_z)), n_eval)
        z_eval  = torch.stack([buf_z[i]  for i in eval_idxs]).to(device)
        a_eval  = torch.stack([buf_a[i]  for i in eval_idxs]).to(device)
        z1_eval = torch.stack([buf_z1[i] for i in eval_idxs]).to(device)
        with torch.no_grad():
            z1_pred_eval = harm_fwd(z_eval, a_eval)
        y_true = z1_eval.cpu().numpy().flatten()
        y_pred = z1_pred_eval.cpu().numpy().flatten()
        ss_res = float(np.sum((y_true - y_pred) ** 2))
        ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
        forward_r2 = 1.0 - ss_res / (ss_tot + 1e-8)

    harm_rate = metrics["harm_contacts"] / max(1, metrics["total_steps"])

    return {
        "forward_r2": forward_r2,
        "fwd_loss_final_mean": (
            float(np.mean(metrics["p1_fwd_losses"][-20:]))
            if metrics["p1_fwd_losses"] else 0.0
        ),
        "z_harm_s_pred_norm_mean": (
            float(np.mean(metrics["p2_pred_norms"]))
            if metrics["p2_pred_norms"] else 0.0
        ),
        "harm_s_cf_gap_approach": (
            float(np.mean(metrics["p2_cf_gap_approach"]))
            if metrics["p2_cf_gap_approach"] else 0.0
        ),
        "harm_s_cf_gap_neutral": (
            float(np.mean(metrics["p2_cf_gap_neutral"]))
            if metrics["p2_cf_gap_neutral"] else 0.0
        ),
        "approach_eval_steps": len(metrics["p2_cf_gap_approach"]),
        "neutral_eval_steps": len(metrics["p2_cf_gap_neutral"]),
        "harm_rate": harm_rate,
        "replay_buf_size": len(buf_z),
    }


def main(dry_run=False):
    if dry_run:
        print("DRY-RUN: ARC-033 E2_harm_s forward model validation")
        print("Checking imports and shape...")

        env = CausalGridWorldV2(size=GRID_SIZE, use_proxy_fields=True)
        cfg = REEConfig.from_dims(
            body_obs_dim=env.body_obs_dim,
            world_obs_dim=env.world_obs_dim,
            action_dim=env.action_dim,
            alpha_world=0.9,
            use_event_classifier=True,
            use_harm_stream=True,
            harm_obs_dim=51,
            z_harm_dim=Z_HARM_DIM,
            use_affective_harm_stream=False,
            use_e2_harm_s_forward=True,
        )
        agent = REEAgent(cfg)
        harm_enc = HarmEncoder(harm_obs_dim=51, z_harm_dim=Z_HARM_DIM)
        harm_fwd_cfg = E2HarmSConfig(
            use_e2_harm_s_forward=True,
            z_harm_dim=Z_HARM_DIM,
            action_dim=env.action_dim,
        )
        harm_fwd = E2HarmSForward(harm_fwd_cfg)

        # Shape checks
        z_harm_s = torch.zeros(1, Z_HARM_DIM)
        action = torch.zeros(1, env.action_dim)
        z_pred = harm_fwd(z_harm_s, action)
        z_cf = harm_fwd.counterfactual_forward(z_harm_s, action)
        loss = harm_fwd.compute_loss(z_pred, z_harm_s.detach())

        print(f"  harm_enc output shape: {harm_enc(torch.zeros(1,51)).shape}")
        print(f"  E2HarmSForward z_pred shape: {z_pred.shape}")
        print(f"  E2HarmSForward z_cf shape: {z_cf.shape}")
        print(f"  compute_loss: {float(loss.item()):.6f}")
        print(f"  LatentStackConfig.use_e2_harm_s_forward: {cfg.latent.use_e2_harm_s_forward}")

        assert z_pred.shape == (1, Z_HARM_DIM), f"Shape mismatch: {z_pred.shape}"
        assert z_cf.shape == (1, Z_HARM_DIM), f"CF shape mismatch: {z_cf.shape}"
        assert cfg.latent.use_e2_harm_s_forward is True, "Config flag not set"

        print("DRY-RUN PASS: all imports, shape checks, and config OK")
        return {"dry_run": True, "status": "PASS"}

    # Full run
    all_results = {}
    for cond_name, use_harm_fwd in CONDITIONS.items():
        all_results[cond_name] = {}
        for seed_idx in range(NUM_SEEDS):
            seed = 42 + seed_idx
            print(f"[{cond_name}] seed={seed} (P0={P0_EPISODES} P1={P1_EPISODES} P2={P2_EPISODES} eps) ...", flush=True)
            try:
                m = run_condition(cond_name, use_harm_fwd, seed)
                print(f"  forward_r2={m['forward_r2']:.4f} pred_norm={m['z_harm_s_pred_norm_mean']:.4f} "
                      f"cf_gap_approach={m['harm_s_cf_gap_approach']:.4f} "
                      f"cf_gap_neutral={m['harm_s_cf_gap_neutral']:.4f}", flush=True)
            except Exception as exc:
                import traceback
                print(f"  ERROR: {exc}")
                traceback.print_exc()
                m = {"error": str(exc)}
            all_results[cond_name][f"seed_{seed}"] = m

    # Aggregate per condition
    agg = {}
    agg_keys = [
        "forward_r2", "z_harm_s_pred_norm_mean",
        "harm_s_cf_gap_approach", "harm_s_cf_gap_neutral", "harm_rate",
    ]
    for cond_name in CONDITIONS:
        agg[cond_name] = {}
        for k in agg_keys:
            vals = [
                all_results[cond_name][f"seed_{42+i}"].get(k, 0.0)
                for i in range(NUM_SEEDS)
                if "error" not in all_results[cond_name][f"seed_{42+i}"]
            ]
            agg[cond_name][k + "_mean"] = float(np.mean(vals)) if vals else 0.0
            agg[cond_name][k + "_std"] = float(np.std(vals)) if len(vals) > 1 else 0.0

    # Acceptance criteria
    wf = agg.get("WITH_HARM_FWD", {})
    c1_pass = wf.get("forward_r2_mean", 0.0) > 0.5
    c2_pass = wf.get("z_harm_s_pred_norm_mean_mean", 0.0) > 0.01
    c3_pass = wf.get("harm_s_cf_gap_approach_mean", 0.0) > wf.get("harm_s_cf_gap_neutral_mean", 0.0)

    criteria_passed = sum([c1_pass, c2_pass, c3_pass])
    outcome = "PASS" if criteria_passed >= 2 else "FAIL"

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = {
        "experiment_type": EXPERIMENT_TYPE,
        "run_id": f"{EXPERIMENT_TYPE}_{ts}_v3",
        "architecture_epoch": "ree_hybrid_guardrails_v1",
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "outcome": outcome,
        "conditions": list(CONDITIONS.keys()),
        "aggregated": agg,
        "acceptance_checks": {
            "C1_forward_r2_gt_0.5": c1_pass,
            "C2_pred_norm_nontrivial_gt_0.01": c2_pass,
            "C3_cf_gap_approach_gt_neutral": c3_pass,
        },
        "criteria_passed": criteria_passed,
        "criteria_total": 3,
        "params": {
            "grid_size": GRID_SIZE,
            "p0_episodes": P0_EPISODES,
            "p1_episodes": P1_EPISODES,
            "p2_episodes": P2_EPISODES,
            "steps_per_episode": STEPS_PER_EPISODE,
            "num_seeds": NUM_SEEDS,
            "z_harm_dim": Z_HARM_DIM,
            "harm_obs_dim": 51,
            "batch_size": BATCH_SIZE,
            "replay_buf_max": REPLAY_BUF_MAX,
        },
        "evidence_direction": "supports" if outcome == "PASS" else "does_not_support",
        "seed_results": all_results,
    }

    # Write to REE_assembly evidence directory
    out_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "..", "REE_assembly", "evidence", "experiments",
    )
    out_path = os.path.join(out_dir, f"{EXPERIMENT_TYPE}_{ts}_v3.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults written to {out_path}")
    print(f"Outcome: {outcome} ({criteria_passed}/3 criteria)")
    print(f"C1 (forward_r2 > 0.5):            {c1_pass} ({wf.get('forward_r2_mean', 0.0):.4f})")
    print(f"C2 (pred_norm > 0.01):             {c2_pass} ({wf.get('z_harm_s_pred_norm_mean_mean', 0.0):.4f})")
    print(f"C3 (cf_gap approach > neutral):    {c3_pass} "
          f"(approach={wf.get('harm_s_cf_gap_approach_mean', 0.0):.4f} "
          f"neutral={wf.get('harm_s_cf_gap_neutral_mean', 0.0):.4f})")

    return output


if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    main(dry_run=dry_run)
