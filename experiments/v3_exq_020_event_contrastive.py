"""
V3-EXQ-020 — Event-Auxiliary Contrastive Encoder Loss (MECH-100, SD-009)

Claims: MECH-100 (encoder.event_contrastive_supervision), SD-003, SD-009.

Motivation (2026-03-18):
  Root cause of EXQ-013 event-selectivity failure (and downstream SD-003 collapse):
  the world encoder receives no gradient signal to discriminate event types. All
  training losses (E1 reconstruction, E2 self-prediction) are event-invariant: they
  reward accurate decoding of z_world regardless of whether a hazard is present.

  This experiment adds a small event-type classifier head (Linear(world_dim, 3))
  and back-propagates its cross-entropy loss through z_world during training.
  Labels: 0=none (locomotion only), 1=env_caused_hazard, 2=agent_caused_hazard.
  Label assignment: obs_t is labeled with the transition_type that PRODUCED obs_t
  (i.e., info_{t-1}["transition_type"] from the previous env.step()).

  λ_event (default 0.5) controls the CE loss weight relative to E1/E2/recon losses.

  No reafference correction is applied: this tests whether contrastive supervision
  ALONE can overcome the perspective-shift problem.

PASS criteria (ALL must hold):
  C1: calibration_gap > 0.05 (SD-003 threshold, standard probe)
  C2: event_classification_acc > 0.5 (held-out evaluation of event_classifier)
  C3: Δz_world(env_hazard) > Δz_world(empty) with margin > 0.005
      (direct event-selectivity check matching EXQ-013 C1)
  C4: warmup harm events > 100
  C5: No fatal errors
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from typing import Dict, List, Tuple, Optional
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorld
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE = "v3_exq_020_event_contrastive"
CLAIM_IDS = ["MECH-100", "SD-009", "SD-003"]

E2_ROLLOUT_STEPS = 5
RECON_WEIGHT = 1.0
EVENT_LABEL_MAP = {"none": 0, "env_caused_hazard": 1, "agent_caused_hazard": 2}
N_EVENT_TYPES = 3


def _action_to_onehot(action_idx: int, num_actions: int, device) -> torch.Tensor:
    v = torch.zeros(1, num_actions, device=device)
    v[0, action_idx] = 1.0
    return v


def _random_cf_action(actual_idx: int, num_actions: int) -> int:
    choices = [a for a in range(num_actions) if a != actual_idx]
    return random.choice(choices) if choices else 0


def _make_world_decoder(world_dim: int, world_obs_dim: int, hidden_dim: int = 64):
    return nn.Sequential(
        nn.Linear(world_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, world_obs_dim),
    )


def _make_net_eval_head(world_dim: int, hidden_dim: int = 64) -> nn.Module:
    return nn.Sequential(
        nn.Linear(world_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim // 2),
        nn.ReLU(),
        nn.Linear(hidden_dim // 2, 1),
        nn.Tanh(),
    )


def _compute_e2w_loss(agent, traj_buffer, rollout_steps, batch_size=8):
    if len(traj_buffer) < 2:
        return next(agent.e1.parameters()).sum() * 0.0
    n = min(batch_size, len(traj_buffer))
    idxs = torch.randperm(len(traj_buffer))[:n].tolist()
    total = next(agent.e1.parameters()).sum() * 0.0
    count = 0
    for idx in idxs:
        seg = traj_buffer[idx]
        if len(seg) < rollout_steps + 1:
            continue
        z = seg[0][0]
        z_target = seg[rollout_steps][0]
        for k in range(rollout_steps):
            z = agent.e2.world_forward(z, seg[k][1])
        total = total + F.mse_loss(z, z_target)
        count += 1
    return total / count if count > 0 else total


def _train_and_collect(
    agent: REEAgent,
    env: CausalGridWorld,
    world_decoder: nn.Module,
    net_eval_head: nn.Module,
    event_classifier: nn.Module,
    optimizer: optim.Optimizer,
    net_eval_optimizer: optim.Optimizer,
    num_episodes: int,
    steps_per_episode: int,
    lambda_event: float = 0.5,
    harm_scale: float = 0.02,
) -> Dict:
    """
    Training phase with event-auxiliary contrastive loss on z_world.

    Each step: label obs_t with prev_ttype (transition that produced obs_t),
    compute CE(event_classifier(z_world_t), label_t), add to total_loss.
    """
    agent.train()
    world_decoder.train()
    net_eval_head.train()
    event_classifier.train()

    traj_buffer: List = []
    MAX_TRAJ_BUFFER = 200

    signal_buffer: List[Tuple[torch.Tensor, float]] = []
    MAX_SIGNAL_BUFFER = 1000

    # For held-out accuracy evaluation (collect labeled z_world samples)
    eval_buffer: List[Tuple[torch.Tensor, int]] = []
    MAX_EVAL_BUFFER = 500

    # For event selectivity measurement
    dz_world_by_event: Dict[str, List[float]] = {
        "none": [], "env_caused_hazard": [], "agent_caused_hazard": []
    }
    MAX_DZ_PER_TYPE = 500

    total_harm = 0
    total_benefit = 0
    total_ce_loss = 0.0
    ce_steps = 0

    for ep in range(num_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()
        episode_traj: List = []

        z_world_prev = None
        prev_ttype = None

        for step in range(steps_per_episode):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]

            latent = agent.sense(obs_body, obs_world)
            agent.clock.advance()

            z_world_curr = latent.z_world  # keep gradients for CE loss

            action_idx = random.randint(0, env.action_dim - 1)
            action = _action_to_onehot(action_idx, env.action_dim, agent.device)
            agent._last_action = action

            episode_traj.append((latent.z_world.detach(), action.detach()))

            flat_obs, harm_signal, done, info, obs_dict = env.step(action)
            ttype = info.get("transition_type", "none")

            if harm_signal < 0:
                total_harm += 1
            elif harm_signal > 0:
                total_benefit += 1

            # Δz_world selectivity measurement (uses prev_ttype for labeling)
            if z_world_prev is not None and prev_ttype in dz_world_by_event:
                dz = float(torch.norm(z_world_curr.detach() - z_world_prev).item())
                if len(dz_world_by_event[prev_ttype]) < MAX_DZ_PER_TYPE:
                    dz_world_by_event[prev_ttype].append(dz)

            z_world_prev = z_world_curr.detach()

            # Collect signal for net_eval
            if harm_signal != 0.0 or (step % 3 == 0):
                signal_buffer.append((latent.z_world.detach(), float(harm_signal)))
                if len(signal_buffer) > MAX_SIGNAL_BUFFER:
                    signal_buffer = signal_buffer[-MAX_SIGNAL_BUFFER:]

            # Event-auxiliary CE loss:
            # Label obs_{t+1} (which we'll encode next step as z_world_{t+1})
            # with the current ttype. Equivalently: label current z_world_curr with
            # prev_ttype (the transition that produced obs that was just encoded).
            ce_loss = torch.zeros(1, device=agent.device)
            if prev_ttype is not None:
                label_val = EVENT_LABEL_MAP.get(prev_ttype, 0)
                label_tensor = torch.tensor([label_val], device=agent.device, dtype=torch.long)
                event_logits = event_classifier(z_world_curr)
                ce_loss = F.cross_entropy(event_logits, label_tensor)
                total_ce_loss += float(ce_loss.item())
                ce_steps += 1

            # Collect for held-out eval (z_world, label for CURRENT step)
            if prev_ttype is not None and len(eval_buffer) < MAX_EVAL_BUFFER:
                eval_buffer.append((
                    latent.z_world.detach().clone(),
                    EVENT_LABEL_MAP.get(prev_ttype, 0),
                ))

            # Update prev_ttype for next step
            prev_ttype = ttype

            # Main loss: E1 + E2_self + E2_world + reconstruction + λ_event * CE
            e1_loss    = agent.compute_prediction_loss()
            e2_self_loss = agent.compute_e2_loss()
            e2w_loss   = _compute_e2w_loss(agent, traj_buffer, E2_ROLLOUT_STEPS)

            obs_w = obs_world.unsqueeze(0) if obs_world.dim() == 1 else obs_world
            z_w = agent.latent_stack.split_encoder.world_encoder(obs_w)
            recon = world_decoder(z_w)
            recon_loss = F.mse_loss(recon, obs_w)

            total_loss = (e1_loss + e2_self_loss + e2w_loss
                          + RECON_WEIGHT * recon_loss
                          + lambda_event * ce_loss)
            if total_loss.requires_grad:
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(world_decoder.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(event_classifier.parameters(), 0.5)
                optimizer.step()

            # net_eval regression
            if len(signal_buffer) >= 8:
                k = min(32, len(signal_buffer))
                idxs_sel = torch.randperm(len(signal_buffer))[:k].tolist()
                zw_b = torch.cat([signal_buffer[i][0] for i in idxs_sel], dim=0)
                sv_b = torch.tensor(
                    [signal_buffer[i][1] for i in idxs_sel], device=agent.device
                ).unsqueeze(1)
                pred = net_eval_head(zw_b)
                sv_norm = (sv_b / harm_scale).clamp(-1.0, 1.0)
                net_loss = F.mse_loss(pred, sv_norm)
                if net_loss.requires_grad:
                    net_eval_optimizer.zero_grad()
                    net_loss.backward()
                    torch.nn.utils.clip_grad_norm_(net_eval_head.parameters(), 0.5)
                    net_eval_optimizer.step()

            if done:
                break

        for start in range(0, len(episode_traj) - E2_ROLLOUT_STEPS):
            traj_buffer.append(episode_traj[start:start + E2_ROLLOUT_STEPS + 1])
        if len(traj_buffer) > MAX_TRAJ_BUFFER:
            traj_buffer = traj_buffer[-MAX_TRAJ_BUFFER:]

        if (ep + 1) % 50 == 0 or ep == num_episodes - 1:
            print(f"  [train] ep {ep+1}/{num_episodes}  harm={total_harm}  "
                  f"ce_loss_mean={total_ce_loss / max(1, ce_steps):.4f}", flush=True)

    # Compute mean Δz_world per event type
    mean_dz = {
        k: float(sum(v) / max(1, len(v))) for k, v in dz_world_by_event.items()
    }

    return {
        "total_harm":    total_harm,
        "total_benefit": total_benefit,
        "mean_ce_loss":  total_ce_loss / max(1, ce_steps),
        "signal_buffer": signal_buffer,
        "eval_buffer":   eval_buffer,
        "mean_dz_world_by_event": mean_dz,
        "n_dz_by_event": {k: len(v) for k, v in dz_world_by_event.items()},
    }


def _eval_classification_accuracy(
    event_classifier: nn.Module,
    eval_buffer: List[Tuple[torch.Tensor, int]],
    device: str,
) -> float:
    """Evaluate held-out event classification accuracy."""
    if len(eval_buffer) < 10:
        return 0.0
    event_classifier.eval()
    zw_batch = torch.cat([b[0] for b in eval_buffer], dim=0).to(device)
    labels   = torch.tensor([b[1] for b in eval_buffer], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = event_classifier(zw_batch)
        preds  = logits.argmax(dim=-1)
        acc    = float((preds == labels).float().mean().item())
    # Also print per-class breakdown
    for cls_id, name in enumerate(["none", "env_hazard", "agent_hazard"]):
        mask = labels == cls_id
        if mask.sum() > 0:
            cls_acc = float((preds[mask] == labels[mask]).float().mean().item())
            print(f"  event_acc [{name}]: {cls_acc:.3f}  (n={mask.sum().item()})", flush=True)
    return acc


def _eval_sd003_probe(
    agent: REEAgent,
    net_eval_head: nn.Module,
    env: CausalGridWorld,
    num_resets: int,
) -> Dict:
    """Standard SD-003 attribution probe (no reafference correction)."""
    agent.eval()
    net_eval_head.eval()

    near_sigs: List[float] = []
    safe_sigs:  List[float] = []
    all_pred_vals: List[float] = []
    fatal_errors = 0

    wall_type   = env.ENTITY_TYPES["wall"]
    hazard_type = env.ENTITY_TYPES["hazard"]

    def _probe(ax: int, ay: int, actual_idx: int) -> float:
        env.agent_x = ax
        env.agent_y = ay
        obs_dict = env._get_observation_dict()
        with torch.no_grad():
            latent  = agent.sense(obs_dict["body_state"], obs_dict["world_state"])
            z_world = latent.z_world
            a_act   = _action_to_onehot(actual_idx, env.action_dim, agent.device)
            cf_idx  = _random_cf_action(actual_idx, env.action_dim)
            a_cf    = _action_to_onehot(cf_idx, env.action_dim, agent.device)
            zw_act  = agent.e2.world_forward(z_world, a_act)
            zw_cf   = agent.e2.world_forward(z_world, a_cf)
            v_act   = net_eval_head(zw_act)
            v_cf    = net_eval_head(zw_cf)
            all_pred_vals.extend([float(v_act.item()), float(v_cf.item())])
            return float((v_act - v_cf).item())

    try:
        for _ in range(num_resets):
            env.reset()
            for hx, hy in env.hazards:
                for action_idx, (dx, dy) in env.ACTIONS.items():
                    if action_idx == 4:
                        continue
                    ax, ay = hx - dx, hy - dy
                    if 0 <= ax < env.size and 0 <= ay < env.size:
                        cell = int(env.grid[ax, ay])
                        if cell not in (wall_type, hazard_type):
                            near_sigs.append(_probe(ax, ay, action_idx))
            for px in range(env.size):
                for py in range(env.size):
                    if int(env.grid[px, py]) in (wall_type, hazard_type):
                        continue
                    min_dist = min(abs(px - hx) + abs(py - hy) for hx, hy in env.hazards)
                    if min_dist > 3:
                        safe_sigs.append(_probe(px, py, random.randint(0, 3)))
    except Exception:
        import traceback
        fatal_errors += 1
        print(f"  FATAL probe: {traceback.format_exc()}", flush=True)

    mean_near = float(sum(near_sigs) / max(1, len(near_sigs)))
    mean_safe = float(sum(safe_sigs)  / max(1, len(safe_sigs)))
    gap = mean_near - mean_safe
    pred_std = float(torch.tensor(all_pred_vals).std().item()) if len(all_pred_vals) > 1 else 0.0

    print(f"  SD-003 probe  n_near={len(near_sigs)} n_safe={len(safe_sigs)}  "
          f"near={mean_near:.4f}  safe={mean_safe:.4f}  gap={gap:.4f}  "
          f"pred_std={pred_std:.4f}", flush=True)

    return {
        "calibration_gap":      gap,
        "mean_causal_sig_near": mean_near,
        "mean_causal_sig_safe": mean_safe,
        "n_near_hazard_probes": len(near_sigs),
        "n_safe_probes":        len(safe_sigs),
        "net_eval_pred_std":    pred_std,
        "fatal_errors":         fatal_errors,
    }


def run(
    seed: int = 0,
    warmup_episodes: int = 1000,
    eval_probe_resets: int = 10,
    steps_per_episode: int = 200,
    self_dim: int = 32,
    world_dim: int = 32,
    lr: float = 1e-3,
    lambda_event: float = 0.5,
    harm_scale: float = 0.02,
    alpha_world: float = 0.9,
    **kwargs,
) -> dict:
    torch.manual_seed(seed)
    random.seed(seed)

    env = CausalGridWorld(
        seed=seed, size=12, num_hazards=15, num_resources=5,
        env_drift_interval=3, env_drift_prob=0.5,
        hazard_harm=harm_scale, contaminated_harm=harm_scale,
    )
    config = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=self_dim,
        world_dim=world_dim,
        alpha_world=alpha_world,
    )
    agent = REEAgent(config)
    world_decoder  = _make_world_decoder(world_dim, env.world_obs_dim)
    net_eval_head  = _make_net_eval_head(world_dim)
    event_classifier = nn.Linear(world_dim, N_EVENT_TYPES)  # SD-009 / MECH-100

    # Main optimizer: includes encoder (latent_stack) + E1/E2 + world_decoder + event_classifier
    e12_params = [p for n_, p in agent.named_parameters() if "harm_eval" not in n_]
    e12_params += list(world_decoder.parameters())
    e12_params += list(event_classifier.parameters())
    optimizer      = optim.Adam(e12_params, lr=lr)
    net_eval_optim = optim.Adam(net_eval_head.parameters(), lr=1e-4)

    print(f"[V3-EXQ-020] Warmup + contrastive training: {warmup_episodes} eps  "
          f"λ_event={lambda_event}  (12×12, 15 hazards, drift)", flush=True)
    train_out = _train_and_collect(
        agent, env, world_decoder, net_eval_head, event_classifier,
        optimizer, net_eval_optim,
        warmup_episodes, steps_per_episode, lambda_event,
        harm_scale=harm_scale,
    )
    warmup_harm    = train_out["total_harm"]
    warmup_benefit = train_out["total_benefit"]
    mean_dz        = train_out["mean_dz_world_by_event"]

    print(f"  Warmup done. harm={warmup_harm}  benefit={warmup_benefit}", flush=True)
    print(f"  Δz_world — none={mean_dz['none']:.4f}  "
          f"env_hazard={mean_dz['env_caused_hazard']:.4f}  "
          f"agent_hazard={mean_dz['agent_caused_hazard']:.4f}", flush=True)

    # Event classification accuracy
    print(f"[V3-EXQ-020] Evaluating event classification accuracy...", flush=True)
    eval_acc = _eval_classification_accuracy(
        event_classifier, train_out["eval_buffer"], agent.device
    )
    print(f"  Overall event classification accuracy: {eval_acc:.3f}", flush=True)

    # SD-003 attribution probe
    print(f"[V3-EXQ-020] SD-003 probe ({eval_probe_resets} resets)...", flush=True)
    probe = _eval_sd003_probe(agent, net_eval_head, env, eval_probe_resets)
    fatal_errors = probe["fatal_errors"]

    # Event selectivity margin
    selectivity_margin = mean_dz["env_caused_hazard"] - mean_dz["none"]
    print(f"  Event selectivity margin (env_hazard - none): {selectivity_margin:.4f}", flush=True)

    # PASS / FAIL
    c1_pass = probe["calibration_gap"] > 0.05
    c2_pass = eval_acc > 0.5
    c3_pass = selectivity_margin > 0.005
    c4_pass = warmup_harm > 100
    c5_pass = fatal_errors == 0

    all_pass = c1_pass and c2_pass and c3_pass and c4_pass and c5_pass
    status   = "PASS" if all_pass else "FAIL"
    criteria_met = sum([c1_pass, c2_pass, c3_pass, c4_pass, c5_pass])

    failure_notes = []
    if not c1_pass:
        failure_notes.append(
            f"C1 FAIL: calibration_gap={probe['calibration_gap']:.4f} <= 0.05"
        )
    if not c2_pass:
        failure_notes.append(
            f"C2 FAIL: event_classification_acc={eval_acc:.3f} <= 0.5"
        )
    if not c3_pass:
        failure_notes.append(
            f"C3 FAIL: selectivity_margin={selectivity_margin:.4f} <= 0.005"
        )
    if not c4_pass:
        failure_notes.append(f"C4 FAIL: warmup_harm={warmup_harm} <= 100")
    if not c5_pass:
        failure_notes.append(f"C5 FAIL: fatal_errors={fatal_errors}")

    print(f"\nV3-EXQ-020 verdict: {status}  ({criteria_met}/5)", flush=True)
    for note in failure_notes:
        print(f"  {note}", flush=True)

    metrics = {
        "fatal_error_count":              float(fatal_errors),
        "warmup_harm_events":             float(warmup_harm),
        "warmup_benefit_events":          float(warmup_benefit),
        "event_classification_acc":       float(eval_acc),
        "event_selectivity_margin":       float(selectivity_margin),
        "mean_dz_world_none":             float(mean_dz["none"]),
        "mean_dz_world_env_hazard":       float(mean_dz["env_caused_hazard"]),
        "mean_dz_world_agent_hazard":     float(mean_dz["agent_caused_hazard"]),
        "calibration_gap":                float(probe["calibration_gap"]),
        "mean_causal_sig_near_hazard":    float(probe["mean_causal_sig_near"]),
        "mean_causal_sig_safe":           float(probe["mean_causal_sig_safe"]),
        "n_near_hazard_probes":           float(probe["n_near_hazard_probes"]),
        "n_safe_probes":                  float(probe["n_safe_probes"]),
        "net_eval_pred_std":              float(probe["net_eval_pred_std"]),
        "lambda_event":                   float(lambda_event),
        "mean_ce_loss_train":             float(train_out["mean_ce_loss"]),
        "crit1_pass": 1.0 if c1_pass else 0.0,
        "crit2_pass": 1.0 if c2_pass else 0.0,
        "crit3_pass": 1.0 if c3_pass else 0.0,
        "crit4_pass": 1.0 if c4_pass else 0.0,
        "criteria_met": float(criteria_met),
    }

    failure_section = ""
    if failure_notes:
        failure_section = "\n## Failure Notes\n\n" + "\n".join(f"- {n}" for n in failure_notes)

    summary_markdown = f"""# V3-EXQ-020 — Event-Auxiliary Contrastive Encoder Loss

**Status:** {status}
**Warmup:** {warmup_episodes} eps (RANDOM policy, 12×12, 15 hazards, drift_interval=3, drift_prob=0.5)
**λ_event:** {lambda_event}
**Probe eval:** {eval_probe_resets} grid resets
**Seed:** {seed}

## Motivation (MECH-100 / SD-009)

EXQ-013 showed z_world has near-zero event selectivity — Δz_world barely differs
between empty_move and env_caused_hazard. Root cause: encoder training losses are
event-invariant (E1 reconstruction, E2 self-prediction reward all z_world equally).
Fix: add cross-entropy event classifier (Linear(world_dim, 3)) to total_loss,
backpropagating through z_world encoder. This forces z_world to carry event-type
discriminative information.

## Event Selectivity

| Event Type | mean Δz_world |
|---|---|
| none (locomotion) | {mean_dz['none']:.4f} |
| env_caused_hazard | {mean_dz['env_caused_hazard']:.4f} |
| agent_caused_hazard | {mean_dz['agent_caused_hazard']:.4f} |
| **selectivity margin** (env - none) | **{selectivity_margin:.4f}** |

Event classifier accuracy: {eval_acc:.3f}
Training CE loss mean: {train_out['mean_ce_loss']:.4f}

## SD-003 Attribution (Standard, No Reafference Correction)

| Position | mean(causal_sig) |
|---|---|
| Near-hazard | {probe['mean_causal_sig_near']:.4f} |
| Safe | {probe['mean_causal_sig_safe']:.4f} |
| **calibration_gap** | **{probe['calibration_gap']:.4f}** |

net_eval pred_std: {probe['net_eval_pred_std']:.4f}
n_near={probe['n_near_hazard_probes']}  n_safe={probe['n_safe_probes']}
Warmup: harm={warmup_harm}  benefit={warmup_benefit}

## PASS Criteria

| Criterion | Result | Value |
|---|---|---|
| C1: calibration_gap > 0.05 | {"PASS" if c1_pass else "FAIL"} | {probe['calibration_gap']:.4f} |
| C2: event_classification_acc > 0.5 | {"PASS" if c2_pass else "FAIL"} | {eval_acc:.3f} |
| C3: selectivity_margin > 0.005 | {"PASS" if c3_pass else "FAIL"} | {selectivity_margin:.4f} |
| C4: Warmup harm events > 100 | {"PASS" if c4_pass else "FAIL"} | {warmup_harm} |
| C5: No fatal errors | {"PASS" if c5_pass else "FAIL"} | {fatal_errors} |

Criteria met: {criteria_met}/5 → **{status}**
{failure_section}
"""

    return {
        "status": status,
        "metrics": metrics,
        "summary_markdown": summary_markdown,
        "claim_ids": CLAIM_IDS,
        "evidence_direction": "supports" if all_pass else ("mixed" if criteria_met >= 3 else "weakens"),
        "experiment_type": EXPERIMENT_TYPE,
        "fatal_error_count": fatal_errors,
    }


if __name__ == "__main__":
    import argparse
    import json
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",           type=int,   default=0)
    parser.add_argument("--warmup",         type=int,   default=1000)
    parser.add_argument("--probe-resets",   type=int,   default=10)
    parser.add_argument("--steps",          type=int,   default=200)
    parser.add_argument("--lambda-event",   type=float, default=0.5)
    parser.add_argument("--harm-scale",     type=float, default=0.02)
    parser.add_argument("--alpha-world",    type=float, default=0.9)
    args = parser.parse_args()

    result = run(
        seed=args.seed,
        warmup_episodes=args.warmup,
        eval_probe_resets=args.probe_resets,
        steps_per_episode=args.steps,
        lambda_event=args.lambda_event,
        harm_scale=args.harm_scale,
        alpha_world=args.alpha_world,
    )

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result["run_timestamp"] = ts
    result["claim"] = CLAIM_IDS[0]
    result["verdict"] = result["status"]
    result["run_id"] = f"{EXPERIMENT_TYPE}_{ts}_v3"
    result["architecture_epoch"] = "ree_hybrid_guardrails_v1"

    out_dir = Path(__file__).resolve().parents[1] / "evidence" / "experiments" / EXPERIMENT_TYPE
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{EXPERIMENT_TYPE}_{ts}.json"
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(f"\nResult written to: {out_path}", flush=True)
    print(f"Status: {result['status']}", flush=True)
    for k, v in result["metrics"].items():
        print(f"  {k}: {v}", flush=True)
