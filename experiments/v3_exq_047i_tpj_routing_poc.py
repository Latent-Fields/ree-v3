#!/opt/local/bin/python3
"""
V3-EXQ-047i -- MECH-095 Supervised TPJ Routing Proof-of-Concept

Claims: MECH-095, SD-005

EXQ-089 FAIL analysis (2026-03-23):
  The unsupervised TPJ comparator (E2 mismatch) shows nearly identical mismatch
  for self-caused vs world-caused events (gap=0.004, threshold 0.02). E2 at 300
  episodes is not yet differentiated enough -- health drops at contact are swamped
  by position changes in the z_self vector. The agency signal saturates near 0.74
  for both event types, making classification trivial (always predict self-caused).

  EXQ-047h showed contact_dissociation=0.000 -- both z_self and z_world learn
  identical contact information because both receive the full observation.

This experiment tests whether EXPLICIT SUPERVISED ROUTING produces the contact
dissociation that unsupervised E2 mismatch cannot yet achieve.

Mechanism under test (MECH-095):
  At world-caused events (env_caused_hazard, hazard_approach), z_self should
  remain stable -- the agent's body state did not change due to ITS OWN MOTOR
  COMMAND. At self-caused events (none -- locomotion), z_world should remain
  stable -- the agent merely moved through the world without causing world-change.

Supervised routing loss:
  At world-caused events: L_route += lambda_r * MSE(z_self_t, z_self_{t-1})
    (z_self shouldn't have changed due to a world-caused event)
  At pure navigation events: L_route += lambda_r * MSE(z_world_t, z_world_{t-1})
    (z_world shouldn't have changed due to pure locomotion)

This uses ground-truth transition_type labels from the environment to supervise
routing -- what the real MECH-095 substrate would learn to do via E2 efference copy.

If this PoC shows contact_dissociation > 0.05 with supervised routing vs 0.000
without, it validates:
  (a) The routing principle is correct (MECH-095 mechanism is sound)
  (b) More E2 training or auxiliary supervision is needed for unsupervised routing
  (c) The architectural split (SD-005) CAN produce dissociation given correct routing

Discriminative pair:
  ROUTED   -- routing loss active (lambda_route=0.1, supervised by transition_type)
  BASELINE -- no routing loss (replicates EXQ-047h conditions)

Both conditions use SPLIT mode (SD-005) to isolate the routing effect.

PASS criteria (ALL required):
  C1: contact_dissociation_routed > 0.05
      (world-caused events now preferentially encoded in z_world)
  C2: contact_dissociation_routed > contact_dissociation_baseline + 0.04
      (routing improves over baseline -- validates the mechanism)
  C3: action_dissociation_routed > 0.05
      (motor events still preferentially encoded in z_self)
  C4: n_contact_min >= 20
  C5: no fatal errors

Decision:
  retain_ree (MECH-095 validated):   C1+C2+C3 pass
  hybridize (partial routing signal): C1+C2 pass but C3 marginal
  retire_ree_claim:                   C1 OR C2 fail
"""

import sys
import random
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE = "v3_exq_047i_tpj_routing_poc"
CLAIM_IDS = ["MECH-095", "SD-005"]

# Transition types considered world-caused (routing: z_self should be stable)
WORLD_CAUSED = frozenset({"env_caused_hazard", "hazard_approach"})
# Transition types considered self-caused pure locomotion (z_world should be stable)
SELF_CAUSED_LOCO = frozenset({"none"})


def _action_to_onehot(action_idx: int, num_actions: int, device) -> torch.Tensor:
    v = torch.zeros(1, num_actions, device=device)
    v[0, action_idx] = 1.0
    return v


def _fit_linear_probe(
    X: torch.Tensor,
    y: torch.Tensor,
    n_classes: int,
    n_steps: int = 400,
    lr: float = 1e-2,
) -> float:
    X = X.detach().float()
    y = y.detach().long()
    probe = nn.Linear(X.shape[1], n_classes)
    opt = optim.Adam(probe.parameters(), lr=lr)
    for _ in range(n_steps):
        logits = probe(X)
        loss = F.cross_entropy(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        preds = probe(X).argmax(dim=1)
        acc = float((preds == y).float().mean().item())
    return acc


def _run_single(
    seed: int,
    use_routing: bool,
    warmup_episodes: int,
    probe_episodes: int,
    steps_per_episode: int,
    self_dim: int,
    world_dim: int,
    lr: float,
    alpha_world: float,
    alpha_self: float,
    harm_scale: float,
    proximity_harm_scale: float,
    lambda_route: float,
) -> Dict:
    """Run one (seed, routing condition) cell."""
    torch.manual_seed(seed)
    random.seed(seed)

    cond_label = "ROUTED" if use_routing else "BASELINE"

    env = CausalGridWorldV2(
        seed=seed,
        size=12,
        num_hazards=4,
        num_resources=5,
        hazard_harm=harm_scale,
        env_drift_interval=5,
        env_drift_prob=0.1,
        proximity_harm_scale=proximity_harm_scale,
        proximity_benefit_scale=proximity_harm_scale * 0.6,
        proximity_approach_threshold=0.15,
        hazard_field_decay=0.5,
    )

    config = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=self_dim,
        world_dim=world_dim,
        alpha_world=alpha_world,
        alpha_self=alpha_self,
        reafference_action_dim=0,
    )
    # Always use SPLIT mode (SD-005) -- isolate routing effect
    config.latent.unified_latent_mode = False

    agent = REEAgent(config)

    harm_buf_pos: List[torch.Tensor] = []
    harm_buf_neg: List[torch.Tensor] = []
    MAX_BUF = 2000

    standard_params = [
        p for n, p in agent.named_parameters()
        if "harm_eval_head" not in n
    ]
    harm_eval_params = list(agent.e3.harm_eval_head.parameters())
    optimizer = optim.Adam(standard_params, lr=lr)
    harm_eval_optimizer = optim.Adam(harm_eval_params, lr=1e-4)

    counts: Dict[str, int] = {
        "hazard_approach": 0, "env_caused_hazard": 0,
        "agent_caused_hazard": 0, "none": 0,
    }
    total_routing_loss = 0.0
    routing_steps = 0

    # --- WARMUP TRAINING ---
    agent.train()

    for ep in range(warmup_episodes):
        _, obs_dict = env.reset()
        agent.reset()

        # Routing anchors: detached latents from previous step
        z_self_prev: torch.Tensor = None
        z_world_prev: torch.Tensor = None
        prev_ttype: str = None

        for _ in range(steps_per_episode):
            obs_body = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            latent = agent.sense(obs_body, obs_world)
            agent.clock.advance()

            z_self_curr = latent.z_self    # [1, self_dim], has grad
            z_world_curr = latent.z_world  # [1, world_dim], has grad

            # --- Routing loss (applied BEFORE action, using prev transition) ---
            routing_loss = torch.zeros(1)
            if use_routing and z_self_prev is not None and prev_ttype is not None:
                if prev_ttype in WORLD_CAUSED:
                    # World caused the last transition: z_self should not have changed
                    routing_loss = lambda_route * F.mse_loss(
                        z_self_curr, z_self_prev
                    )
                elif prev_ttype in SELF_CAUSED_LOCO:
                    # Pure locomotion: z_world should be stable (agent just moved)
                    routing_loss = lambda_route * F.mse_loss(
                        z_world_curr, z_world_prev
                    )
                total_routing_loss += routing_loss.item()
                routing_steps += 1

            # Save detached anchors for next iteration
            z_self_prev = z_self_curr.detach()
            z_world_prev = z_world_curr.detach()

            z_world_track = z_world_curr.detach()

            # Take action
            action_idx = random.randint(0, env.action_dim - 1)
            action = _action_to_onehot(action_idx, env.action_dim, agent.device)
            agent._last_action = action

            _, harm_signal, done, info, obs_dict = env.step(action)
            prev_ttype = info.get("transition_type", "none")
            ttype = prev_ttype
            if ttype in counts:
                counts[ttype] += 1

            is_harm = float(harm_signal) < 0
            if is_harm:
                harm_buf_pos.append(z_world_track)
                if len(harm_buf_pos) > MAX_BUF:
                    harm_buf_pos = harm_buf_pos[-MAX_BUF:]
            else:
                harm_buf_neg.append(z_world_track)
                if len(harm_buf_neg) > MAX_BUF:
                    harm_buf_neg = harm_buf_neg[-MAX_BUF:]

            e1_loss = agent.compute_prediction_loss()
            e2_loss = agent.compute_e2_loss()
            total_loss = e1_loss + e2_loss + routing_loss
            if total_loss.requires_grad:
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                optimizer.step()

            if len(harm_buf_pos) >= 4 and len(harm_buf_neg) >= 4:
                k_pos = min(16, len(harm_buf_pos))
                k_neg = min(16, len(harm_buf_neg))
                pos_idx = torch.randperm(len(harm_buf_pos))[:k_pos].tolist()
                neg_idx = torch.randperm(len(harm_buf_neg))[:k_neg].tolist()
                zw_pos = torch.cat([harm_buf_pos[i] for i in pos_idx], dim=0)
                zw_neg = torch.cat([harm_buf_neg[i] for i in neg_idx], dim=0)
                zw_b = torch.cat([zw_pos, zw_neg], dim=0)
                target = torch.cat([
                    torch.ones(k_pos, 1, device=agent.device),
                    torch.zeros(k_neg, 1, device=agent.device),
                ], dim=0)
                pred_harm = agent.e3.harm_eval(zw_b)
                harm_loss = F.mse_loss(pred_harm, target)
                if harm_loss.requires_grad:
                    harm_eval_optimizer.zero_grad()
                    harm_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        agent.e3.harm_eval_head.parameters(), 0.5,
                    )
                    harm_eval_optimizer.step()

            if done:
                break

        if (ep + 1) % 100 == 0 or ep == warmup_episodes - 1:
            mean_route = (
                total_routing_loss / max(1, routing_steps) if use_routing else 0.0
            )
            print(
                f"  [train] seed={seed} cond={cond_label}"
                f" ep {ep+1}/{warmup_episodes}"
                f" approach={counts['hazard_approach']}"
                f" contact={counts['env_caused_hazard']+counts['agent_caused_hazard']}"
                f" route_loss={mean_route:.4f}",
                flush=True,
            )

    # --- PROBE COLLECTION ---
    agent.eval()

    probe_self: List[torch.Tensor] = []
    probe_world: List[torch.Tensor] = []
    probe_actions: List[int] = []
    probe_contact_labels: List[int] = []
    n_fatal = 0

    for _ in range(probe_episodes):
        _, obs_dict = env.reset()
        agent.reset()

        for _ in range(steps_per_episode):
            obs_body = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]

            with torch.no_grad():
                latent = agent.sense(obs_body, obs_world)
                agent.clock.advance()

            action_idx = random.randint(0, env.action_dim - 1)
            action = _action_to_onehot(action_idx, env.action_dim, agent.device)
            agent._last_action = action

            _, _, done, info, obs_dict = env.step(action)
            ttype = info.get("transition_type", "none")

            try:
                with torch.no_grad():
                    obs_body_next = obs_dict["body_state"]
                    obs_world_next = obs_dict["world_state"]
                    latent_next = agent.sense(obs_body_next, obs_world_next)
                    agent.clock.advance()

                    probe_self.append(latent_next.z_self.detach())
                    probe_world.append(latent_next.z_world.detach())
                    probe_actions.append(action_idx)
                    is_contact = int(ttype in ("hazard_approach", "agent_caused_hazard",
                                               "env_caused_hazard"))
                    probe_contact_labels.append(is_contact)

                    _, _, done2, _, obs_dict = env.step(
                        _action_to_onehot(
                            random.randint(0, env.action_dim - 1),
                            env.action_dim, agent.device,
                        )
                    )
                    if done2:
                        break
            except Exception:
                n_fatal += 1

            if done:
                break

    # --- FIT PROBES ---
    action_acc_self = 0.0
    action_acc_world = 0.0
    contact_acc_self = 0.0
    contact_acc_world = 0.0
    n_contact_probe = int(sum(probe_contact_labels))

    if len(probe_self) >= 20:
        X_self  = torch.cat(probe_self,  dim=0).float()
        X_world = torch.cat(probe_world, dim=0).float()
        y_action  = torch.tensor(probe_actions,       dtype=torch.long)
        y_contact = torch.tensor(probe_contact_labels, dtype=torch.long)

        action_acc_self  = _fit_linear_probe(X_self,  y_action,  env.action_dim)
        action_acc_world = _fit_linear_probe(X_world, y_action,  env.action_dim)

        if n_contact_probe >= 5:
            contact_acc_self  = _fit_linear_probe(X_self,  y_contact, 2)
            contact_acc_world = _fit_linear_probe(X_world, y_contact, 2)

    action_dissociation  = action_acc_self  - action_acc_world
    contact_dissociation = contact_acc_world - contact_acc_self

    mean_route_loss = total_routing_loss / max(1, routing_steps) if use_routing else 0.0

    print(
        f"  [probe] seed={seed} cond={cond_label}"
        f" action: self={action_acc_self:.3f} world={action_acc_world:.3f}"
        f" dissoc={action_dissociation:+.3f}"
        f"  contact: world={contact_acc_world:.3f} self={contact_acc_self:.3f}"
        f" dissoc={contact_dissociation:+.3f}"
        f" n={n_contact_probe}",
        flush=True,
    )

    return {
        "seed": seed,
        "condition": cond_label,
        "use_routing": use_routing,
        "action_acc_self":       float(action_acc_self),
        "action_acc_world":      float(action_acc_world),
        "action_dissociation":   float(action_dissociation),
        "contact_acc_self":      float(contact_acc_self),
        "contact_acc_world":     float(contact_acc_world),
        "contact_dissociation":  float(contact_dissociation),
        "n_contact_probe":       int(n_contact_probe),
        "n_probe_steps":         int(len(probe_self)),
        "mean_routing_loss":     float(mean_route_loss),
        "train_approach_events": int(counts["hazard_approach"]),
        "train_contact_events":  int(
            counts["env_caused_hazard"] + counts["agent_caused_hazard"]
        ),
        "n_fatal": int(n_fatal),
    }


def run(
    seeds: Tuple = (42, 7),
    warmup_episodes: int = 400,
    probe_episodes: int = 20,
    steps_per_episode: int = 200,
    self_dim: int = 32,
    world_dim: int = 32,
    lr: float = 1e-3,
    alpha_world: float = 0.9,
    alpha_self: float = 0.3,
    harm_scale: float = 0.02,
    proximity_harm_scale: float = 0.05,
    lambda_route: float = 0.1,
    **kwargs,
) -> dict:
    """ROUTED vs BASELINE (both SPLIT) on functional separation probes."""
    results_routed:   List[Dict] = []
    results_baseline: List[Dict] = []

    for seed in seeds:
        for use_routing in [True, False]:
            label = "ROUTED" if use_routing else "BASELINE"
            print(
                f"\n[V3-EXQ-047i] {label} seed={seed}"
                f" warmup={warmup_episodes} lambda_route={lambda_route}"
                f" alpha_world={alpha_world}",
                flush=True,
            )
            r = _run_single(
                seed=seed,
                use_routing=use_routing,
                warmup_episodes=warmup_episodes,
                probe_episodes=probe_episodes,
                steps_per_episode=steps_per_episode,
                self_dim=self_dim,
                world_dim=world_dim,
                lr=lr,
                alpha_world=alpha_world,
                alpha_self=alpha_self,
                harm_scale=harm_scale,
                proximity_harm_scale=proximity_harm_scale,
                lambda_route=lambda_route,
            )
            if use_routing:
                results_routed.append(r)
            else:
                results_baseline.append(r)

    def _avg(results: List[Dict], key: str) -> float:
        vals = [r[key] for r in results]
        return float(sum(vals) / max(1, len(vals)))

    contact_dissoc_routed   = _avg(results_routed,   "contact_dissociation")
    contact_dissoc_baseline = _avg(results_baseline, "contact_dissociation")
    action_dissoc_routed    = _avg(results_routed,   "action_dissociation")
    action_dissoc_baseline  = _avg(results_baseline, "action_dissociation")
    n_contact_min = min(
        r["n_contact_probe"] for r in results_routed + results_baseline
    )

    c1_pass = contact_dissoc_routed   > 0.05
    c2_pass = (contact_dissoc_routed - contact_dissoc_baseline) > 0.04
    c3_pass = action_dissoc_routed    > 0.05
    c4_pass = n_contact_min >= 20
    c5_pass = all(r["n_fatal"] == 0 for r in results_routed + results_baseline)

    all_pass = c1_pass and c2_pass and c3_pass and c4_pass and c5_pass
    criteria_met = sum([c1_pass, c2_pass, c3_pass, c4_pass, c5_pass])
    status = "PASS" if all_pass else "FAIL"

    if all_pass:
        decision = "retain_ree"
    elif c1_pass and c2_pass:
        decision = "hybridize"
    else:
        decision = "retire_ree_claim"

    print(f"\n[V3-EXQ-047i] Final results:", flush=True)
    print(
        f"  contact dissoc: routed={contact_dissoc_routed:+.3f}"
        f"  baseline={contact_dissoc_baseline:+.3f}"
        f"  improvement={contact_dissoc_routed - contact_dissoc_baseline:+.3f}",
        flush=True,
    )
    print(
        f"  action dissoc:  routed={action_dissoc_routed:+.3f}"
        f"  baseline={action_dissoc_baseline:+.3f}",
        flush=True,
    )
    print(
        f"  decision={decision}  status={status} ({criteria_met}/5)",
        flush=True,
    )

    failure_notes: List[str] = []
    if not c1_pass:
        failure_notes.append(
            f"C1 FAIL: contact_dissoc_routed={contact_dissoc_routed:+.3f} <= 0.05"
        )
    if not c2_pass:
        gap = contact_dissoc_routed - contact_dissoc_baseline
        failure_notes.append(
            f"C2 FAIL: routing improvement={gap:+.3f} <= 0.04"
            f" (routed={contact_dissoc_routed:+.3f}"
            f" vs baseline={contact_dissoc_baseline:+.3f})"
        )
    if not c3_pass:
        failure_notes.append(
            f"C3 FAIL: action_dissoc_routed={action_dissoc_routed:+.3f} <= 0.05"
        )
    if not c4_pass:
        failure_notes.append(f"C4 FAIL: n_contact_min={n_contact_min} < 20")
    if not c5_pass:
        failure_notes.append("C5 FAIL: fatal errors occurred")

    for note in failure_notes:
        print(f"  {note}", flush=True)

    per_routed_rows = "\n".join(
        f"  seed={r['seed']}: contact_world={r['contact_acc_world']:.3f}"
        f" contact_self={r['contact_acc_self']:.3f}"
        f" contact_dissoc={r['contact_dissociation']:+.3f}"
        f" action_dissoc={r['action_dissociation']:+.3f}"
        f" route_loss={r['mean_routing_loss']:.4f}"
        for r in results_routed
    )
    per_baseline_rows = "\n".join(
        f"  seed={r['seed']}: contact_world={r['contact_acc_world']:.3f}"
        f" contact_self={r['contact_acc_self']:.3f}"
        f" contact_dissoc={r['contact_dissociation']:+.3f}"
        f" action_dissoc={r['action_dissociation']:+.3f}"
        for r in results_baseline
    )

    failure_section = ""
    if failure_notes:
        failure_section = "\n## Failure Notes\n\n" + "\n".join(
            f"- {n}" for n in failure_notes
        )

    summary_markdown = (
        f"# V3-EXQ-047i -- MECH-095 Supervised TPJ Routing PoC\n\n"
        f"**Status:** {status}\n"
        f"**Claims:** MECH-095, SD-005\n"
        f"**Decision:** {decision}\n"
        f"**Seeds:** {list(seeds)}\n"
        f"**alpha_world:** {alpha_world}  (SD-008)\n"
        f"**lambda_route:** {lambda_route}\n"
        f"**Warmup:** {warmup_episodes} eps  **Probe:** {probe_episodes} eps\n"
        f"**Design note:** Supervised routing via transition_type labels."
        f" EXQ-089 showed unsupervised E2 mismatch insufficient at 300 eps"
        f" (gap=0.004, false_attribution=1.0). This PoC tests the routing"
        f" PRINCIPLE using ground-truth labels.\n\n"
        f"## Pre-Registered Thresholds\n\n"
        f"C1: contact_dissociation_routed > 0.05\n"
        f"C2: routing improvement (routed - baseline) > 0.04\n"
        f"C3: action_dissociation_routed > 0.05\n"
        f"C4: n_contact_probe >= 20\n"
        f"C5: no fatal errors\n\n"
        f"## Results\n\n"
        f"| Condition | contact_dissoc | action_dissoc |\n"
        f"|-----------|---------------|---------------|\n"
        f"| ROUTED    | {contact_dissoc_routed:+.3f}          | {action_dissoc_routed:+.3f}         |\n"
        f"| BASELINE  | {contact_dissoc_baseline:+.3f}          | {action_dissoc_baseline:+.3f}         |\n\n"
        f"**Routing improvement: {contact_dissoc_routed - contact_dissoc_baseline:+.3f}**\n\n"
        f"## PASS Criteria\n\n"
        f"| Criterion | Result | Value |\n"
        f"|---|---|---|\n"
        f"| C1: contact_dissoc_routed > 0.05 | {'PASS' if c1_pass else 'FAIL'}"
        f" | {contact_dissoc_routed:+.3f} |\n"
        f"| C2: improvement > 0.04 | {'PASS' if c2_pass else 'FAIL'}"
        f" | {contact_dissoc_routed - contact_dissoc_baseline:+.3f} |\n"
        f"| C3: action_dissoc_routed > 0.05 | {'PASS' if c3_pass else 'FAIL'}"
        f" | {action_dissoc_routed:+.3f} |\n"
        f"| C4: n_contact_min >= 20 | {'PASS' if c4_pass else 'FAIL'}"
        f" | {n_contact_min} |\n"
        f"| C5: no fatal errors | {'PASS' if c5_pass else 'FAIL'} | -- |\n\n"
        f"Criteria met: {criteria_met}/5 -> **{status}**\n\n"
        f"## Per-Seed\n\n"
        f"ROUTED:\n{per_routed_rows}\n\n"
        f"BASELINE:\n{per_baseline_rows}\n"
        f"{failure_section}\n"
    )

    metrics = {
        "contact_dissoc_routed":   float(contact_dissoc_routed),
        "contact_dissoc_baseline": float(contact_dissoc_baseline),
        "routing_improvement":     float(contact_dissoc_routed - contact_dissoc_baseline),
        "action_dissoc_routed":    float(action_dissoc_routed),
        "action_dissoc_baseline":  float(action_dissoc_baseline),
        "n_contact_min":           float(n_contact_min),
        "lambda_route":            float(lambda_route),
        "n_seeds":                 float(len(seeds)),
        "alpha_world":             float(alpha_world),
        "crit1_pass": 1.0 if c1_pass else 0.0,
        "crit2_pass": 1.0 if c2_pass else 0.0,
        "crit3_pass": 1.0 if c3_pass else 0.0,
        "crit4_pass": 1.0 if c4_pass else 0.0,
        "crit5_pass": 1.0 if c5_pass else 0.0,
        "criteria_met": float(criteria_met),
    }

    return {
        "status": status,
        "metrics": metrics,
        "summary_markdown": summary_markdown,
        "claim_ids": CLAIM_IDS,
        "evidence_direction": (
            "supports" if all_pass
            else ("mixed" if criteria_met >= 3 else "weakens")
        ),
        "experiment_type": EXPERIMENT_TYPE,
        "fatal_error_count": sum(
            r["n_fatal"] for r in results_routed + results_baseline
        ),
    }


if __name__ == "__main__":
    import argparse
    import json
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds",          type=int,   nargs="+", default=[42, 7])
    parser.add_argument("--warmup",         type=int,   default=400)
    parser.add_argument("--probe-eps",      type=int,   default=20)
    parser.add_argument("--steps",          type=int,   default=200)
    parser.add_argument("--alpha-world",    type=float, default=0.9)
    parser.add_argument("--alpha-self",     type=float, default=0.3)
    parser.add_argument("--harm-scale",     type=float, default=0.02)
    parser.add_argument("--proximity-scale",type=float, default=0.05)
    parser.add_argument("--lambda-route",   type=float, default=0.1)
    args = parser.parse_args()

    result = run(
        seeds=tuple(args.seeds),
        warmup_episodes=args.warmup,
        probe_episodes=args.probe_eps,
        steps_per_episode=args.steps,
        alpha_world=args.alpha_world,
        alpha_self=args.alpha_self,
        harm_scale=args.harm_scale,
        proximity_harm_scale=args.proximity_scale,
        lambda_route=args.lambda_route,
    )

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result["run_timestamp"] = ts
    result["claim"] = CLAIM_IDS[0]
    result["verdict"] = result["status"]
    result["run_id"] = f"{EXPERIMENT_TYPE}_{ts}_v3"
    result["architecture_epoch"] = "ree_hybrid_guardrails_v1"

    out_dir = (
        Path(__file__).resolve().parents[2]
        / "REE_assembly" / "evidence" / "experiments" / EXPERIMENT_TYPE
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{EXPERIMENT_TYPE}_{ts}.json"
    out_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(f"\nResult written to: {out_path}", flush=True)
    print(f"Status: {result['status']}", flush=True)
    for k, v in result["metrics"].items():
        print(f"  {k}: {v}", flush=True)
