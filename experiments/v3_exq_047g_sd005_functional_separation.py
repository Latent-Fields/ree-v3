#!/opt/local/bin/python3
"""
V3-EXQ-047g -- SD-005 Functional Separation Probe (Correct Test)

Claims: SD-005

Prior EXQ-047e/f used orthogonality constraints and geometric probes.
These tested the WRONG property. Functional specialisation via differential
loss routing does NOT require geometric orthogonality -- there are
legitimate shared latent dimensions (e.g., top-down z_beta conditioning,
blurry body-world boundary near hazard approach). Orthogonality tests fail
precisely in the zones where overlap is biologically expected.

The CORRECT test is functional cross-decoding superiority:

  - z_self should be more decodable for MOTOR content (action taken)
    than z_world. E2 motor-sensory loss routes motor gradient through z_self.
  - z_world should be more decodable for WORLD content (hazard contact)
    than z_self. E3 harm loss routes world gradient through z_world.

Both specialisations emerge from differential loss routing alone -- no
orthogonality penalty needed. With sufficient training, the split should
produce reliably better per-channel probing accuracy.

Discriminative pair (SPLIT vs UNIFIED):
  SPLIT   -- unified_latent_mode=False (default SD-005 design)
  UNIFIED -- unified_latent_mode=True  (ablation: z_self = z_world = avg)

Both conditions use identical differential loss routing. The question is
whether the architectural separation (separate encoders + separate loss
routing) amplifies the functional distinction relative to a fused baseline.

Probe methodology:
  After 400 warmup episodes, collect 20 probe episodes of latent activations
  (z_self_t, z_world_t) labeled by:
    - action_taken: the random action chosen at step t (motor label)
    - is_contact: whether step t was a hazard contact/approach (world label)
  Fit linear classifiers (logistic regression via SGD) on each channel.
  Compare: z_self vs z_world for each label type.

PASS criteria (ALL required):
  C1: action_acc_self_split > action_acc_world_split + 0.10
      (z_self decodes action 10pp better than z_world in split condition)
  C2: contact_acc_world_split > contact_acc_self_split + 0.05
      (z_world decodes contact 5pp better than z_self in split condition)
  C3: action_acc_self_split > action_acc_self_unified + 0.03
      (split z_self better motor probe than unified z_self)
  C4: contact_acc_world_split > contact_acc_world_unified + 0.03
      (split z_world better world probe than unified z_world)
  C5: n_contact_probe_min >= 20 (sufficient contact events in probe phase)

Decision scoring:
  retain_ree:        all C1-C4 pass
  hybridize:         C1+C2 pass but C3 or C4 marginal
  retire_ree_claim:  C1 OR C2 fail
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


EXPERIMENT_TYPE = "v3_exq_047g_sd005_functional_separation"
CLAIM_IDS = ["SD-005"]


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
    """
    Fit a linear probe (logistic regression) on X -> y and return accuracy.
    X: [N, dim], y: [N] integer class labels.
    """
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
    unified: bool,
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
) -> Dict:
    """Run one (seed, condition) cell and return functional probe accuracy metrics."""
    torch.manual_seed(seed)
    random.seed(seed)

    cond_label = "UNIFIED" if unified else "SPLIT"

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
    config.latent.unified_latent_mode = unified

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

    # --- WARMUP TRAINING ---
    agent.train()

    for ep in range(warmup_episodes):
        _, obs_dict = env.reset()
        agent.reset()

        for _ in range(steps_per_episode):
            obs_body = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            latent = agent.sense(obs_body, obs_world)
            agent.clock.advance()

            z_world_curr = latent.z_world.detach()

            action_idx = random.randint(0, env.action_dim - 1)
            action = _action_to_onehot(action_idx, env.action_dim, agent.device)
            agent._last_action = action

            _, harm_signal, done, info, obs_dict = env.step(action)
            ttype = info.get("transition_type", "none")
            if ttype in counts:
                counts[ttype] += 1

            is_harm = float(harm_signal) < 0
            if is_harm:
                harm_buf_pos.append(z_world_curr)
                if len(harm_buf_pos) > MAX_BUF:
                    harm_buf_pos = harm_buf_pos[-MAX_BUF:]
            else:
                harm_buf_neg.append(z_world_curr)
                if len(harm_buf_neg) > MAX_BUF:
                    harm_buf_neg = harm_buf_neg[-MAX_BUF:]

            e1_loss = agent.compute_prediction_loss()
            e2_loss = agent.compute_e2_loss()
            total_loss = e1_loss + e2_loss
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
            print(
                f"  [train] seed={seed} cond={cond_label}"
                f" ep {ep+1}/{warmup_episodes}"
                f" approach={counts['hazard_approach']}"
                f" contact={counts['env_caused_hazard']+counts['agent_caused_hazard']}",
                flush=True,
            )

    # --- PROBE COLLECTION ---
    # Collect (z_self, z_world, action_idx, is_contact) for linear probe fitting.
    # Use steps AFTER taking the action so z_self encodes motor consequence.
    agent.eval()

    probe_self: List[torch.Tensor] = []
    probe_world: List[torch.Tensor] = []
    probe_actions: List[int] = []
    probe_contact_labels: List[int] = []  # 1 if hazard contact/approach, 0 otherwise
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

            # Collect state AFTER action for action-prediction probe
            try:
                with torch.no_grad():
                    obs_body_next = obs_dict["body_state"]
                    obs_world_next = obs_dict["world_state"]
                    latent_next = agent.sense(obs_body_next, obs_world_next)
                    agent.clock.advance()

                    probe_self.append(latent_next.z_self.detach())
                    probe_world.append(latent_next.z_world.detach())
                    probe_actions.append(action_idx)
                    # Contact label: approach or contact = 1
                    is_contact = int(ttype in ("hazard_approach", "agent_caused_hazard",
                                               "env_caused_hazard"))
                    probe_contact_labels.append(is_contact)

                    _, _, done2, info2, obs_dict = env.step(
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

    # --- FIT LINEAR PROBES ---
    action_acc_self = 0.0
    action_acc_world = 0.0
    contact_acc_self = 0.0
    contact_acc_world = 0.0
    n_contact_probe = int(sum(probe_contact_labels))

    if len(probe_self) >= 20:
        X_self  = torch.cat(probe_self,  dim=0).float()
        X_world = torch.cat(probe_world, dim=0).float()
        y_action  = torch.tensor(probe_actions, dtype=torch.long)
        y_contact = torch.tensor(probe_contact_labels, dtype=torch.long)

        n_actions = env.action_dim
        action_acc_self  = _fit_linear_probe(X_self,  y_action,  n_actions)
        action_acc_world = _fit_linear_probe(X_world, y_action,  n_actions)

        if n_contact_probe >= 5:
            contact_acc_self  = _fit_linear_probe(X_self,  y_contact, 2)
            contact_acc_world = _fit_linear_probe(X_world, y_contact, 2)

    action_dissociation  = action_acc_self  - action_acc_world
    contact_dissociation = contact_acc_world - contact_acc_self

    print(
        f"  [probe] seed={seed} cond={cond_label}"
        f" action: self={action_acc_self:.3f} world={action_acc_world:.3f}"
        f" dissoc={action_dissociation:+.3f}",
        flush=True,
    )
    print(
        f"  [probe] seed={seed} cond={cond_label}"
        f" contact: world={contact_acc_world:.3f} self={contact_acc_self:.3f}"
        f" dissoc={contact_dissociation:+.3f}"
        f" n_contact={n_contact_probe}",
        flush=True,
    )

    return {
        "seed": seed,
        "condition": cond_label,
        "unified": unified,
        "action_acc_self": float(action_acc_self),
        "action_acc_world": float(action_acc_world),
        "action_dissociation": float(action_dissociation),
        "contact_acc_self": float(contact_acc_self),
        "contact_acc_world": float(contact_acc_world),
        "contact_dissociation": float(contact_dissociation),
        "n_contact_probe": int(n_contact_probe),
        "n_probe_steps": int(len(probe_self)),
        "train_approach_events": int(counts["hazard_approach"]),
        "train_contact_events": int(
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
    **kwargs,
) -> dict:
    """Discriminative pair: SPLIT (SD-005) vs UNIFIED on functional separation probes."""
    results_split: List[Dict] = []
    results_unified: List[Dict] = []

    for seed in seeds:
        for unified in [False, True]:
            label = "UNIFIED" if unified else "SPLIT"
            print(
                f"\n[V3-EXQ-047g] {label} seed={seed}"
                f" warmup={warmup_episodes} probe={probe_episodes}"
                f" alpha_world={alpha_world}",
                flush=True,
            )
            r = _run_single(
                seed=seed,
                unified=unified,
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
            )
            if unified:
                results_unified.append(r)
            else:
                results_split.append(r)

    def _avg(results: List[Dict], key: str) -> float:
        vals = [r[key] for r in results]
        return float(sum(vals) / max(1, len(vals)))

    action_acc_self_split    = _avg(results_split,   "action_acc_self")
    action_acc_world_split   = _avg(results_split,   "action_acc_world")
    contact_acc_world_split  = _avg(results_split,   "contact_acc_world")
    contact_acc_self_split   = _avg(results_split,   "contact_acc_self")
    action_acc_self_unified  = _avg(results_unified, "action_acc_self")
    contact_acc_world_unified = _avg(results_unified, "contact_acc_world")

    action_dissociation_split  = action_acc_self_split  - action_acc_world_split
    contact_dissociation_split = contact_acc_world_split - contact_acc_self_split

    n_contact_min = min(r["n_contact_probe"] for r in results_split + results_unified)

    # Pre-registered PASS criteria
    c1_pass = action_dissociation_split   > 0.10
    c2_pass = contact_dissociation_split  > 0.05
    c3_pass = (action_acc_self_split - action_acc_self_unified)   > 0.03
    c4_pass = (contact_acc_world_split - contact_acc_world_unified) > 0.03
    c5_pass = n_contact_min >= 20

    all_pass = c1_pass and c2_pass and c3_pass and c4_pass and c5_pass
    criteria_met = sum([c1_pass, c2_pass, c3_pass, c4_pass, c5_pass])
    status = "PASS" if all_pass else "FAIL"

    if all_pass:
        decision = "retain_ree"
    elif c1_pass and c2_pass:
        decision = "hybridize"
    else:
        decision = "retire_ree_claim"

    print(f"\n[V3-EXQ-047g] Final results:", flush=True)
    print(
        f"  action: self_split={action_acc_self_split:.3f}"
        f"  world_split={action_acc_world_split:.3f}"
        f"  self_unified={action_acc_self_unified:.3f}",
        flush=True,
    )
    print(
        f"  contact: world_split={contact_acc_world_split:.3f}"
        f"  self_split={contact_acc_self_split:.3f}"
        f"  world_unified={contact_acc_world_unified:.3f}",
        flush=True,
    )
    print(
        f"  action_dissoc_split={action_dissociation_split:+.3f}"
        f"  contact_dissoc_split={contact_dissociation_split:+.3f}",
        flush=True,
    )
    print(
        f"  decision={decision}  status={status} ({criteria_met}/5)",
        flush=True,
    )

    failure_notes: List[str] = []
    if not c1_pass:
        failure_notes.append(
            f"C1 FAIL: action_dissociation_split={action_dissociation_split:+.3f} <= 0.10"
            " (z_self not better motor probe than z_world in split condition)"
        )
    if not c2_pass:
        failure_notes.append(
            f"C2 FAIL: contact_dissociation_split={contact_dissociation_split:+.3f} <= 0.05"
            " (z_world not better world probe than z_self in split condition)"
        )
    if not c3_pass:
        failure_notes.append(
            f"C3 FAIL: split z_self action advantage over unified ="
            f" {action_acc_self_split - action_acc_self_unified:+.3f} <= 0.03"
        )
    if not c4_pass:
        failure_notes.append(
            f"C4 FAIL: split z_world contact advantage over unified ="
            f" {contact_acc_world_split - contact_acc_world_unified:+.3f} <= 0.03"
        )
    if not c5_pass:
        failure_notes.append(
            f"C5 FAIL: n_contact_min={n_contact_min} < 20 (insufficient contact events)"
        )

    for note in failure_notes:
        print(f"  {note}", flush=True)

    metrics = {
        "action_acc_self_split":     float(action_acc_self_split),
        "action_acc_world_split":    float(action_acc_world_split),
        "action_acc_self_unified":   float(action_acc_self_unified),
        "contact_acc_world_split":   float(contact_acc_world_split),
        "contact_acc_self_split":    float(contact_acc_self_split),
        "contact_acc_world_unified": float(contact_acc_world_unified),
        "action_dissociation_split": float(action_dissociation_split),
        "contact_dissociation_split": float(contact_dissociation_split),
        "n_contact_min":             float(n_contact_min),
        "n_seeds":                   float(len(seeds)),
        "alpha_world":               float(alpha_world),
        "crit1_pass":                1.0 if c1_pass else 0.0,
        "crit2_pass":                1.0 if c2_pass else 0.0,
        "crit3_pass":                1.0 if c3_pass else 0.0,
        "crit4_pass":                1.0 if c4_pass else 0.0,
        "crit5_pass":                1.0 if c5_pass else 0.0,
        "criteria_met":              float(criteria_met),
    }

    per_split_rows = "\n".join(
        f"  seed={r['seed']}: action_self={r['action_acc_self']:.3f}"
        f" action_world={r['action_acc_world']:.3f}"
        f" contact_world={r['contact_acc_world']:.3f}"
        f" contact_self={r['contact_acc_self']:.3f}"
        f" n_contact={r['n_contact_probe']}"
        for r in results_split
    )
    per_unified_rows = "\n".join(
        f"  seed={r['seed']}: action_self={r['action_acc_self']:.3f}"
        f" action_world={r['action_acc_world']:.3f}"
        f" contact_world={r['contact_acc_world']:.3f}"
        f" contact_self={r['contact_acc_self']:.3f}"
        for r in results_unified
    )

    failure_section = ""
    if failure_notes:
        failure_section = "\n## Failure Notes\n\n" + "\n".join(
            f"- {n}" for n in failure_notes
        )

    summary_markdown = (
        f"# V3-EXQ-047g -- SD-005 Functional Separation Probe\n\n"
        f"**Status:** {status}\n"
        f"**Claims:** SD-005\n"
        f"**Decision:** {decision}\n"
        f"**Seeds:** {list(seeds)}\n"
        f"**alpha_world:** {alpha_world}  (SD-008)\n"
        f"**Warmup:** {warmup_episodes} eps  **Probe:** {probe_episodes} eps\n"
        f"**Design note:** Functional cross-decoding probes. No orthogonality constraint.\n\n"
        f"## Pre-Registered Thresholds\n\n"
        f"C1: action_dissociation_split (self - world) > 0.10\n"
        f"C2: contact_dissociation_split (world - self) > 0.05\n"
        f"C3: action_acc_self_split - unified > 0.03\n"
        f"C4: contact_acc_world_split - unified > 0.03\n"
        f"C5: n_contact_probe >= 20\n\n"
        f"## Results\n\n"
        f"| Condition | action_self | action_world | contact_world | contact_self |\n"
        f"|-----------|------------|--------------|---------------|-------------|\n"
        f"| SPLIT     | {action_acc_self_split:.3f}       | {action_acc_world_split:.3f}"
        f"         | {contact_acc_world_split:.3f}          | {contact_acc_self_split:.3f}        |\n"
        f"| UNIFIED   | {action_acc_self_unified:.3f}       | n/a"
        f"          | {contact_acc_world_unified:.3f}          | n/a         |\n\n"
        f"**action_dissociation_split (self - world): {action_dissociation_split:+.3f}**\n"
        f"**contact_dissociation_split (world - self): {contact_dissociation_split:+.3f}**\n\n"
        f"## PASS Criteria\n\n"
        f"| Criterion | Result | Value |\n"
        f"|---|---|---|\n"
        f"| C1: action dissociation > 0.10 | {'PASS' if c1_pass else 'FAIL'}"
        f" | {action_dissociation_split:+.3f} |\n"
        f"| C2: contact dissociation > 0.05 | {'PASS' if c2_pass else 'FAIL'}"
        f" | {contact_dissociation_split:+.3f} |\n"
        f"| C3: split vs unified action (+0.03) | {'PASS' if c3_pass else 'FAIL'}"
        f" | {action_acc_self_split - action_acc_self_unified:+.3f} |\n"
        f"| C4: split vs unified contact (+0.03) | {'PASS' if c4_pass else 'FAIL'}"
        f" | {contact_acc_world_split - contact_acc_world_unified:+.3f} |\n"
        f"| C5: n_contact_min >= 20 | {'PASS' if c5_pass else 'FAIL'}"
        f" | {n_contact_min} |\n\n"
        f"Criteria met: {criteria_met}/5 -> **{status}**\n\n"
        f"## Per-Seed\n\n"
        f"SPLIT:\n{per_split_rows}\n\n"
        f"UNIFIED:\n{per_unified_rows}\n"
        f"{failure_section}\n"
    )

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
            r["n_fatal"] for r in results_split + results_unified
        ),
    }


if __name__ == "__main__":
    import argparse
    import json
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds",       type=int, nargs="+", default=[42, 7])
    parser.add_argument("--warmup",      type=int,   default=400)
    parser.add_argument("--probe-eps",   type=int,   default=20)
    parser.add_argument("--steps",       type=int,   default=200)
    parser.add_argument("--alpha-world", type=float, default=0.9)
    parser.add_argument("--alpha-self",  type=float, default=0.3)
    parser.add_argument("--harm-scale",  type=float, default=0.02)
    parser.add_argument("--proximity-scale", type=float, default=0.05)
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
