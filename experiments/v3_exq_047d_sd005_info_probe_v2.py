"""
V3-EXQ-047d -- SD-005: Latent Information Separation Probe (Design Fix)

Claims: SD-005

Context:
  EXQ-047c FAIL -- design flaws made the experiment uninformative:
  1. Actions were random -> no decodable action signal in any representation.
     Action probe accuracy was at chance (~0.2) for BOTH z_self and z_world,
     so C1 (z_self > z_world + 0.10 for action) could not pass.
  2. Event class imbalance: 86% of steps are "hazard_approach", majority-class
     baseline = 0.86. Both split and unified probes converged to predicting
     the majority class, producing identical accuracy regardless of features.
  3. Identical seed reset in _build_agent for both conditions + same majority-class
     convergence produced bit-for-bit identical results.

Fixes in EXQ-047d:
  1. ACTION PROBE: Use round-robin action assignment during collection.
     Episodes are assigned a fixed "dominant action" (ep 0 -> action 0, ep 1 -> action 1,
     cycling). 80% of steps use the dominant action; 20% random. This creates a systematic
     action->body_state correlation that z_self should encode better than z_world.
     Rationale: body_obs reflects proprioceptive outcome of movement. Repeatedly moving
     in one direction creates a distinctive body trace. World layout (z_world) is largely
     invariant to which direction the agent moved.

  2. EVENT PROBE: Use binary classification (hazard_contact vs everything_else) with
     balanced sampling. Contact events are rare (~4%) but represent the most salient
     environmental events. We oversample contacts to achieve 50/50 train distribution.
     z_world should encode proximity signals predicting contact; z_self (body position
     alone) should be less predictive.

  3. PROBE STEPS: 500 (up from 200) to allow convergence beyond majority class.

Design:
  - Two conditions: SPLIT (z_self != z_world) vs UNIFIED (z_self = z_world = avg)
  - Warmup: 500 eps. Collect (z_self, z_world, dominant_action, contact) on eval.
  - Action probe: predict dominant_action_category (which ep-assigned action was dominant)
  - Contact probe: predict hazard_contact (binary), balanced sampling
  - Fit probes with 500 gradient steps

PASS criteria (ALL must hold):
  C1: action_acc_self_split > action_acc_world_split + 0.10
      (z_self carries more action signal than z_world in split condition)
  C2: contact_acc_world_split > contact_acc_self_split + 0.10
      (z_world carries more contact signal than z_self in split condition)
  C3: action_acc_self_split > action_acc_self_unified + 0.05
      (split z_self is more action-selective than merged z)
  C4: contact_acc_world_split > contact_acc_world_unified + 0.05
      (split z_world is more contact-selective than merged z)
  C5: No fatal errors
"""

import sys
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE = "v3_exq_047d_sd005_info_probe_v2"
CLAIM_IDS = ["SD-005"]

WARMUP_EPS = 500
EVAL_EPS = 200
STEPS_PER_EP = 200
PROBE_STEPS = 500


def _action_to_onehot(idx: int, n: int, device) -> torch.Tensor:
    v = torch.zeros(1, n, device=device)
    v[0, idx] = 1.0
    return v


def _make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        seed=seed, size=12, num_hazards=4, num_resources=5,
        hazard_harm=0.02,
        env_drift_interval=5, env_drift_prob=0.1,
        proximity_harm_scale=0.05,
        proximity_benefit_scale=0.03,
        proximity_approach_threshold=0.15,
        hazard_field_decay=0.5,
    )


def _build_agent(env: CausalGridWorldV2, seed: int, alpha_world: float,
                 self_dim: int, world_dim: int, unified: bool) -> REEAgent:
    torch.manual_seed(seed)
    random.seed(seed)
    config = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=self_dim,
        world_dim=world_dim,
        alpha_world=alpha_world,
        alpha_self=0.3,
        reafference_action_dim=env.action_dim,
    )
    config.latent.unified_latent_mode = unified
    return REEAgent(config)


def _train(agent: REEAgent, env: CausalGridWorldV2, world_dim: int,
           num_episodes: int, steps: int, label: str) -> None:
    agent.train()
    optimizer = optim.Adam(
        list(agent.e1.parameters()) +
        list(agent.e2.world_transition.parameters()) +
        list(agent.e2.world_action_encoder.parameters()),
        lr=1e-3,
    )
    wf_buf: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    for ep in range(num_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()
        z_world_prev: Optional[torch.Tensor] = None
        a_prev: Optional[torch.Tensor] = None

        for _ in range(steps):
            obs_body = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            latent = agent.sense(obs_body, obs_world)
            agent.clock.advance()
            z_world_curr = latent.z_world.detach()

            action_idx = random.randint(0, env.action_dim - 1)
            action = _action_to_onehot(action_idx, env.action_dim, agent.device)
            agent._last_action = action

            flat_obs, _, done, _, obs_dict = env.step(action)

            e1_loss = agent.compute_prediction_loss()
            if e1_loss.requires_grad:
                optimizer.zero_grad()
                e1_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                optimizer.step()

            if z_world_prev is not None and a_prev is not None:
                wf_buf.append((z_world_prev.cpu(), a_prev.cpu(), z_world_curr.cpu()))
                if len(wf_buf) > 3000:
                    wf_buf = wf_buf[-3000:]

            if len(wf_buf) >= 16:
                k = min(32, len(wf_buf))
                idxs = torch.randperm(len(wf_buf))[:k].tolist()
                zw_b = torch.cat([wf_buf[i][0] for i in idxs]).to(agent.device)
                a_b = torch.cat([wf_buf[i][1] for i in idxs]).to(agent.device)
                zw1_b = torch.cat([wf_buf[i][2] for i in idxs]).to(agent.device)
                wf_pred = agent.e2.world_forward(zw_b, a_b)
                wf_loss = F.mse_loss(wf_pred, zw1_b)
                if wf_loss.requires_grad:
                    optimizer.zero_grad()
                    wf_loss.backward()
                    torch.nn.utils.clip_grad_norm_(agent.parameters(), 1.0)
                    optimizer.step()

            z_world_prev = z_world_curr
            a_prev = action.detach()
            if done:
                break

        if (ep + 1) % 100 == 0 or ep == num_episodes - 1:
            print(f"  [{label}|train] ep {ep+1}/{num_episodes}", flush=True)


def _collect_probe_data(
    agent: REEAgent,
    env: CausalGridWorldV2,
    num_episodes: int,
    steps: int,
    label: str,
) -> Dict[str, object]:
    """Collect probe data using round-robin dominant actions.

    Each episode is assigned a dominant action (ep % n_actions).
    80% of steps use the dominant action; 20% random noise.
    This gives action probe a decodable signal while maintaining exploration.
    """
    agent.eval()
    n_actions = env.action_dim
    z_selfs, z_worlds, dominant_actions, contact_flags = [], [], [], []

    for ep in range(num_episodes):
        dominant_action_idx = ep % n_actions  # round-robin assignment
        flat_obs, obs_dict = env.reset()
        agent.reset()

        for _ in range(steps):
            obs_body = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            with torch.no_grad():
                latent = agent.sense(obs_body, obs_world)
                agent.clock.advance()

            # 80% dominant action, 20% random
            if random.random() < 0.8:
                action_idx = dominant_action_idx
            else:
                action_idx = random.randint(0, n_actions - 1)

            action = _action_to_onehot(action_idx, n_actions, agent.device)
            agent._last_action = action

            flat_obs, _, done, info, obs_dict = env.step(action)
            ttype = info.get("transition_type", "none")
            is_contact = 1 if ttype in ("env_caused_hazard", "agent_caused_hazard") else 0

            z_selfs.append(latent.z_self.detach().cpu())
            z_worlds.append(latent.z_world.detach().cpu())
            dominant_actions.append(dominant_action_idx)
            contact_flags.append(is_contact)

            if done:
                break

    n = len(z_selfs)
    n_contacts = sum(contact_flags)
    print(
        f"  [{label}|collect] {n} samples"
        f"  contacts={n_contacts} ({100*n_contacts/max(1,n):.1f}%)",
        flush=True,
    )

    return {
        "z_self":          torch.cat(z_selfs, dim=0),
        "z_world":         torch.cat(z_worlds, dim=0),
        "dominant_action": torch.tensor(dominant_actions, dtype=torch.long),
        "contact":         torch.tensor(contact_flags, dtype=torch.long),
        "n_contacts":      n_contacts,
        "n_total":         n,
    }


def _fit_and_eval_probe_balanced(
    features: torch.Tensor,   # (N, D)
    labels: torch.Tensor,     # (N,) long, binary (0 or 1)
    n_steps: int,
    lr: float = 5e-3,
    label: str = "",
) -> float:
    """Fit a binary probe with balanced class sampling and return test accuracy."""
    pos_idx = (labels == 1).nonzero(as_tuple=True)[0]
    neg_idx = (labels == 0).nonzero(as_tuple=True)[0]
    n_pos = len(pos_idx)
    n_neg = len(neg_idx)

    if n_pos == 0 or n_neg == 0:
        print(f"    [{label}] SKIP: n_pos={n_pos}  n_neg={n_neg} (degenerate)", flush=True)
        return 0.0

    # Balance: downsample majority to match minority
    n_min = min(n_pos, n_neg)
    pos_sample = pos_idx[torch.randperm(n_pos)[:n_min]]
    neg_sample = neg_idx[torch.randperm(n_neg)[:n_min]]
    bal_idx = torch.cat([pos_sample, neg_sample])
    bal_idx = bal_idx[torch.randperm(len(bal_idx))]

    n_train = int(len(bal_idx) * 0.8)
    train_idx = bal_idx[:n_train]
    test_idx = bal_idx[n_train:]

    if len(test_idx) < 4:
        print(f"    [{label}] SKIP: too few test samples ({len(test_idx)})", flush=True)
        return 0.0

    x_train, y_train = features[train_idx], labels[train_idx]
    x_test, y_test = features[test_idx], labels[test_idx]

    probe = nn.Linear(features.shape[1], 2)
    opt = optim.Adam(probe.parameters(), lr=lr)

    probe.train()
    for _ in range(n_steps):
        logits = probe(x_train)
        loss = F.cross_entropy(logits, y_train)
        opt.zero_grad()
        loss.backward()
        opt.step()

    probe.eval()
    with torch.no_grad():
        preds = probe(x_test).argmax(dim=1)
        acc = float((preds == y_test).float().mean().item())

    print(
        f"    [{label}] acc={acc:.3f}  n_test={len(y_test)}"
        f"  (pos={int((y_test==1).sum())}  neg={int((y_test==0).sum())})",
        flush=True,
    )
    return acc


def _fit_and_eval_probe(
    features: torch.Tensor,   # (N, D)
    labels: torch.Tensor,     # (N,) long
    n_classes: int,
    n_steps: int,
    lr: float = 5e-3,
    label: str = "",
) -> float:
    """Fit a multi-class linear probe and return test accuracy."""
    N = features.shape[0]
    n_train = int(N * 0.8)
    idx = torch.randperm(N)
    train_idx, test_idx = idx[:n_train], idx[n_train:]

    x_train, y_train = features[train_idx], labels[train_idx]
    x_test, y_test = features[test_idx], labels[test_idx]

    probe = nn.Linear(features.shape[1], n_classes)
    opt = optim.Adam(probe.parameters(), lr=lr)

    probe.train()
    for _ in range(n_steps):
        logits = probe(x_train)
        loss = F.cross_entropy(logits, y_train)
        opt.zero_grad()
        loss.backward()
        opt.step()

    probe.eval()
    with torch.no_grad():
        preds = probe(x_test).argmax(dim=1)
        acc = float((preds == y_test).float().mean().item())

    print(f"    [{label}] acc={acc:.3f}  n_test={len(y_test)}", flush=True)
    return acc


def _run_probes(data: Dict, n_actions: int,
                label: str, probe_steps: int) -> Dict[str, float]:
    """Fit and evaluate probes on collected data."""
    print(f"\n  [{label}] Fitting probes...", flush=True)

    # Action probe: multi-class, predict dominant_action from z_self vs z_world
    acc_action_self = _fit_and_eval_probe(
        data["z_self"], data["dominant_action"], n_actions, probe_steps,
        label=f"{label}|action<-z_self",
    )
    acc_action_world = _fit_and_eval_probe(
        data["z_world"], data["dominant_action"], n_actions, probe_steps,
        label=f"{label}|action<-z_world",
    )

    # Contact probe: binary balanced, predict hazard_contact from z_world vs z_self
    acc_contact_world = _fit_and_eval_probe_balanced(
        data["z_world"], data["contact"], probe_steps,
        label=f"{label}|contact<-z_world",
    )
    acc_contact_self = _fit_and_eval_probe_balanced(
        data["z_self"], data["contact"], probe_steps,
        label=f"{label}|contact<-z_self",
    )

    return {
        "action_acc_self":   acc_action_self,
        "action_acc_world":  acc_action_world,
        "contact_acc_world": acc_contact_world,
        "contact_acc_self":  acc_contact_self,
    }


def run(
    seed: int = 0,
    warmup_episodes: int = WARMUP_EPS,
    eval_episodes: int = EVAL_EPS,
    steps_per_episode: int = STEPS_PER_EP,
    probe_steps: int = PROBE_STEPS,
    alpha_world: float = 0.9,
    self_dim: int = 32,
    world_dim: int = 32,
    **kwargs,
) -> dict:
    torch.manual_seed(seed)
    random.seed(seed)

    print(
        f"[V3-EXQ-047d] SD-005: Latent Information Separation Probe (Design Fix)\n"
        f"  Fix 1: round-robin action assignment for decodable action signal\n"
        f"  Fix 2: binary contact probe with balanced sampling\n"
        f"  Fix 3: 500 probe steps (was 200)\n"
        f"  Two conditions: SPLIT (z_self != z_world) vs UNIFIED (z_self = z_world)\n"
        f"  seed={seed}  warmup={warmup_episodes}  eval_eps={eval_episodes}\n"
        f"  probe_steps={probe_steps}  alpha_world={alpha_world}",
        flush=True,
    )

    probe_results: Dict[str, Dict] = {}
    n_contacts_by_condition: Dict[str, int] = {}

    for label, unified in [("split", False), ("unified", True)]:
        print(f"\n{'='*60}", flush=True)
        print(f"[V3-EXQ-047d] CONDITION: {label} (unified={unified})", flush=True)
        print('='*60, flush=True)

        env = _make_env(seed)
        agent = _build_agent(env, seed, alpha_world, self_dim, world_dim, unified)

        _train(agent, env, world_dim, warmup_episodes, steps_per_episode, label)

        print(f"\n[V3-EXQ-047d] Collecting probe data ({eval_episodes} eps)...", flush=True)
        data = _collect_probe_data(agent, env, eval_episodes, steps_per_episode, label)
        n_contacts_by_condition[label] = data["n_contacts"]

        probe_results[label] = _run_probes(data, env.action_dim, label, probe_steps)

    r_s = probe_results["split"]
    r_u = probe_results["unified"]

    n_contacts_split = n_contacts_by_condition["split"]

    # -- PASS / FAIL ---------------------------------------------------------
    c1_pass = r_s["action_acc_self"]   > r_s["action_acc_world"]  + 0.10
    c2_pass = r_s["contact_acc_world"] > r_s["contact_acc_self"]  + 0.10
    c3_pass = r_s["action_acc_self"]   > r_u["action_acc_self"]   + 0.05
    c4_pass = r_s["contact_acc_world"] > r_u["contact_acc_world"] + 0.05
    c5_pass = True

    all_pass = c1_pass and c2_pass and c3_pass and c4_pass and c5_pass
    status = "PASS" if all_pass else "FAIL"
    criteria_met = sum([c1_pass, c2_pass, c3_pass, c4_pass, c5_pass])

    failure_notes = []
    if not c1_pass:
        failure_notes.append(
            f"C1 FAIL: action_acc_self={r_s['action_acc_self']:.3f}"
            f" not > action_acc_world={r_s['action_acc_world']:.3f} + 0.10"
            " (z_self not more action-selective than z_world)"
        )
    if not c2_pass:
        failure_notes.append(
            f"C2 FAIL: contact_acc_world={r_s['contact_acc_world']:.3f}"
            f" not > contact_acc_self={r_s['contact_acc_self']:.3f} + 0.10"
            " (z_world not more contact-selective than z_self)"
        )
    if not c3_pass:
        failure_notes.append(
            f"C3 FAIL: action_acc_self_split={r_s['action_acc_self']:.3f}"
            f" not > action_acc_self_unified={r_u['action_acc_self']:.3f} + 0.05"
        )
    if not c4_pass:
        failure_notes.append(
            f"C4 FAIL: contact_acc_world_split={r_s['contact_acc_world']:.3f}"
            f" not > contact_acc_world_unified={r_u['contact_acc_world']:.3f} + 0.05"
        )

    print(f"\nV3-EXQ-047d verdict: {status}  ({criteria_met}/5)", flush=True)
    for note in failure_notes:
        print(f"  {note}", flush=True)

    print(
        f"\n  C1 action_acc: self={r_s['action_acc_self']:.3f}  world={r_s['action_acc_world']:.3f}"
        f"  -> {'PASS' if c1_pass else 'FAIL'}", flush=True,
    )
    print(
        f"  C2 contact_acc: world={r_s['contact_acc_world']:.3f}  self={r_s['contact_acc_self']:.3f}"
        f"  -> {'PASS' if c2_pass else 'FAIL'}", flush=True,
    )
    print(
        f"  C3 action_acc_self: split={r_s['action_acc_self']:.3f}  unified={r_u['action_acc_self']:.3f}"
        f"  -> {'PASS' if c3_pass else 'FAIL'}", flush=True,
    )
    print(
        f"  C4 contact_acc_world: split={r_s['contact_acc_world']:.3f}  unified={r_u['contact_acc_world']:.3f}"
        f"  -> {'PASS' if c4_pass else 'FAIL'}", flush=True,
    )

    metrics = {
        "split_action_acc_self":    float(r_s["action_acc_self"]),
        "split_action_acc_world":   float(r_s["action_acc_world"]),
        "split_contact_acc_world":  float(r_s["contact_acc_world"]),
        "split_contact_acc_self":   float(r_s["contact_acc_self"]),
        "split_action_dissociation":  float(r_s["action_acc_self"] - r_s["action_acc_world"]),
        "split_contact_dissociation": float(r_s["contact_acc_world"] - r_s["contact_acc_self"]),
        "unified_action_acc_self":    float(r_u["action_acc_self"]),
        "unified_action_acc_world":   float(r_u["action_acc_world"]),
        "unified_contact_acc_world":  float(r_u["contact_acc_world"]),
        "unified_contact_acc_self":   float(r_u["contact_acc_self"]),
        "action_self_improvement":  float(r_s["action_acc_self"] - r_u["action_acc_self"]),
        "contact_world_improvement": float(r_s["contact_acc_world"] - r_u["contact_acc_world"]),
        "n_contacts_split": float(n_contacts_split),
        "crit1_pass": 1.0 if c1_pass else 0.0,
        "crit2_pass": 1.0 if c2_pass else 0.0,
        "crit3_pass": 1.0 if c3_pass else 0.0,
        "crit4_pass": 1.0 if c4_pass else 0.0,
        "crit5_pass": 1.0,
        "criteria_met": float(criteria_met),
        "fatal_error_count": 0.0,
    }

    failure_section = ""
    if failure_notes:
        failure_section = "\n## Failure Notes\n\n" + "\n".join(f"- {n}" for n in failure_notes)

    summary_markdown = f"""# V3-EXQ-047d -- SD-005: Latent Information Separation Probe (Design Fix)

**Status:** {status}
**Claim:** SD-005 -- z_gamma split into z_self (motor-sensory) and z_world (environmental)
**Design:** Linear probes on frozen representations -- round-robin action decodability + binary contact probe
**alpha_world:** {alpha_world}  |  **Warmup:** {warmup_episodes} eps  |  **Eval:** {eval_episodes} eps  |  **Probe steps:** {probe_steps}  |  **Seed:** {seed}

## Design Rationale and Fixes

EXQ-047c used random actions (no decodable signal) and a 3-class event probe dominated
by 86% majority class (approach). Both split and unified conditions produced bit-for-bit
identical probe accuracies because both probes converged to majority-class prediction.

EXQ-047d fixes:
1. Round-robin action assignment (ep%n_actions = dominant action, 80% compliance)
   creates systematic action->body_state correlation that z_self should encode.
2. Binary contact probe (hazard_contact vs not) with balanced class sampling.
   Contact events are rare but represent salient environmental events requiring
   genuine proximity knowledge from z_world.
3. 500 probe steps for convergence beyond majority-class baseline.

## Probe Results

| Probe | Split | Unified | delta |
|---|---|---|---|
| Action accuracy <- z_self | {r_s['action_acc_self']:.3f} | {r_u['action_acc_self']:.3f} | {r_s['action_acc_self'] - r_u['action_acc_self']:+.3f} |
| Action accuracy <- z_world | {r_s['action_acc_world']:.3f} | {r_u['action_acc_world']:.3f} | {r_s['action_acc_world'] - r_u['action_acc_world']:+.3f} |
| Contact accuracy <- z_world | {r_s['contact_acc_world']:.3f} | {r_u['contact_acc_world']:.3f} | {r_s['contact_acc_world'] - r_u['contact_acc_world']:+.3f} |
| Contact accuracy <- z_self | {r_s['contact_acc_self']:.3f} | {r_u['contact_acc_self']:.3f} | {r_s['contact_acc_self'] - r_u['contact_acc_self']:+.3f} |

Action dissociation (split): {r_s['action_acc_self'] - r_s['action_acc_world']:+.3f}
Contact dissociation (split): {r_s['contact_acc_world'] - r_s['contact_acc_self']:+.3f}
Contacts collected (split): {n_contacts_split}

## PASS Criteria

| Criterion | Result | Value |
|---|---|---|
| C1: action_self > action_world + 0.10 (z_self more action-selective, split) | {"PASS" if c1_pass else "FAIL"} | {r_s['action_acc_self']:.3f} vs {r_s['action_acc_world']:.3f} |
| C2: contact_world > contact_self + 0.10 (z_world more contact-selective, split) | {"PASS" if c2_pass else "FAIL"} | {r_s['contact_acc_world']:.3f} vs {r_s['contact_acc_self']:.3f} |
| C3: action_self_split > action_self_unified + 0.05 | {"PASS" if c3_pass else "FAIL"} | {r_s['action_acc_self']:.3f} vs {r_u['action_acc_self']:.3f} |
| C4: contact_world_split > contact_world_unified + 0.05 | {"PASS" if c4_pass else "FAIL"} | {r_s['contact_acc_world']:.3f} vs {r_u['contact_acc_world']:.3f} |
| C5: No fatal errors | PASS | 0 |

Criteria met: {criteria_met}/5 -> **{status}**
{failure_section}
"""

    return {
        "status": status,
        "metrics": metrics,
        "summary_markdown": summary_markdown,
        "claim_ids": CLAIM_IDS,
        "evidence_direction": (
            "supports" if all_pass else ("mixed" if criteria_met >= 3 else "weakens")
        ),
        "experiment_type": EXPERIMENT_TYPE,
        "fatal_error_count": 0.0,
    }


if __name__ == "__main__":
    import argparse
    import json
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",        type=int,   default=0)
    parser.add_argument("--warmup",      type=int,   default=WARMUP_EPS)
    parser.add_argument("--eval-eps",    type=int,   default=EVAL_EPS)
    parser.add_argument("--steps",       type=int,   default=STEPS_PER_EP)
    parser.add_argument("--probe-steps", type=int,   default=PROBE_STEPS)
    parser.add_argument("--alpha-world", type=float, default=0.9)
    args = parser.parse_args()

    result = run(
        seed=args.seed,
        warmup_episodes=args.warmup,
        eval_episodes=args.eval_eps,
        steps_per_episode=args.steps,
        probe_steps=args.probe_steps,
        alpha_world=args.alpha_world,
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
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}", flush=True)
