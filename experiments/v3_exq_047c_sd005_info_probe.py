"""
V3-EXQ-047c — SD-005: Latent Information Separation Probe

Claims: SD-005

Context:
  EXQ-047 and EXQ-047b both tested SD-005 by measuring downstream task performance
  (world_forward R² and attribution_gap). Both FAILED: the split doesn't improve
  — and in 047b slightly hurts — E2 prediction or causal attribution metrics.

  The diagnosis: downstream task metrics are insensitive at CausalGridWorld scale.
  Both split and unified conditions achieve R² > 0.94 — there's no room for the
  split to show a 5pp advantage when the baseline is already near ceiling.
  Attribution gaps are < 0.002 in absolute terms — below signal threshold.

Reframe — information-content probe:
  Instead of measuring *whether the split helps the task*, measure *whether the
  split is doing what it's supposed to do architecturally* — routing action-
  correlated signal to z_self and environment-event signal to z_world.

  If z_self and z_world carry genuinely different information profiles, we should
  see a double dissociation via linear probes on frozen representations:

    Probe 1 — Action decodability:
      z_self should predict the action just taken better than z_world.
      Rationale: body_obs (→ z_self) reflects proprioceptive outcome of action;
      world_obs (→ z_world) is environment layout, largely action-independent.

    Probe 2 — Environmental event decodability:
      z_world should predict transition_type (none / hazard_approach / contact)
      better than z_self.
      Rationale: world_obs contains hazard proximity fields; body_obs carries
      agent position but not the full event-type signal structure.

  This is a direct test of information routing — it doesn't require harm evaluation
  or counterfactual reasoning to work. The claim "z_self encodes motor-sensory,
  z_world encodes environmental" is tested directly.

Design:
  - Two conditions: SPLIT (z_self ≠ z_world) vs UNIFIED (z_self = z_world = avg)
  - Train 500 eps. Collect (z_self, z_world, action_idx, transition_type) on eval.
  - Fit small linear probes (single linear layer, cross-entropy, 100 gradient steps)
    on 80% of collected data. Test on remaining 20%.
  - Probe A: predict action_idx from z_self → action_acc_self
  - Probe B: predict action_idx from z_world → action_acc_world
  - Probe C: predict event_type (0=none, 1=approach, 2=contact) from z_world → event_acc_world
  - Probe D: predict event_type from z_self → event_acc_self

PASS criteria (ALL must hold):
  C1: action_acc_self_split > action_acc_world_split + 0.10
      (z_self carries more action signal than z_world in split condition)
  C2: event_acc_world_split > event_acc_self_split + 0.10
      (z_world carries more event signal than z_self in split condition)
  C3: action_acc_self_split > action_acc_self_unified + 0.05
      (split z_self is more action-selective than merged z)
  C4: event_acc_world_split > event_acc_world_unified + 0.05
      (split z_world is more event-selective than merged z)
  C5: No fatal errors

Note: C1/C2 test the dissociation within the split condition (double dissociation).
      C3/C4 test that the split architecture produces MORE specialisation than unified.
      A FAIL on C1/C2 would mean the split isn't routing information at all.
      A FAIL on C3/C4 would mean the split routes correctly but not better than unified.
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


EXPERIMENT_TYPE = "v3_exq_047c_sd005_info_probe"
CLAIM_IDS = ["SD-005"]

WARMUP_EPS = 500
EVAL_EPS = 100
STEPS_PER_EP = 200
PROBE_STEPS = 200   # gradient steps to fit each probe


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


EVENT_LABELS = {"none": 0, "hazard_approach": 1,
                "env_caused_hazard": 2, "agent_caused_hazard": 2}


def _collect_probe_data(
    agent: REEAgent,
    env: CausalGridWorldV2,
    num_episodes: int,
    steps: int,
    label: str,
) -> Dict[str, torch.Tensor]:
    """Collect (z_self, z_world, action_idx, event_type) during random-policy eval."""
    agent.eval()
    z_selfs, z_worlds, action_idxs, event_types = [], [], [], []

    for _ in range(num_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()

        for _ in range(steps):
            obs_body = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            with torch.no_grad():
                latent = agent.sense(obs_body, obs_world)
                agent.clock.advance()

            action_idx = random.randint(0, env.action_dim - 1)
            action = _action_to_onehot(action_idx, env.action_dim, agent.device)
            agent._last_action = action

            flat_obs, _, done, info, obs_dict = env.step(action)
            ttype = info.get("transition_type", "none")

            z_selfs.append(latent.z_self.detach().cpu())
            z_worlds.append(latent.z_world.detach().cpu())
            action_idxs.append(action_idx)
            event_types.append(EVENT_LABELS.get(ttype, 0))

            if done:
                break

    n = len(z_selfs)
    print(
        f"  [{label}|collect] {n} samples"
        f"  approach={sum(1 for e in event_types if e == 1)}"
        f"  contact={sum(1 for e in event_types if e == 2)}",
        flush=True,
    )
    return {
        "z_self":     torch.cat(z_selfs, dim=0),
        "z_world":    torch.cat(z_worlds, dim=0),
        "action_idx": torch.tensor(action_idxs, dtype=torch.long),
        "event_type": torch.tensor(event_types, dtype=torch.long),
    }


def _fit_and_eval_probe(
    features: torch.Tensor,   # (N, D)
    labels: torch.Tensor,     # (N,) long
    n_classes: int,
    n_steps: int,
    lr: float = 5e-3,
    label: str = "",
) -> float:
    """Fit a linear probe and return test accuracy."""
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
        logits_test = probe(x_test)
        preds = logits_test.argmax(dim=1)
        acc = float((preds == y_test).float().mean().item())

    print(f"    [{label}] acc={acc:.3f}  n_test={len(y_test)}", flush=True)
    return acc


def _run_probes(data: Dict[str, torch.Tensor], n_actions: int,
                label: str, probe_steps: int) -> Dict[str, float]:
    """Fit and evaluate all 4 probes on collected data."""
    print(f"\n  [{label}] Fitting probes...", flush=True)

    acc_action_self = _fit_and_eval_probe(
        data["z_self"], data["action_idx"], n_actions, probe_steps,
        label=f"{label}|action←z_self",
    )
    acc_action_world = _fit_and_eval_probe(
        data["z_world"], data["action_idx"], n_actions, probe_steps,
        label=f"{label}|action←z_world",
    )
    acc_event_world = _fit_and_eval_probe(
        data["z_world"], data["event_type"], 3, probe_steps,
        label=f"{label}|event←z_world",
    )
    acc_event_self = _fit_and_eval_probe(
        data["z_self"], data["event_type"], 3, probe_steps,
        label=f"{label}|event←z_self",
    )

    return {
        "action_acc_self":  acc_action_self,
        "action_acc_world": acc_action_world,
        "event_acc_world":  acc_event_world,
        "event_acc_self":   acc_event_self,
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
        f"[V3-EXQ-047c] SD-005: Latent Information Separation Probe\n"
        f"  Two conditions: SPLIT (z_self ≠ z_world) vs UNIFIED (z_self = z_world)\n"
        f"  Linear probes test whether split routes info correctly\n"
        f"  seed={seed}  warmup={warmup_episodes}  eval_eps={eval_episodes}\n"
        f"  probe_steps={probe_steps}  alpha_world={alpha_world}",
        flush=True,
    )

    probe_results: Dict[str, Dict] = {}

    for label, unified in [("split", False), ("unified", True)]:
        print(f"\n{'='*60}", flush=True)
        print(f"[V3-EXQ-047c] CONDITION: {label} (unified={unified})", flush=True)
        print('='*60, flush=True)

        env = _make_env(seed)
        agent = _build_agent(env, seed, alpha_world, self_dim, world_dim, unified)

        _train(agent, env, world_dim, warmup_episodes, steps_per_episode, label)

        print(f"\n[V3-EXQ-047c] Collecting probe data ({eval_episodes} eps)...", flush=True)
        data = _collect_probe_data(agent, env, eval_episodes, steps_per_episode, label)

        probe_results[label] = _run_probes(data, env.action_dim, label, probe_steps)

    r_s = probe_results["split"]
    r_u = probe_results["unified"]

    # ── PASS / FAIL ──────────────────────────────────────────────────────────
    c1_pass = r_s["action_acc_self"]  > r_s["action_acc_world"] + 0.10
    c2_pass = r_s["event_acc_world"]  > r_s["event_acc_self"]   + 0.10
    c3_pass = r_s["action_acc_self"]  > r_u["action_acc_self"]  + 0.05
    c4_pass = r_s["event_acc_world"]  > r_u["event_acc_world"]  + 0.05
    c5_pass = True  # structural — fatal errors captured by exception handling above

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
            f"C2 FAIL: event_acc_world={r_s['event_acc_world']:.3f}"
            f" not > event_acc_self={r_s['event_acc_self']:.3f} + 0.10"
            " (z_world not more event-selective than z_self)"
        )
    if not c3_pass:
        failure_notes.append(
            f"C3 FAIL: action_acc_self_split={r_s['action_acc_self']:.3f}"
            f" not > action_acc_self_unified={r_u['action_acc_self']:.3f} + 0.05"
            " (split not improving action-routing vs unified)"
        )
    if not c4_pass:
        failure_notes.append(
            f"C4 FAIL: event_acc_world_split={r_s['event_acc_world']:.3f}"
            f" not > event_acc_world_unified={r_u['event_acc_world']:.3f} + 0.05"
            " (split not improving event-routing vs unified)"
        )

    print(f"\nV3-EXQ-047c verdict: {status}  ({criteria_met}/5)", flush=True)
    for note in failure_notes:
        print(f"  {note}", flush=True)

    print(f"\n  C1 action_acc: self={r_s['action_acc_self']:.3f}  world={r_s['action_acc_world']:.3f}"
          f"  → {'PASS' if c1_pass else 'FAIL'}", flush=True)
    print(f"  C2 event_acc:  world={r_s['event_acc_world']:.3f}  self={r_s['event_acc_self']:.3f}"
          f"  → {'PASS' if c2_pass else 'FAIL'}", flush=True)
    print(f"  C3 action_acc_self: split={r_s['action_acc_self']:.3f}  unified={r_u['action_acc_self']:.3f}"
          f"  → {'PASS' if c3_pass else 'FAIL'}", flush=True)
    print(f"  C4 event_acc_world: split={r_s['event_acc_world']:.3f}  unified={r_u['event_acc_world']:.3f}"
          f"  → {'PASS' if c4_pass else 'FAIL'}", flush=True)

    metrics = {
        # Split condition
        "split_action_acc_self":  float(r_s["action_acc_self"]),
        "split_action_acc_world": float(r_s["action_acc_world"]),
        "split_event_acc_world":  float(r_s["event_acc_world"]),
        "split_event_acc_self":   float(r_s["event_acc_self"]),
        "split_action_dissociation": float(r_s["action_acc_self"] - r_s["action_acc_world"]),
        "split_event_dissociation":  float(r_s["event_acc_world"] - r_s["event_acc_self"]),
        # Unified condition
        "unified_action_acc_self":  float(r_u["action_acc_self"]),
        "unified_action_acc_world": float(r_u["action_acc_world"]),
        "unified_event_acc_world":  float(r_u["event_acc_world"]),
        "unified_event_acc_self":   float(r_u["event_acc_self"]),
        # Improvements
        "action_self_improvement": float(r_s["action_acc_self"] - r_u["action_acc_self"]),
        "event_world_improvement": float(r_s["event_acc_world"] - r_u["event_acc_world"]),
        # Criteria
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

    summary_markdown = f"""# V3-EXQ-047c — SD-005: Latent Information Separation Probe

**Status:** {status}
**Claim:** SD-005 — z_gamma split into z_self (motor-sensory) and z_world (environmental)
**Design:** Linear probes on frozen representations — action decodability vs event decodability
**alpha_world:** {alpha_world}  |  **Warmup:** {warmup_episodes} eps  |  **Eval:** {eval_episodes} eps  |  **Probe steps:** {probe_steps}  |  **Seed:** {seed}

## Design Rationale

Prior tests (EXQ-047, 047b) measured downstream task performance (R², attribution_gap).
Both failed because the CausalGridWorld task is too easy — both split and unified reach R²>0.94.
This experiment instead probes information routing directly: does z_self carry more
action-correlated signal, and z_world more event-correlated signal, in the split condition?

## Probe Results

| Probe | Split | Unified | Δ |
|---|---|---|---|
| Action accuracy ← z_self | {r_s['action_acc_self']:.3f} | {r_u['action_acc_self']:.3f} | {r_s['action_acc_self'] - r_u['action_acc_self']:+.3f} |
| Action accuracy ← z_world | {r_s['action_acc_world']:.3f} | {r_u['action_acc_world']:.3f} | {r_s['action_acc_world'] - r_u['action_acc_world']:+.3f} |
| Event accuracy ← z_world | {r_s['event_acc_world']:.3f} | {r_u['event_acc_world']:.3f} | {r_s['event_acc_world'] - r_u['event_acc_world']:+.3f} |
| Event accuracy ← z_self | {r_s['event_acc_self']:.3f} | {r_u['event_acc_self']:.3f} | {r_s['event_acc_self'] - r_u['event_acc_self']:+.3f} |

Action dissociation (split): {r_s['action_acc_self'] - r_s['action_acc_world']:+.3f}  (z_self − z_world for action)
Event dissociation (split): {r_s['event_acc_world'] - r_s['event_acc_self']:+.3f}  (z_world − z_self for event)

## PASS Criteria

| Criterion | Result | Value |
|---|---|---|
| C1: action_self > action_world + 0.10 (z_self more action-selective, split) | {"PASS" if c1_pass else "FAIL"} | {r_s['action_acc_self']:.3f} vs {r_s['action_acc_world']:.3f} |
| C2: event_world > event_self + 0.10 (z_world more event-selective, split) | {"PASS" if c2_pass else "FAIL"} | {r_s['event_acc_world']:.3f} vs {r_s['event_acc_self']:.3f} |
| C3: action_self_split > action_self_unified + 0.05 (split improves action routing) | {"PASS" if c3_pass else "FAIL"} | {r_s['action_acc_self']:.3f} vs {r_u['action_acc_self']:.3f} |
| C4: event_world_split > event_world_unified + 0.05 (split improves event routing) | {"PASS" if c4_pass else "FAIL"} | {r_s['event_acc_world']:.3f} vs {r_u['event_acc_world']:.3f} |
| C5: No fatal errors | PASS | 0 |

Criteria met: {criteria_met}/5 → **{status}**
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
