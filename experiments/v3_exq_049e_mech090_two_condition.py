"""
V3-EXQ-049e — MECH-090: Beta-Gated Policy Propagation (two-condition design)

Claims: MECH-090

Root cause of EXQ-049d FAIL:
  Single-agent eval: once the agent trains down to variance ~0, it is permanently
  committed for all eval episodes. n_uncommitted_steps=0 means C2
  (uncommitted_release_concordance) has no data and trivially fails.

  The agent can't be in both committed and uncommitted states in the same eval
  without manufacturing artificial variance spikes — which would conflate the
  training dynamics with the test.

Fix — two-condition design (no hysteresis needed):
  Condition A — TRAINED agent (400 warmup episodes).
    E2 world-forward trained → running_variance collapses to ~0.
    Agent persistently committed. Tests: does gate ELEVATE when committed?
    Expected: committed_hold_concordance >> 0.6 (confirmed EXQ-049d: 1.0).

  Condition B — FRESH agent (same architecture, zero training).
    running_variance = precision_init = 0.5 > commit_threshold = 0.40.
    Agent persistently uncommitted throughout eval. Tests: does gate RELEASE
    when uncommitted?
    Expected: uncommitted_release_concordance >> 0.5.

  No hysteresis required: both conditions are at stable extremes of the variance
  distribution (near-0 vs 0.5). The boundary dynamics (oscillation around threshold)
  are not being tested here — that is a separate question about commitment onset.
  Any commit_threshold between 1e-4 and 0.45 correctly classifies both conditions.

Root cause chain:
  EXQ-049:  select() bypassed → gate never wired
  EXQ-049b: gate wired, post_action_update missing → variance frozen
  EXQ-049c: post_action_update deadlocked by _committed_trajectory guard
  EXQ-049d: direct update_running_variance(wf_err) — deadlock broken.
            C1 PASS (1.0). C2 FAIL: n_uncommitted_steps=0 (always committed).
  EXQ-049e: two-condition design separates C1 and C2 into distinct agents.

PASS criteria (ALL must hold):
  C1: trained agent committed_hold_concordance > 0.6
      (gate elevates when agent is committed — gate hold is correct)
  C2: fresh agent uncommitted_release_concordance > 0.5
      (gate releases when agent is uncommitted — gate release is correct)
  C3: trained agent hold_count > 0
      (gate actually held at some point during eval)
  C4: fresh agent propagation_count > 0
      (gate actually propagated / released at some point)
  C5: No fatal errors in either condition
"""

import sys
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
import torch.optim as optim

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE = "v3_exq_049e_mech090_two_condition"
CLAIM_IDS = ["MECH-090"]

ENV_KWARGS = dict(
    size=12, num_hazards=4, num_resources=5,
    hazard_harm=0.02,
    env_drift_interval=5, env_drift_prob=0.1,
    proximity_harm_scale=0.05,
    proximity_benefit_scale=0.03,
    proximity_approach_threshold=0.15,
    hazard_field_decay=0.5,
)


def _action_to_onehot(idx: int, n: int, device) -> torch.Tensor:
    v = torch.zeros(1, n, device=device)
    v[0, idx] = 1.0
    return v


def _mean_safe(lst: List[float]) -> float:
    return float(sum(lst) / len(lst)) if lst else 0.0


def _make_agent_and_env(
    seed: int,
    self_dim: int,
    world_dim: int,
    alpha_world: float,
) -> Tuple[REEAgent, CausalGridWorldV2]:
    torch.manual_seed(seed)
    random.seed(seed)
    env = CausalGridWorldV2(seed=seed, **ENV_KWARGS)
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
    agent = REEAgent(config)
    return agent, env


def _train_agent(
    agent: REEAgent,
    env: CausalGridWorldV2,
    num_episodes: int,
    steps_per_episode: int,
    world_dim: int,
    lr: float = 1e-3,
) -> Dict:
    """Train agent until running_variance collapses (expected ~400 eps from EXQ-049d)."""
    agent.train()

    optimizer = optim.Adam(list(agent.e1.parameters()), lr=lr)
    wf_optimizer = optim.Adam(
        list(agent.e2.world_transition.parameters()) +
        list(agent.e2.world_action_encoder.parameters()),
        lr=1e-3,
    )
    harm_eval_optimizer = optim.Adam(
        list(agent.e3.harm_eval_head.parameters()), lr=1e-4,
    )

    wf_buf: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    harm_buf_pos: List[torch.Tensor] = []
    harm_buf_neg: List[torch.Tensor] = []
    committed_frac_log: List[float] = []

    for ep in range(num_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()
        z_world_prev: Optional[torch.Tensor] = None
        action_prev:  Optional[torch.Tensor] = None
        z_self_prev:  Optional[torch.Tensor] = None

        for _ in range(steps_per_episode):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            latent = agent.sense(obs_body, obs_world)

            if z_self_prev is not None and action_prev is not None:
                agent.record_transition(z_self_prev, action_prev, latent.z_self.detach())

            ticks    = agent.clock.advance()
            e1_prior = (
                agent._e1_tick(latent) if ticks.get("e1_tick", False)
                else torch.zeros(1, world_dim, device=agent.device)
            )
            candidates = agent.generate_trajectories(latent, e1_prior, ticks)
            theta_z    = agent.theta_buffer.summary()
            z_world_curr = latent.z_world.detach()

            action = agent.select_action(candidates, ticks, temperature=1.0)
            if action is None:
                action = _action_to_onehot(
                    random.randint(0, env.action_dim - 1), env.action_dim, agent.device
                )
                agent._last_action = action

            flat_obs, harm_signal, done, info, obs_dict = env.step(action)

            committed_frac_log.append(
                1.0 if agent.e3._running_variance < agent.e3.commit_threshold else 0.0
            )

            if z_world_prev is not None and action_prev is not None:
                wf_buf.append((z_world_prev.cpu(), action_prev.cpu(), z_world_curr.cpu()))
                if len(wf_buf) > 2000:
                    wf_buf = wf_buf[-2000:]

            (harm_buf_pos if harm_signal < 0 else harm_buf_neg).append(theta_z.detach())
            if len(harm_buf_pos) > 1000: harm_buf_pos = harm_buf_pos[-1000:]
            if len(harm_buf_neg) > 1000: harm_buf_neg = harm_buf_neg[-1000:]

            e1_loss = agent.compute_prediction_loss()
            if e1_loss.requires_grad:
                optimizer.zero_grad()
                e1_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.e1.parameters(), 1.0)
                optimizer.step()

            if len(wf_buf) >= 16:
                k = min(32, len(wf_buf))
                idxs = torch.randperm(len(wf_buf))[:k].tolist()
                zw_b  = torch.cat([wf_buf[i][0] for i in idxs]).to(agent.device)
                a_b   = torch.cat([wf_buf[i][1] for i in idxs]).to(agent.device)
                zw1_b = torch.cat([wf_buf[i][2] for i in idxs]).to(agent.device)
                wf_pred = agent.e2.world_forward(zw_b, a_b)
                wf_loss = F.mse_loss(wf_pred, zw1_b)
                if wf_loss.requires_grad:
                    wf_optimizer.zero_grad()
                    wf_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        list(agent.e2.world_transition.parameters()) +
                        list(agent.e2.world_action_encoder.parameters()), 1.0,
                    )
                    wf_optimizer.step()
                # Direct variance update — breaks chicken-and-egg deadlock (EXQ-049d fix)
                with torch.no_grad():
                    agent.e3.update_running_variance(
                        (wf_pred.detach() - zw1_b).detach()
                    )

            if len(harm_buf_pos) >= 4 and len(harm_buf_neg) >= 4:
                k_p = min(16, len(harm_buf_pos))
                k_n = min(16, len(harm_buf_neg))
                pi = torch.randperm(len(harm_buf_pos))[:k_p].tolist()
                ni = torch.randperm(len(harm_buf_neg))[:k_n].tolist()
                zw_b = torch.cat(
                    [harm_buf_pos[i] for i in pi] + [harm_buf_neg[i] for i in ni], dim=0
                )
                target = torch.cat([
                    torch.ones(k_p, 1, device=agent.device),
                    torch.zeros(k_n, 1, device=agent.device),
                ], dim=0)
                pred = agent.e3.harm_eval(zw_b)
                harm_loss = F.mse_loss(pred, target)
                if harm_loss.requires_grad:
                    harm_eval_optimizer.zero_grad()
                    harm_loss.backward()
                    torch.nn.utils.clip_grad_norm_(agent.e3.harm_eval_head.parameters(), 0.5)
                    harm_eval_optimizer.step()

            z_world_prev = z_world_curr
            z_self_prev  = latent.z_self.detach()
            action_prev  = action.detach()
            if done:
                break

        if (ep + 1) % 100 == 0 or ep == num_episodes - 1:
            rv = agent.e3._running_variance
            cf = _mean_safe(committed_frac_log[-500:])
            print(
                f"  [train] ep {ep+1}/{num_episodes}"
                f"  running_var={rv:.6f}  committed_frac={cf:.3f}",
                flush=True,
            )

    return {
        "final_running_variance": agent.e3._running_variance,
        "mean_committed_fraction": _mean_safe(committed_frac_log),
    }


def _eval_gate_concordance(
    agent: REEAgent,
    env: CausalGridWorldV2,
    num_episodes: int,
    steps_per_episode: int,
    world_dim: int,
    label: str,
) -> Dict:
    """Measure gate concordance over eval episodes (no training)."""
    agent.eval()
    agent.beta_gate.reset()

    committed_elevated   = 0
    committed_not_elev   = 0
    uncommitted_elevated = 0
    uncommitted_not_elev = 0
    fatal = 0
    rvs: List[float] = []

    for _ in range(num_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()

        for _ in range(steps_per_episode):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            with torch.no_grad():
                latent = agent.sense(obs_body, obs_world)

            rvs.append(agent.e3._running_variance)

            with torch.no_grad():
                ticks    = agent.clock.advance()
                e1_prior = (
                    agent._e1_tick(latent) if ticks.get("e1_tick", False)
                    else torch.zeros(1, world_dim, device=agent.device)
                )
                candidates = agent.generate_trajectories(latent, e1_prior, ticks)

            try:
                with torch.no_grad():
                    action = agent.select_action(candidates, ticks, temperature=1.0)
                if action is None:
                    action = _action_to_onehot(
                        random.randint(0, env.action_dim - 1), env.action_dim, agent.device
                    )
                    agent._last_action = action

                is_committed = agent.e3._running_variance < agent.e3.commit_threshold
                is_elevated  = agent.beta_gate.is_elevated

                if is_committed and is_elevated:
                    committed_elevated += 1
                elif is_committed and not is_elevated:
                    committed_not_elev += 1
                elif not is_committed and is_elevated:
                    uncommitted_elevated += 1
                else:
                    uncommitted_not_elev += 1

            except Exception as exc:
                fatal += 1
                action = _action_to_onehot(
                    random.randint(0, env.action_dim - 1), env.action_dim, agent.device
                )
                agent._last_action = action

            flat_obs, _, done, _, obs_dict = env.step(action)
            if done:
                break

    gate_state = agent.beta_gate.get_state()
    n_committed   = committed_elevated + committed_not_elev
    n_uncommitted = uncommitted_elevated + uncommitted_not_elev

    committed_hold_conc       = committed_elevated   / max(1, n_committed)
    uncommitted_release_conc  = uncommitted_not_elev / max(1, n_uncommitted)
    mean_rv = _mean_safe(rvs)

    print(
        f"\n  [{label}] Gate concordance:"
        f"\n    committed steps: {n_committed}  uncommitted steps: {n_uncommitted}"
        f"\n    committed_hold_concordance:      {committed_hold_conc:.3f}"
        f"\n    uncommitted_release_concordance: {uncommitted_release_conc:.3f}"
        f"\n    hold_count={gate_state['hold_count']}"
        f"  propagation_count={gate_state['propagation_count']}"
        f"  mean_running_variance={mean_rv:.6f}",
        flush=True,
    )

    return {
        "committed_elevated":            committed_elevated,
        "committed_not_elevated":        committed_not_elev,
        "uncommitted_elevated":          uncommitted_elevated,
        "uncommitted_not_elevated":      uncommitted_not_elev,
        "n_committed_steps":             n_committed,
        "n_uncommitted_steps":           n_uncommitted,
        "committed_hold_concordance":    committed_hold_conc,
        "uncommitted_release_concordance": uncommitted_release_conc,
        "hold_count":                    gate_state["hold_count"],
        "propagation_count":             gate_state["propagation_count"],
        "mean_running_variance":         mean_rv,
        "fatal_errors":                  fatal,
    }


def run(
    seed: int = 0,
    warmup_episodes: int = 400,
    eval_episodes: int = 50,
    steps_per_episode: int = 200,
    alpha_world: float = 0.9,
    self_dim: int = 32,
    world_dim: int = 32,
    **kwargs,
) -> dict:
    torch.manual_seed(seed)
    random.seed(seed)

    print(
        f"[V3-EXQ-049e] MECH-090: Two-Condition Beta Gate Test\n"
        f"  Condition A: trained agent ({warmup_episodes} eps) → variance ~0 → committed\n"
        f"  Condition B: fresh agent (0 eps) → variance=precision_init > threshold → uncommitted\n"
        f"  C1 from Condition A, C2 from Condition B\n"
        f"  alpha_world={alpha_world}  seed={seed}",
        flush=True,
    )

    # ── Condition A: TRAINED agent ───────────────────────────────────────────
    print(f"\n{'='*60}", flush=True)
    print(f"[V3-EXQ-049e] Condition A — TRAINED ({warmup_episodes} episodes)", flush=True)
    print('='*60, flush=True)

    agent_trained, env_a = _make_agent_and_env(seed, self_dim, world_dim, alpha_world)

    print(
        f"  precision_init={agent_trained.e3._running_variance:.3f}"
        f"  commit_threshold={agent_trained.e3.commit_threshold:.3f}",
        flush=True,
    )
    train_out = _train_agent(
        agent_trained, env_a, warmup_episodes, steps_per_episode, world_dim
    )
    print(
        f"\n  Post-train: running_var={train_out['final_running_variance']:.6f}"
        f"  committed_frac={train_out['mean_committed_fraction']:.3f}",
        flush=True,
    )
    print(f"\n[V3-EXQ-049e] Eval Condition A ({eval_episodes} eps)...", flush=True)
    result_a = _eval_gate_concordance(
        agent_trained, env_a, eval_episodes, steps_per_episode, world_dim, label="trained"
    )

    # ── Condition B: FRESH agent ─────────────────────────────────────────────
    print(f"\n{'='*60}", flush=True)
    print(f"[V3-EXQ-049e] Condition B — FRESH (0 training episodes)", flush=True)
    print('='*60, flush=True)

    # Fresh agent: same config, no training
    agent_fresh, env_b = _make_agent_and_env(seed + 1000, self_dim, world_dim, alpha_world)

    print(
        f"  precision_init={agent_fresh.e3._running_variance:.3f}"
        f"  commit_threshold={agent_fresh.e3.commit_threshold:.3f}"
        f"  → uncommitted (variance > threshold)",
        flush=True,
    )
    print(f"\n[V3-EXQ-049e] Eval Condition B ({eval_episodes} eps)...", flush=True)
    result_b = _eval_gate_concordance(
        agent_fresh, env_b, eval_episodes, steps_per_episode, world_dim, label="fresh"
    )

    # ── PASS / FAIL ──────────────────────────────────────────────────────────
    c1_pass = result_a["committed_hold_concordance"]    > 0.6
    c2_pass = result_b["uncommitted_release_concordance"] > 0.5
    c3_pass = result_a["hold_count"]                   > 0
    c4_pass = result_b["propagation_count"]             > 0
    c5_pass = (result_a["fatal_errors"] + result_b["fatal_errors"]) == 0

    all_pass     = c1_pass and c2_pass and c3_pass and c4_pass and c5_pass
    status       = "PASS" if all_pass else "FAIL"
    criteria_met = sum([c1_pass, c2_pass, c3_pass, c4_pass, c5_pass])

    failure_notes = []
    if not c1_pass:
        failure_notes.append(
            f"C1 FAIL: trained committed_hold_concordance="
            f"{result_a['committed_hold_concordance']:.3f} <= 0.6"
        )
    if not c2_pass:
        failure_notes.append(
            f"C2 FAIL: fresh uncommitted_release_concordance="
            f"{result_b['uncommitted_release_concordance']:.3f} <= 0.5"
        )
    if not c3_pass:
        failure_notes.append(f"C3 FAIL: trained hold_count={result_a['hold_count']} == 0")
    if not c4_pass:
        failure_notes.append(f"C4 FAIL: fresh propagation_count={result_b['propagation_count']} == 0")
    if not c5_pass:
        failure_notes.append(
            f"C5 FAIL: fatal_errors trained={result_a['fatal_errors']}"
            f"  fresh={result_b['fatal_errors']}"
        )

    print(f"\nV3-EXQ-049e verdict: {status}  ({criteria_met}/5)", flush=True)
    for note in failure_notes:
        print(f"  {note}", flush=True)

    metrics = {
        # Condition A (trained)
        "trained_committed_steps":           float(result_a["n_committed_steps"]),
        "trained_uncommitted_steps":         float(result_a["n_uncommitted_steps"]),
        "trained_committed_hold_concordance":float(result_a["committed_hold_concordance"]),
        "trained_hold_count":                float(result_a["hold_count"]),
        "trained_propagation_count":         float(result_a["propagation_count"]),
        "trained_mean_running_variance":     float(result_a["mean_running_variance"]),
        "trained_final_variance_pretrain":   float(train_out["final_running_variance"]),
        "trained_mean_committed_frac_train": float(train_out["mean_committed_fraction"]),
        # Condition B (fresh)
        "fresh_committed_steps":             float(result_b["n_committed_steps"]),
        "fresh_uncommitted_steps":           float(result_b["n_uncommitted_steps"]),
        "fresh_uncommitted_release_concordance": float(result_b["uncommitted_release_concordance"]),
        "fresh_hold_count":                  float(result_b["hold_count"]),
        "fresh_propagation_count":           float(result_b["propagation_count"]),
        "fresh_mean_running_variance":       float(result_b["mean_running_variance"]),
        # Criteria
        "crit1_pass":    1.0 if c1_pass else 0.0,
        "crit2_pass":    1.0 if c2_pass else 0.0,
        "crit3_pass":    1.0 if c3_pass else 0.0,
        "crit4_pass":    1.0 if c4_pass else 0.0,
        "crit5_pass":    1.0 if c5_pass else 0.0,
        "criteria_met":  float(criteria_met),
        "fatal_error_count": float(result_a["fatal_errors"] + result_b["fatal_errors"]),
    }

    failure_section = ""
    if failure_notes:
        failure_section = "\n## Failure Notes\n\n" + "\n".join(f"- {n}" for n in failure_notes)

    summary_markdown = f"""# V3-EXQ-049e — MECH-090: Two-Condition Beta Gate Test

**Status:** {status}
**Claim:** MECH-090 — beta gate holds E3 policy output during committed action, releases when uncommitted
**Design:** Two-condition — trained agent (C1/C3) vs fresh agent (C2/C4)
**alpha_world:** {alpha_world}  |  **Warmup:** {warmup_episodes} eps  |  **Eval:** {eval_episodes} eps/condition  |  **Seed:** {seed}

## Design Rationale

Single-agent eval (EXQ-049d) cannot test C2: once trained, the agent is permanently
committed (variance ~0). A fresh agent starts at precision_init=0.5 > commit_threshold=0.40,
so it is persistently uncommitted without any artificial variance manipulation.
No hysteresis is needed: both conditions are at stable extremes of the variance distribution.

## Condition A — Trained Agent

| Metric | Value |
|--------|-------|
| Final running_variance (post-train) | {train_out['final_running_variance']:.6f} |
| Mean committed fraction (train) | {train_out['mean_committed_fraction']:.3f} |
| Committed steps (eval) | {result_a['n_committed_steps']} |
| Uncommitted steps (eval) | {result_a['n_uncommitted_steps']} |
| committed_hold_concordance | {result_a['committed_hold_concordance']:.3f} |
| hold_count | {result_a['hold_count']} |

## Condition B — Fresh Agent

| Metric | Value |
|--------|-------|
| precision_init (= running_variance) | {result_b['mean_running_variance']:.6f} |
| Committed steps (eval) | {result_b['n_committed_steps']} |
| Uncommitted steps (eval) | {result_b['n_uncommitted_steps']} |
| uncommitted_release_concordance | {result_b['uncommitted_release_concordance']:.3f} |
| propagation_count | {result_b['propagation_count']} |

## PASS Criteria

| Criterion | Source | Result | Value |
|---|---|---|---|
| C1: trained committed_hold_concordance > 0.6 | Cond A | {"PASS" if c1_pass else "FAIL"} | {result_a['committed_hold_concordance']:.3f} |
| C2: fresh uncommitted_release_concordance > 0.5 | Cond B | {"PASS" if c2_pass else "FAIL"} | {result_b['uncommitted_release_concordance']:.3f} |
| C3: trained hold_count > 0 | Cond A | {"PASS" if c3_pass else "FAIL"} | {result_a['hold_count']} |
| C4: fresh propagation_count > 0 | Cond B | {"PASS" if c4_pass else "FAIL"} | {result_b['propagation_count']} |
| C5: No fatal errors | Both | {"PASS" if c5_pass else "FAIL"} | {result_a['fatal_errors'] + result_b['fatal_errors']} |

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
        "fatal_error_count": float(result_a["fatal_errors"] + result_b["fatal_errors"]),
    }


if __name__ == "__main__":
    import argparse
    import json
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",        type=int,   default=0)
    parser.add_argument("--warmup",      type=int,   default=400)
    parser.add_argument("--eval-eps",    type=int,   default=50)
    parser.add_argument("--steps",       type=int,   default=200)
    parser.add_argument("--alpha-world", type=float, default=0.9)
    args = parser.parse_args()

    result = run(
        seed=args.seed,
        warmup_episodes=args.warmup,
        eval_episodes=args.eval_eps,
        steps_per_episode=args.steps,
        alpha_world=args.alpha_world,
    )

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result["run_timestamp"] = ts
    result["claim"]              = CLAIM_IDS[0]
    result["verdict"]            = result["status"]
    result["run_id"]             = f"{EXPERIMENT_TYPE}_{ts}_v3"
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
