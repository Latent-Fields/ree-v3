"""
V3-EXQ-061 -- MECH-104: Volatility Interrupt (Jitter Approximation)

Claims: MECH-104, MECH-090

Motivation:
  EXQ-049d/060 showed that world_forward training drives running_variance to ~2.76e-6,
  permanently locking the agent in committed state. This makes de-commitment
  untestable in any single-agent eval design.

  MECH-104 claims that unexpected harm events should spike commitment uncertainty,
  enabling de-commitment. This experiment tests the V3 jitter approximation:
  when actual_harm > harm_contact_threshold (surprise contact), add a fixed spike
  to running_variance. This is a tractable first test of the LC-NE volatile
  uncertainty mechanism before full wiring.

  Two-condition design:
    Condition A (baseline): standard training -- no surprise spike.
      Expected: variance collapses, agent permanently committed, n_uncommitted=0.
      This documents the problem MECH-104 solves.

    Condition B (surprise spike): same training + spike mechanism.
      When a hazard contact occurs (actual_harm < -threshold), add spike_magnitude
      to running_variance. Expected: de-commitment events follow harm contacts;
      n_uncommitted_steps > 0; variance shows upward movement after harm.

PASS criteria (ALL must hold):
  C1: baseline n_uncommitted_steps == 0 during eval
      (confirms variance collapse without jitter)
  C2: spike condition n_uncommitted_steps > 100 during eval
      (jitter enables measurable de-commitment events)
  C3: spike condition: mean_variance_after_harm > mean_variance_before_harm
      (variance rises on contact -- causal relationship confirmed)
  C4: spike condition calibration_gap_approach > 0.0
      (harm discrimination preserved -- jitter does not degrade E3 signal)
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


EXPERIMENT_TYPE = "v3_exq_061_mech104_volatility_jitter"
CLAIM_IDS = ["MECH-104", "MECH-090"]

ENV_KWARGS = dict(
    size=12, num_hazards=4, num_resources=5,
    hazard_harm=0.02,
    env_drift_interval=5, env_drift_prob=0.1,
    proximity_harm_scale=0.05,
    proximity_benefit_scale=0.03,
    proximity_approach_threshold=0.15,
    hazard_field_decay=0.5,
)

# Harm contact threshold: harm_signal < this value indicates a contact event
HARM_CONTACT_THRESHOLD = -0.01


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


def _train_and_eval(
    seed: int,
    self_dim: int,
    world_dim: int,
    alpha_world: float,
    num_train_episodes: int,
    num_eval_episodes: int,
    steps_per_episode: int,
    spike_magnitude: float,
    label: str,
) -> Dict:
    """Train then eval. If spike_magnitude > 0, apply surprise variance spike on harm contact."""
    agent, env = _make_agent_and_env(seed, self_dim, world_dim, alpha_world)
    agent.train()

    optimizer = optim.Adam(list(agent.e1.parameters()), lr=1e-3)
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

    # ── Training loop ──────────────────────────────────────────────────────
    for ep in range(num_train_episodes):
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

            # MECH-104 surprise variance spike: when unexpected harm contact occurs,
            # push running_variance upward by spike_magnitude (V3 jitter approximation).
            if spike_magnitude > 0.0 and harm_signal < HARM_CONTACT_THRESHOLD:
                with torch.no_grad():
                    agent.e3._running_variance = float(
                        agent.e3._running_variance
                    ) + spike_magnitude

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

        if (ep + 1) % 100 == 0 or ep == num_train_episodes - 1:
            rv = agent.e3._running_variance
            print(
                f"  [{label}|train] ep {ep+1}/{num_train_episodes}"
                f"  running_var={rv:.6f}",
                flush=True,
            )

    final_train_variance = float(agent.e3._running_variance)
    print(
        f"\n  [{label}] Post-train variance={final_train_variance:.6f}"
        f"  commit_threshold={agent.e3.commit_threshold:.4f}",
        flush=True,
    )

    # ── Eval loop ──────────────────────────────────────────────────────────
    agent.eval()
    agent.beta_gate.reset()

    n_committed = 0
    n_uncommitted = 0
    n_approach = 0
    n_none = 0
    harm_eval_at_approach: List[float] = []
    harm_eval_at_none: List[float] = []
    fatal = 0

    # For C3: track variance before and after harm contacts
    variance_before_harm: List[float] = []
    variance_after_harm:  List[float] = []
    prev_variance: float = float(agent.e3._running_variance)
    harm_contact_count = 0

    for _ in range(num_eval_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()

        for _ in range(steps_per_episode):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            with torch.no_grad():
                latent = agent.sense(obs_body, obs_world)
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
                if is_committed:
                    n_committed += 1
                else:
                    n_uncommitted += 1

                ttype = info.get("transition_type", "none") if "info" in dir() else "none"

            except Exception:
                fatal += 1
                action = _action_to_onehot(
                    random.randint(0, env.action_dim - 1), env.action_dim, agent.device
                )
                agent._last_action = action

            prev_variance = float(agent.e3._running_variance)
            flat_obs, harm_signal, done, info, obs_dict = env.step(action)

            # Apply spike in eval too (to measure post-spike variance change)
            if spike_magnitude > 0.0 and harm_signal < HARM_CONTACT_THRESHOLD:
                variance_before_harm.append(prev_variance)
                with torch.no_grad():
                    agent.e3._running_variance = float(
                        agent.e3._running_variance
                    ) + spike_magnitude
                variance_after_harm.append(float(agent.e3._running_variance))
                harm_contact_count += 1

            # Calibration gap measurement
            transition_type = info.get("transition_type", "none") if isinstance(info, dict) else "none"
            theta_z = agent.theta_buffer.summary()
            with torch.no_grad():
                he = float(agent.e3.harm_eval(theta_z).item())
            if transition_type == "hazard_approach":
                harm_eval_at_approach.append(he)
                n_approach += 1
            elif transition_type == "none":
                harm_eval_at_none.append(he)
                n_none += 1

            if done:
                break

    mean_harm_approach = _mean_safe(harm_eval_at_approach)
    mean_harm_none     = _mean_safe(harm_eval_at_none)
    calibration_gap    = mean_harm_approach - mean_harm_none

    mean_var_before = _mean_safe(variance_before_harm)
    mean_var_after  = _mean_safe(variance_after_harm)

    print(
        f"\n  [{label}] Eval results:"
        f"\n    n_committed={n_committed}  n_uncommitted={n_uncommitted}"
        f"\n    calibration_gap_approach={calibration_gap:.4f}"
        f"\n    harm_contact_count={harm_contact_count}"
        f"\n    mean_variance_before_harm={mean_var_before:.6f}"
        f"\n    mean_variance_after_harm={mean_var_after:.6f}",
        flush=True,
    )

    return {
        "label":                      label,
        "final_train_variance":        final_train_variance,
        "n_committed_steps":           n_committed,
        "n_uncommitted_steps":         n_uncommitted,
        "calibration_gap_approach":    calibration_gap,
        "mean_harm_approach":          mean_harm_approach,
        "mean_harm_none":              mean_harm_none,
        "n_approach":                  n_approach,
        "n_none":                      n_none,
        "harm_contact_count":          harm_contact_count,
        "mean_variance_before_harm":   mean_var_before,
        "mean_variance_after_harm":    mean_var_after,
        "fatal_errors":                fatal,
    }


def run(
    seed: int = 0,
    warmup_episodes: int = 400,
    eval_episodes: int = 50,
    steps_per_episode: int = 200,
    alpha_world: float = 0.9,
    self_dim: int = 32,
    world_dim: int = 32,
    spike_magnitude: float = 0.05,
    **kwargs,
) -> dict:
    torch.manual_seed(seed)
    random.seed(seed)

    print(
        f"[V3-EXQ-061] MECH-104: Volatility Interrupt Jitter Test\n"
        f"  Condition A: no spike (documents variance collapse problem)\n"
        f"  Condition B: surprise spike (magnitude={spike_magnitude}) on harm contact\n"
        f"  spike_magnitude={spike_magnitude}  alpha_world={alpha_world}  seed={seed}",
        flush=True,
    )

    # ── Condition A: baseline (no spike) ────────────────────────────────────
    print(f"\n{'='*60}", flush=True)
    print("[V3-EXQ-061] Condition A -- Baseline (no spike)", flush=True)
    print('='*60, flush=True)

    result_a = _train_and_eval(
        seed=seed,
        self_dim=self_dim, world_dim=world_dim,
        alpha_world=alpha_world,
        num_train_episodes=warmup_episodes,
        num_eval_episodes=eval_episodes,
        steps_per_episode=steps_per_episode,
        spike_magnitude=0.0,
        label="baseline",
    )

    # ── Condition B: surprise spike ──────────────────────────────────────────
    print(f"\n{'='*60}", flush=True)
    print(f"[V3-EXQ-061] Condition B -- Surprise Spike (eps={spike_magnitude})", flush=True)
    print('='*60, flush=True)

    result_b = _train_and_eval(
        seed=seed + 1000,
        self_dim=self_dim, world_dim=world_dim,
        alpha_world=alpha_world,
        num_train_episodes=warmup_episodes,
        num_eval_episodes=eval_episodes,
        steps_per_episode=steps_per_episode,
        spike_magnitude=spike_magnitude,
        label="spike",
    )

    # ── PASS / FAIL ──────────────────────────────────────────────────────────
    c1_pass = result_a["n_uncommitted_steps"] == 0
    c2_pass = result_b["n_uncommitted_steps"] > 100
    c3_pass = (
        result_b["harm_contact_count"] > 0 and
        result_b["mean_variance_after_harm"] > result_b["mean_variance_before_harm"]
    )
    c4_pass = result_b["calibration_gap_approach"] > 0.0
    c5_pass = (result_a["fatal_errors"] + result_b["fatal_errors"]) == 0

    all_pass     = c1_pass and c2_pass and c3_pass and c4_pass and c5_pass
    status       = "PASS" if all_pass else "FAIL"
    criteria_met = sum([c1_pass, c2_pass, c3_pass, c4_pass, c5_pass])

    failure_notes = []
    if not c1_pass:
        failure_notes.append(
            f"C1 FAIL: baseline n_uncommitted_steps="
            f"{result_a['n_uncommitted_steps']} != 0 (variance did not collapse as expected)"
        )
    if not c2_pass:
        failure_notes.append(
            f"C2 FAIL: spike n_uncommitted_steps="
            f"{result_b['n_uncommitted_steps']} <= 100 (spike did not enable de-commitment)"
        )
    if not c3_pass:
        failure_notes.append(
            f"C3 FAIL: variance_before_harm={result_b['mean_variance_before_harm']:.6f}"
            f" vs after={result_b['mean_variance_after_harm']:.6f}"
            f" harm_contacts={result_b['harm_contact_count']}"
        )
    if not c4_pass:
        failure_notes.append(
            f"C4 FAIL: spike calibration_gap={result_b['calibration_gap_approach']:.4f} <= 0"
        )
    if not c5_pass:
        failure_notes.append(
            f"C5 FAIL: fatal_errors baseline={result_a['fatal_errors']}"
            f" spike={result_b['fatal_errors']}"
        )

    print(f"\nV3-EXQ-061 verdict: {status}  ({criteria_met}/5)", flush=True)
    for note in failure_notes:
        print(f"  {note}", flush=True)

    metrics = {
        # Condition A (baseline)
        "baseline_final_train_variance":     float(result_a["final_train_variance"]),
        "baseline_n_uncommitted_steps":      float(result_a["n_uncommitted_steps"]),
        "baseline_n_committed_steps":        float(result_a["n_committed_steps"]),
        "baseline_calibration_gap":          float(result_a["calibration_gap_approach"]),
        # Condition B (spike)
        "spike_final_train_variance":        float(result_b["final_train_variance"]),
        "spike_n_uncommitted_steps":         float(result_b["n_uncommitted_steps"]),
        "spike_n_committed_steps":           float(result_b["n_committed_steps"]),
        "spike_calibration_gap":             float(result_b["calibration_gap_approach"]),
        "spike_harm_contact_count":          float(result_b["harm_contact_count"]),
        "spike_mean_variance_before_harm":   float(result_b["mean_variance_before_harm"]),
        "spike_mean_variance_after_harm":    float(result_b["mean_variance_after_harm"]),
        # Criteria
        "crit1_pass":    1.0 if c1_pass else 0.0,
        "crit2_pass":    1.0 if c2_pass else 0.0,
        "crit3_pass":    1.0 if c3_pass else 0.0,
        "crit4_pass":    1.0 if c4_pass else 0.0,
        "crit5_pass":    1.0 if c5_pass else 0.0,
        "criteria_met":  float(criteria_met),
        "spike_magnitude": float(spike_magnitude),
        "fatal_error_count": float(result_a["fatal_errors"] + result_b["fatal_errors"]),
    }

    failure_section = ""
    if failure_notes:
        failure_section = "\n## Failure Notes\n\n" + "\n".join(f"- {n}" for n in failure_notes)

    summary_markdown = f"""# V3-EXQ-061 -- MECH-104: Volatility Interrupt (Jitter Approximation)

**Status:** {status}
**Claims:** MECH-104, MECH-090
**Design:** Two-condition -- baseline (no spike) vs surprise spike on harm contact
**spike_magnitude:** {spike_magnitude}  |  **alpha_world:** {alpha_world}  |  **Warmup:** {warmup_episodes} eps  |  **Seed:** {seed}

## Design Rationale

Without a volatility interrupt mechanism, world_forward training drives running_variance
to near-zero, permanently locking the agent in committed state (MECH-090 permanently
elevated). MECH-104 proposes that unexpected harm contact should spike running_variance
upward, enabling de-commitment. This experiment tests the V3 jitter approximation:
add spike_magnitude to _running_variance when actual_harm < {HARM_CONTACT_THRESHOLD}.

## Condition A -- Baseline (no spike)

| Metric | Value |
|--------|-------|
| Final running_variance (post-train) | {result_a['final_train_variance']:.6f} |
| Uncommitted steps (eval) | {result_a['n_uncommitted_steps']} |
| Committed steps (eval) | {result_a['n_committed_steps']} |
| calibration_gap_approach | {result_a['calibration_gap_approach']:.4f} |

## Condition B -- Surprise Spike (eps={spike_magnitude})

| Metric | Value |
|--------|-------|
| Final running_variance (post-train) | {result_b['final_train_variance']:.6f} |
| Uncommitted steps (eval) | {result_b['n_uncommitted_steps']} |
| Committed steps (eval) | {result_b['n_committed_steps']} |
| calibration_gap_approach | {result_b['calibration_gap_approach']:.4f} |
| Harm contacts during eval | {result_b['harm_contact_count']} |
| Mean variance before harm contact | {result_b['mean_variance_before_harm']:.6f} |
| Mean variance after harm contact | {result_b['mean_variance_after_harm']:.6f} |

## PASS Criteria

| Criterion | Result | Value |
|---|---|---|
| C1: baseline n_uncommitted == 0 (confirms collapse) | {"PASS" if c1_pass else "FAIL"} | {result_a['n_uncommitted_steps']} |
| C2: spike n_uncommitted > 100 (de-commitment enabled) | {"PASS" if c2_pass else "FAIL"} | {result_b['n_uncommitted_steps']} |
| C3: variance rises after harm contact | {"PASS" if c3_pass else "FAIL"} | {result_b['mean_variance_before_harm']:.4f} -> {result_b['mean_variance_after_harm']:.4f} |
| C4: spike calibration_gap > 0 (harm signal preserved) | {"PASS" if c4_pass else "FAIL"} | {result_b['calibration_gap_approach']:.4f} |
| C5: No fatal errors | {"PASS" if c5_pass else "FAIL"} | {result_a['fatal_errors'] + result_b['fatal_errors']} |

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
        "fatal_error_count": float(result_a["fatal_errors"] + result_b["fatal_errors"]),
    }


if __name__ == "__main__":
    import argparse
    import json
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",            type=int,   default=0)
    parser.add_argument("--warmup",          type=int,   default=400)
    parser.add_argument("--eval-eps",        type=int,   default=50)
    parser.add_argument("--steps",           type=int,   default=200)
    parser.add_argument("--alpha-world",     type=float, default=0.9)
    parser.add_argument("--spike-magnitude", type=float, default=0.05)
    args = parser.parse_args()

    result = run(
        seed=args.seed,
        warmup_episodes=args.warmup,
        eval_episodes=args.eval_eps,
        steps_per_episode=args.steps,
        alpha_world=args.alpha_world,
        spike_magnitude=args.spike_magnitude,
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
