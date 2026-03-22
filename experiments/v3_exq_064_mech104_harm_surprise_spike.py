"""
V3-EXQ-064 -- MECH-104: Harm-Surprise Spike (Route-2 Validation)

Claims: MECH-104

Context:
  MECH-104: unexpected harm events should spike commitment uncertainty
  (LC-NE volatility interrupt), enabling de-commitment.

  EXQ-049e PASS: Route-1 validated -- noise floor (jitter) prevents permanent
  variance collapse and enables de-commitment.
  EXQ-061  PASS: Route-1 validated -- harm-contact spike (any harm) triggers
  variance increase; jitter approximation works.

  Route-2 (this experiment): the spike is SELECTIVE -- it fires only on
  UNEXPECTED harm events (|actual_harm - predicted_harm| > surprise_threshold).
  Expected harm does not trigger a spike because the agent already anticipated
  it; the spike should carry new information. This is the proper LC-NE mechanism:
  norepinephrine release is driven by unexpected outcomes, not routine harm.

  EXQ-062a/062b tested the SELECTIVITY of the surprise gate (fewer spikes than
  always-spike). This experiment tests the SIGNAL itself: does the surprise gate
  produce measurably higher variance on unexpected harm events than on expected
  ones, and compared to a no-spike baseline?

Design -- two-condition:
  Single trained agent (400 eps, collapsed variance). Eval under:

  Condition A -- SURPRISE-GATED SPIKE (Route-2)
    When committed AND actual_harm < -0.01:
      predicted_harm = agent.e3.harm_eval(theta_z)  [evaluated before env step]
      surprise = |actual_harm - predicted_harm|
      if surprise > surprise_threshold:
        running_variance += spike_magnitude * (surprise - surprise_threshold)
    Harm steps are classified as:
      "unexpected" if surprise > surprise_threshold (spike fires)
      "expected"   if surprise <= surprise_threshold (no spike)

  Condition B -- NO-SPIKE BASELINE
    Same trained agent, no harm-triggered variance change.
    Records same surprise values to confirm the two conditions see similar events.

PASS criteria (ALL must hold):
  C1: Condition A mean_variance_after_unexpected_harm > mean_variance_before_unexpected_harm + 0.005
      (surprise gate actually spikes variance on unexpected harm)
  C2: Condition A mean_variance_after_expected_harm - mean_variance_before_expected_harm < 0.002
      (gate does not spike on expected harm -- selectivity confirmed)
  C3: Condition B mean_variance_after_unexpected_harm - mean_variance_before_unexpected_harm < 0.002
      (baseline stays flat on unexpected harm -- no spurious spikes without gate)
  C4: n_unexpected_harm_events_A >= 10
      (enough unexpected harm events to make measurements meaningful)
  C5: No fatal errors
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


EXPERIMENT_TYPE = "v3_exq_064_mech104_harm_surprise_spike"
CLAIM_IDS = ["MECH-104"]

HARM_CONTACT_THRESHOLD = -0.01

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
    return REEAgent(config), env


def _train_agent(
    agent: REEAgent,
    env: CausalGridWorldV2,
    num_episodes: int,
    steps_per_episode: int,
    world_dim: int,
) -> Dict:
    """Train agent until running_variance collapses (committed state)."""
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

    for ep in range(num_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()
        z_world_prev: Optional[torch.Tensor] = None
        action_prev: Optional[torch.Tensor] = None

        for _ in range(steps_per_episode):
            obs_body = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            latent = agent.sense(obs_body, obs_world)

            ticks = agent.clock.advance()
            e1_prior = (
                agent._e1_tick(latent) if ticks.get("e1_tick", False)
                else torch.zeros(1, world_dim, device=agent.device)
            )
            candidates = agent.generate_trajectories(latent, e1_prior, ticks)
            theta_z = agent.theta_buffer.summary()
            z_world_curr = latent.z_world.detach()

            action = agent.select_action(candidates, ticks, temperature=1.0)
            if action is None:
                action = _action_to_onehot(
                    random.randint(0, env.action_dim - 1), env.action_dim, agent.device
                )
                agent._last_action = action

            flat_obs, harm_signal, done, info, obs_dict = env.step(action)

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
                zw_b = torch.cat([wf_buf[i][0] for i in idxs]).to(agent.device)
                a_b = torch.cat([wf_buf[i][1] for i in idxs]).to(agent.device)
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
            action_prev = action.detach()
            if done:
                break

        if (ep + 1) % 100 == 0 or ep == num_episodes - 1:
            rv = agent.e3._running_variance
            print(
                f"  [train] ep {ep+1}/{num_episodes}"
                f"  running_var={rv:.6f}",
                flush=True,
            )

    return {"final_running_variance": float(agent.e3._running_variance)}


def _eval_surprise_gate(
    agent: REEAgent,
    env: CausalGridWorldV2,
    num_episodes: int,
    steps_per_episode: int,
    world_dim: int,
    spike_magnitude: float,
    surprise_threshold: float,
    use_spike: bool,
    label: str,
    initial_variance: float,
) -> Dict:
    """Eval surprise-gate behavior on unexpected vs expected harm events.

    For each harm step, records variance before and after, and classifies
    the event as 'unexpected' (surprise > threshold) or 'expected' (surprise <= threshold).

    If use_spike=True: applies Route-2 spike on committed unexpected harm steps.
    If use_spike=False: baseline -- no harm-triggered variance change.
    """
    agent.eval()

    # Variance measurements before/after harm events, split by type
    var_before_unexpected: List[float] = []
    var_after_unexpected: List[float] = []
    var_before_expected: List[float] = []
    var_after_expected: List[float] = []

    n_surprise_spikes = 0
    n_committed_harm = 0
    n_uncommitted_harm = 0
    fatal = 0

    for _ in range(num_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()
        agent.e3._running_variance = initial_variance

        for _ in range(steps_per_episode):
            obs_body = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            with torch.no_grad():
                latent = agent.sense(obs_body, obs_world)

            # Query predicted harm BEFORE env step
            with torch.no_grad():
                theta_z = agent.theta_buffer.summary()
                predicted_harm = float(agent.e3.harm_eval(theta_z).item())

            # Commitment state BEFORE select
            is_committed_pre = agent.e3._running_variance < agent.e3.commit_threshold
            variance_pre = float(agent.e3._running_variance)

            try:
                with torch.no_grad():
                    ticks = agent.clock.advance()
                    e1_prior = (
                        agent._e1_tick(latent) if ticks.get("e1_tick", False)
                        else torch.zeros(1, world_dim, device=agent.device)
                    )
                    candidates = agent.generate_trajectories(latent, e1_prior, ticks)
                    action = agent.select_action(candidates, ticks, temperature=1.0)

                if action is None:
                    action = _action_to_onehot(
                        random.randint(0, env.action_dim - 1), env.action_dim, agent.device
                    )
                    agent._last_action = action

                flat_obs, actual_harm, done, info, obs_dict = env.step(action)

                # Process harm events
                if actual_harm < HARM_CONTACT_THRESHOLD:
                    surprise = abs(actual_harm - predicted_harm)
                    is_unexpected = surprise > surprise_threshold

                    if is_committed_pre:
                        n_committed_harm += 1
                        if use_spike and is_unexpected:
                            impulse = spike_magnitude * (surprise - surprise_threshold)
                            if impulse > 0:
                                agent.e3._running_variance += impulse
                                n_surprise_spikes += 1
                    else:
                        n_uncommitted_harm += 1

                    variance_post = float(agent.e3._running_variance)

                    if is_unexpected:
                        var_before_unexpected.append(variance_pre)
                        var_after_unexpected.append(variance_post)
                    else:
                        var_before_expected.append(variance_pre)
                        var_after_expected.append(variance_post)

            except Exception:
                fatal += 1
                flat_obs, _, done, info, obs_dict = env.reset()

            if done:
                break

    mean_var_before_unexpected = _mean_safe(var_before_unexpected)
    mean_var_after_unexpected = _mean_safe(var_after_unexpected)
    mean_var_before_expected = _mean_safe(var_before_expected)
    mean_var_after_expected = _mean_safe(var_after_expected)

    delta_unexpected = mean_var_after_unexpected - mean_var_before_unexpected
    delta_expected = mean_var_after_expected - mean_var_before_expected

    print(
        f"\n  [{label}] use_spike={use_spike}"
        f"\n    n_unexpected_harm={len(var_before_unexpected)}"
        f"  n_expected_harm={len(var_before_expected)}"
        f"  n_surprise_spikes={n_surprise_spikes}"
        f"\n    delta_var_unexpected={delta_unexpected:.6f}"
        f"  delta_var_expected={delta_expected:.6f}"
        f"\n    var_before_unexpected={mean_var_before_unexpected:.6f}"
        f"  var_after_unexpected={mean_var_after_unexpected:.6f}"
        f"\n    var_before_expected={mean_var_before_expected:.6f}"
        f"  var_after_expected={mean_var_after_expected:.6f}"
        f"\n    committed_harm_steps={n_committed_harm}"
        f"  uncommitted_harm_steps={n_uncommitted_harm}"
        f"  fatal={fatal}",
        flush=True,
    )

    return {
        "n_unexpected_harm": len(var_before_unexpected),
        "n_expected_harm": len(var_before_expected),
        "n_surprise_spikes": n_surprise_spikes,
        "mean_var_before_unexpected": mean_var_before_unexpected,
        "mean_var_after_unexpected": mean_var_after_unexpected,
        "delta_var_unexpected": delta_unexpected,
        "mean_var_before_expected": mean_var_before_expected,
        "mean_var_after_expected": mean_var_after_expected,
        "delta_var_expected": delta_expected,
        "n_committed_harm": n_committed_harm,
        "n_uncommitted_harm": n_uncommitted_harm,
        "fatal_errors": fatal,
    }


def run(
    seed: int = 0,
    warmup_episodes: int = 400,
    eval_episodes: int = 50,
    steps_per_episode: int = 200,
    alpha_world: float = 0.9,
    spike_magnitude: float = 0.05,
    surprise_threshold: float = 0.02,
    self_dim: int = 32,
    world_dim: int = 32,
    **kwargs,
) -> dict:
    torch.manual_seed(seed)
    random.seed(seed)

    print(
        f"[V3-EXQ-064] MECH-104: Harm-Surprise Spike (Route-2 Validation)\n"
        f"  Route-2: spike on UNEXPECTED harm (|actual - predicted| > threshold)\n"
        f"  Condition A: surprise-gated spike (Route-2)\n"
        f"  Condition B: no-spike baseline\n"
        f"  spike_magnitude={spike_magnitude}  surprise_threshold={surprise_threshold}\n"
        f"  alpha_world={alpha_world}  seed={seed}",
        flush=True,
    )

    # -- Train agent ----------------------------------------------------------
    print(f"\n{'='*60}", flush=True)
    print(f"[V3-EXQ-064] Training ({warmup_episodes} eps)...", flush=True)
    print('='*60, flush=True)
    agent, env = _make_agent_and_env(seed, self_dim, world_dim, alpha_world)
    train_out = _train_agent(agent, env, warmup_episodes, steps_per_episode, world_dim)

    initial_variance = train_out["final_running_variance"]
    print(
        f"\n  Post-train running_variance={initial_variance:.6f}"
        f"  commit_threshold={agent.e3.commit_threshold:.4f}"
        f"  committed={initial_variance < agent.e3.commit_threshold}",
        flush=True,
    )

    if initial_variance >= agent.e3.commit_threshold:
        print(
            "  WARNING: agent not in committed state after training. "
            "Surprise gate will not fire (only fires on committed steps).",
            flush=True,
        )

    # -- Condition A: surprise-gated spike ------------------------------------
    print(f"\n{'='*60}", flush=True)
    print("[V3-EXQ-064] Condition A -- Surprise-Gated Spike (Route-2)", flush=True)
    print('='*60, flush=True)
    result_a = _eval_surprise_gate(
        agent=agent,
        env=env,
        num_episodes=eval_episodes,
        steps_per_episode=steps_per_episode,
        world_dim=world_dim,
        spike_magnitude=spike_magnitude,
        surprise_threshold=surprise_threshold,
        use_spike=True,
        label="Condition A (surprise-gated)",
        initial_variance=initial_variance,
    )

    # -- Condition B: no-spike baseline ---------------------------------------
    print(f"\n{'='*60}", flush=True)
    print("[V3-EXQ-064] Condition B -- No-Spike Baseline", flush=True)
    print('='*60, flush=True)
    result_b = _eval_surprise_gate(
        agent=agent,
        env=env,
        num_episodes=eval_episodes,
        steps_per_episode=steps_per_episode,
        world_dim=world_dim,
        spike_magnitude=spike_magnitude,
        surprise_threshold=surprise_threshold,
        use_spike=False,
        label="Condition B (no-spike baseline)",
        initial_variance=initial_variance,
    )

    # -- PASS / FAIL ----------------------------------------------------------
    # C1: spike fires and raises variance on unexpected harm in A
    c1_pass = result_a["delta_var_unexpected"] > 0.005

    # C2: spike does NOT fire on expected harm in A (selective)
    c2_pass = result_a["delta_var_expected"] < 0.002

    # C3: baseline stays flat on unexpected harm in B
    c3_pass = result_b["delta_var_unexpected"] < 0.002

    # C4: enough unexpected harm events for meaningful measurements
    c4_pass = result_a["n_unexpected_harm"] >= 10

    # C5: no fatal errors
    c5_pass = (result_a["fatal_errors"] + result_b["fatal_errors"]) == 0

    all_pass = c1_pass and c2_pass and c3_pass and c4_pass and c5_pass
    status = "PASS" if all_pass else "FAIL"
    criteria_met = sum([c1_pass, c2_pass, c3_pass, c4_pass, c5_pass])

    failure_notes = []
    if not c1_pass:
        failure_notes.append(
            f"C1 FAIL: delta_var_unexpected_A={result_a['delta_var_unexpected']:.6f} <= 0.005 "
            f"(surprise gate not spiking variance on unexpected harm; "
            f"n_spikes={result_a['n_surprise_spikes']})"
        )
    if not c2_pass:
        failure_notes.append(
            f"C2 FAIL: delta_var_expected_A={result_a['delta_var_expected']:.6f} >= 0.002 "
            f"(gate firing on expected harm -- not selective)"
        )
    if not c3_pass:
        failure_notes.append(
            f"C3 FAIL: delta_var_unexpected_B={result_b['delta_var_unexpected']:.6f} >= 0.002 "
            f"(baseline not flat on unexpected harm -- spurious spikes)"
        )
    if not c4_pass:
        failure_notes.append(
            f"C4 FAIL: n_unexpected_harm_A={result_a['n_unexpected_harm']} < 10 "
            f"(insufficient unexpected harm events; surprise_threshold may be too high)"
        )
    if not c5_pass:
        failure_notes.append(
            f"C5 FAIL: fatal_errors A={result_a['fatal_errors']} B={result_b['fatal_errors']}"
        )

    print(f"\nV3-EXQ-064 verdict: {status}  ({criteria_met}/5)", flush=True)
    for note in failure_notes:
        print(f"  {note}", flush=True)

    metrics = {
        # Condition A (surprise-gated spike)
        "a_n_unexpected_harm":          float(result_a["n_unexpected_harm"]),
        "a_n_expected_harm":            float(result_a["n_expected_harm"]),
        "a_n_surprise_spikes":          float(result_a["n_surprise_spikes"]),
        "a_delta_var_unexpected":       float(result_a["delta_var_unexpected"]),
        "a_delta_var_expected":         float(result_a["delta_var_expected"]),
        "a_var_before_unexpected":      float(result_a["mean_var_before_unexpected"]),
        "a_var_after_unexpected":       float(result_a["mean_var_after_unexpected"]),
        "a_var_before_expected":        float(result_a["mean_var_before_expected"]),
        "a_var_after_expected":         float(result_a["mean_var_after_expected"]),
        "a_n_committed_harm":           float(result_a["n_committed_harm"]),
        # Condition B (baseline)
        "b_n_unexpected_harm":          float(result_b["n_unexpected_harm"]),
        "b_n_expected_harm":            float(result_b["n_expected_harm"]),
        "b_delta_var_unexpected":       float(result_b["delta_var_unexpected"]),
        "b_delta_var_expected":         float(result_b["delta_var_expected"]),
        "b_var_before_unexpected":      float(result_b["mean_var_before_unexpected"]),
        "b_var_after_unexpected":       float(result_b["mean_var_after_unexpected"]),
        # Parameters
        "train_final_variance":         float(initial_variance),
        "spike_magnitude":              float(spike_magnitude),
        "surprise_threshold":           float(surprise_threshold),
        # Criteria
        "crit1_pass":     1.0 if c1_pass else 0.0,
        "crit2_pass":     1.0 if c2_pass else 0.0,
        "crit3_pass":     1.0 if c3_pass else 0.0,
        "crit4_pass":     1.0 if c4_pass else 0.0,
        "crit5_pass":     1.0 if c5_pass else 0.0,
        "criteria_met":   float(criteria_met),
        "fatal_error_count": float(result_a["fatal_errors"] + result_b["fatal_errors"]),
    }

    failure_section = ""
    if failure_notes:
        failure_section = "\n## Failure Notes\n\n" + "\n".join(f"- {n}" for n in failure_notes)

    summary_markdown = f"""# V3-EXQ-064 -- MECH-104: Harm-Surprise Spike (Route-2 Validation)

**Status:** {status}
**Claims:** MECH-104
**Design:** Two-condition -- Condition A: surprise-gated spike (Route-2) vs Condition B: no-spike baseline
**spike_magnitude:** {spike_magnitude}  |  **surprise_threshold:** {surprise_threshold}  |  **Warmup:** {warmup_episodes} eps  |  **Eval:** {eval_episodes} eps  |  **Seed:** {seed}

## Design Rationale

Route-2 MECH-104: the surprise gate fires selectively on UNEXPECTED harm events
(|actual_harm - predicted_harm| > surprise_threshold). Expected harm does not trigger
a spike because the agent already anticipated it -- no new information, no LC-NE release.

Prior validation:
- EXQ-049e PASS: Route-1 (jitter noise floor) prevents variance collapse.
- EXQ-061  PASS: Route-1 (any harm contact) triggers variance spike.
- EXQ-062b tests SELECTIVITY (fewer spikes than always-gate).
- This experiment tests the SIGNAL: variance rises specifically on unexpected harm steps.

## Training

| Metric | Value |
|--------|-------|
| Final running_variance (post-train) | {initial_variance:.6f} |
| Committed? (variance < {agent.e3.commit_threshold:.4f}) | {"Yes" if initial_variance < agent.e3.commit_threshold else "No"} |

## Variance Response to Harm Events

| | Condition A (surprise-gated) | Condition B (baseline) |
|---|---|---|
| n_unexpected_harm | {result_a['n_unexpected_harm']} | {result_b['n_unexpected_harm']} |
| n_expected_harm | {result_a['n_expected_harm']} | {result_b['n_expected_harm']} |
| n_surprise_spikes | {result_a['n_surprise_spikes']} | 0 |
| delta_var_unexpected | {result_a['delta_var_unexpected']:.6f} | {result_b['delta_var_unexpected']:.6f} |
| delta_var_expected | {result_a['delta_var_expected']:.6f} | {result_b['delta_var_expected']:.6f} |

## PASS Criteria

| Criterion | Result | Value |
|---|---|---|
| C1: A delta_var_unexpected > 0.005 (spike fires on unexpected) | {"PASS" if c1_pass else "FAIL"} | {result_a['delta_var_unexpected']:.6f} |
| C2: A delta_var_expected < 0.002 (gate selective on expected) | {"PASS" if c2_pass else "FAIL"} | {result_a['delta_var_expected']:.6f} |
| C3: B delta_var_unexpected < 0.002 (baseline stays flat) | {"PASS" if c3_pass else "FAIL"} | {result_b['delta_var_unexpected']:.6f} |
| C4: n_unexpected_harm_A >= 10 (sufficient events) | {"PASS" if c4_pass else "FAIL"} | {result_a['n_unexpected_harm']} |
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
        "fatal_error_count": float(metrics["fatal_error_count"]),
    }


if __name__ == "__main__":
    import argparse
    import json
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",               type=int,   default=0)
    parser.add_argument("--warmup",             type=int,   default=400)
    parser.add_argument("--eval-eps",           type=int,   default=50)
    parser.add_argument("--steps",              type=int,   default=200)
    parser.add_argument("--alpha-world",        type=float, default=0.9)
    parser.add_argument("--spike-magnitude",    type=float, default=0.05)
    parser.add_argument("--surprise-threshold", type=float, default=0.02)
    args = parser.parse_args()

    result = run(
        seed=args.seed,
        warmup_episodes=args.warmup,
        eval_episodes=args.eval_eps,
        steps_per_episode=args.steps,
        alpha_world=args.alpha_world,
        spike_magnitude=args.spike_magnitude,
        surprise_threshold=args.surprise_threshold,
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
            print(f"  {k}: {v:.6f}", flush=True)
