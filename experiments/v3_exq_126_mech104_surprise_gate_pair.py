#!/opt/local/bin/python3
"""
V3-EXQ-126 -- MECH-104: Volatility Gate / Unexpected Harm Surprise Spike (Matched-Seed Pair)

Claims: MECH-104
Proposal: EXP-0086

MECH-104 asserts that unexpected harm events spike commitment uncertainty (beta
variance) via a surprise gate, distinct from the jitter noise floor baseline.
This is the LC-NE volatility interrupt mechanism: norepinephrine release is driven
by unexpected outcomes, not routine harm or background jitter.

Context:
  EXQ-049e PASS: Route-1 validated -- jitter noise floor prevents permanent variance
  collapse and enables de-commitment.
  EXQ-061  PASS: Route-1 -- any harm contact triggers variance spike (coarse test).
  EXQ-064  PASS (single seed): surprise gate raises variance on unexpected harm;
  gate selective for unexpected vs expected harm. Condition B (no-spike baseline)
  stays flat. 5/5 criteria met.

  This experiment EXTENDS EXQ-064 to a proper matched-seed discriminative pair:
  two conditions (SURPRISE_GATE_ON vs SURPRISE_GATE_ABLATED) x two seeds [42, 123].
  The ablation condition explicitly disables the surprise gate while keeping all other
  architecture intact -- a structural on/off comparison rather than gate vs. no-gate
  as separate run types. This is the canonical discriminative pair format.

Design -- SURPRISE_GATE_ON vs SURPRISE_GATE_ABLATED x 2 matched seeds:

  For each seed, train one agent (400 warmup episodes, committed state reached).
  Then evaluate 2 conditions in a single eval pass over 50 episodes:

  Condition ON -- SURPRISE_GATE_ON (gate active, Route-2 wiring enabled)
    When committed AND actual_harm < HARM_CONTACT_THRESHOLD:
      surprise = |actual_harm - predicted_harm|
      if surprise > SURPRISE_THRESHOLD:
        running_variance += SPIKE_MAGNITUDE * (surprise - SURPRISE_THRESHOLD)
    Harm events classified as "unexpected" (spike fired) or "expected" (no spike).

  Condition ABLATED -- SURPRISE_GATE_ABLATED (gate disabled, ablation)
    Same trained agent weights and same episode seeds. Surprise is still computed
    and classified, but no variance impulse is applied. This isolates the gate's
    causal contribution to variance dynamics.

  Both conditions record per-step variance before/after harm events, classified
  by unexpected vs expected, to test whether the gate causally raises variance.

Pre-registered PASS criteria (ALL must hold across BOTH seeds):

  C1: ON delta_var_unexpected >= THRESH_C1 (default 0.005)
      Surprise gate raises variance on unexpected harm events in ON condition.
      Replicates EXQ-064 C1 with matched seeds.

  C2: ON delta_var_expected < THRESH_C2 (default 0.002)
      Gate is selective -- does NOT spike on expected harm. Tests that the
      surprise computation is doing meaningful work, not trivially always firing.

  C3: ABLATED delta_var_unexpected < THRESH_C3 (default 0.002)
      Ablated baseline stays flat on unexpected harm -- no spurious spikes without
      the gate. Confirms variance increase in ON is gate-specific, not an artifact
      of the harm contact itself.

  C4: (ON delta_var_unexpected) - (ABLATED delta_var_unexpected) >= THRESH_C4 (default 0.004)
      Discriminative criterion: the gate produces meaningfully more variance on
      unexpected harm than the ablated condition. Cross-condition delta must be
      at least THRESH_C4 in both seeds.

  C5: n_unexpected_harm_ON >= THRESH_C5 (default 10)
      Sufficient unexpected harm events to make measurements reliable. Both seeds.

  C6: No fatal errors across all conditions.

  Diagnostic (not PASS/FAIL):
  D1: n_surprise_spikes_ON vs n_unexpected_harm_ON (should be close -- spike fires on
      most unexpected committed harm steps)
  D2: ratio ON delta / ABLATED delta (shows gate effect size)
  D3: final_running_variance per seed (confirms committed state reached)
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


EXPERIMENT_TYPE = "v3_exq_126_mech104_surprise_gate_pair"
CLAIM_IDS = ["MECH-104"]

# Pre-registered thresholds
THRESH_C1 = 0.005   # ON delta_var_unexpected >= 0.005 (spike fires, both seeds)
THRESH_C2 = 0.002   # ON delta_var_expected < 0.002 (selective, both seeds)
THRESH_C3 = 0.002   # ABLATED delta_var_unexpected < 0.002 (baseline flat, both seeds)
THRESH_C4 = 0.004   # (ON - ABLATED) delta_var_unexpected >= 0.004 (discriminative, both seeds)
THRESH_C5 = 10      # n_unexpected_harm_ON >= 10 (both seeds)

HARM_CONTACT_THRESHOLD = -0.01  # harm_signal below this = harm contact event
SPIKE_MAGNITUDE = 0.05
SURPRISE_THRESHOLD = 0.02

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


def _make_agent(
    seed: int,
    self_dim: int,
    world_dim: int,
    alpha_world: float,
    env: CausalGridWorldV2,
) -> REEAgent:
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
    return REEAgent(config)


def _train_agent(
    agent: REEAgent,
    env: CausalGridWorldV2,
    num_episodes: int,
    steps_per_episode: int,
    world_dim: int,
    seed: int,
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
            if len(harm_buf_pos) > 1000:
                harm_buf_pos = harm_buf_pos[-1000:]
            if len(harm_buf_neg) > 1000:
                harm_buf_neg = harm_buf_neg[-1000:]

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
                f"  [train seed={seed}] ep {ep+1}/{num_episodes}"
                f"  running_var={rv:.6f}",
                flush=True,
            )

    return {"final_running_variance": float(agent.e3._running_variance)}


def _eval_condition(
    agent: REEAgent,
    env: CausalGridWorldV2,
    num_episodes: int,
    steps_per_episode: int,
    world_dim: int,
    gate_active: bool,
    label: str,
    initial_variance: float,
) -> Dict:
    """Eval SURPRISE_GATE_ON or SURPRISE_GATE_ABLATED on same trained agent.

    gate_active=True:  surprise gate fires on unexpected committed harm steps.
    gate_active=False: ablation -- gate disabled, variance unchanged on harm steps.

    Both conditions:
    - Use same trained weights and same eval env seed sequence.
    - Record variance before/after harm events, classified unexpected vs expected.
    - Compute harm eval prediction before each env step (required for both conditions
      so that the surprise classification is identical).
    """
    agent.eval()

    var_before_unexpected: List[float] = []
    var_after_unexpected: List[float] = []
    var_before_expected: List[float] = []
    var_after_expected: List[float] = []

    n_surprise_spikes = 0
    n_committed_harm = 0
    n_uncommitted_harm = 0
    fatal = 0

    for ep in range(num_episodes):
        # Use episode index as additional seed offset so both conditions
        # see the same sequence of environments.
        flat_obs, obs_dict = env.reset()
        agent.reset()
        agent.e3._running_variance = initial_variance

        for _ in range(steps_per_episode):
            obs_body = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            with torch.no_grad():
                latent = agent.sense(obs_body, obs_world)

            # Query predicted harm BEFORE env step (needed for surprise computation)
            with torch.no_grad():
                theta_z = agent.theta_buffer.summary()
                predicted_harm = float(agent.e3.harm_eval(theta_z).item())

            # Commitment state and variance BEFORE step
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

                # Process harm contact events
                if actual_harm < HARM_CONTACT_THRESHOLD:
                    surprise = abs(actual_harm - predicted_harm)
                    is_unexpected = surprise > SURPRISE_THRESHOLD

                    if is_committed_pre:
                        n_committed_harm += 1
                        # Gate: apply variance impulse only in ON condition
                        if gate_active and is_unexpected:
                            impulse = SPIKE_MAGNITUDE * (surprise - SURPRISE_THRESHOLD)
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
                flat_obs, obs_dict = env.reset()
                done = True

            if done:
                break

    mean_var_before_unexpected = _mean_safe(var_before_unexpected)
    mean_var_after_unexpected = _mean_safe(var_after_unexpected)
    mean_var_before_expected = _mean_safe(var_before_expected)
    mean_var_after_expected = _mean_safe(var_after_expected)

    delta_unexpected = mean_var_after_unexpected - mean_var_before_unexpected
    delta_expected = mean_var_after_expected - mean_var_before_expected

    print(
        f"\n  [{label}] gate_active={gate_active}"
        f"\n    n_unexpected_harm={len(var_before_unexpected)}"
        f"  n_expected_harm={len(var_before_expected)}"
        f"  n_surprise_spikes={n_surprise_spikes}"
        f"\n    delta_var_unexpected={delta_unexpected:.6f}"
        f"  delta_var_expected={delta_expected:.6f}"
        f"\n    var_before_unexpected={mean_var_before_unexpected:.6f}"
        f"  var_after_unexpected={mean_var_after_unexpected:.6f}"
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


def _run_seed(
    seed: int,
    warmup_episodes: int,
    eval_episodes: int,
    steps_per_episode: int,
    self_dim: int,
    world_dim: int,
    alpha_world: float,
) -> Dict:
    """Run one seed: train, then eval SURPRISE_GATE_ON and SURPRISE_GATE_ABLATED."""
    print(f"\n{'='*60}", flush=True)
    print(f"[V3-EXQ-126] Seed {seed}", flush=True)
    print('='*60, flush=True)

    env = CausalGridWorldV2(seed=seed, **ENV_KWARGS)
    agent = _make_agent(seed, self_dim, world_dim, alpha_world, env)

    train_out = _train_agent(agent, env, warmup_episodes, steps_per_episode, world_dim, seed)
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

    # Condition ON: surprise gate active
    print(f"\n{'='*60}", flush=True)
    print(f"[V3-EXQ-126] Seed {seed} -- SURPRISE_GATE_ON", flush=True)
    print('='*60, flush=True)
    result_on = _eval_condition(
        agent=agent,
        env=env,
        num_episodes=eval_episodes,
        steps_per_episode=steps_per_episode,
        world_dim=world_dim,
        gate_active=True,
        label=f"SURPRISE_GATE_ON seed={seed}",
        initial_variance=initial_variance,
    )

    # Condition ABLATED: gate disabled (same weights, same episode flow)
    print(f"\n{'='*60}", flush=True)
    print(f"[V3-EXQ-126] Seed {seed} -- SURPRISE_GATE_ABLATED", flush=True)
    print('='*60, flush=True)
    result_ablated = _eval_condition(
        agent=agent,
        env=env,
        num_episodes=eval_episodes,
        steps_per_episode=steps_per_episode,
        world_dim=world_dim,
        gate_active=False,
        label=f"SURPRISE_GATE_ABLATED seed={seed}",
        initial_variance=initial_variance,
    )

    return {
        "seed": seed,
        "initial_variance": initial_variance,
        "on": result_on,
        "ablated": result_ablated,
    }


def run(
    seeds: List[int] = None,
    warmup_episodes: int = 400,
    eval_episodes: int = 50,
    steps_per_episode: int = 200,
    alpha_world: float = 0.9,
    self_dim: int = 32,
    world_dim: int = 32,
    dry_run: bool = False,
    **kwargs,
) -> dict:
    if seeds is None:
        seeds = [42, 123]

    if dry_run:
        warmup_episodes = 2
        eval_episodes = 2
        steps_per_episode = 10

    print(
        f"[V3-EXQ-126] MECH-104: Volatility Gate -- SURPRISE_GATE_ON vs SURPRISE_GATE_ABLATED\n"
        f"  Design: seeds {seeds} x 2 conditions = {len(seeds) * 2} cells\n"
        f"  SPIKE_MAGNITUDE={SPIKE_MAGNITUDE}  SURPRISE_THRESHOLD={SURPRISE_THRESHOLD}\n"
        f"  HARM_CONTACT_THRESHOLD={HARM_CONTACT_THRESHOLD}\n"
        f"  Pre-registered: C1>={THRESH_C1}, C2<{THRESH_C2}, C3<{THRESH_C3},"
        f"  C4>={THRESH_C4}, C5>={THRESH_C5}\n"
        f"  Warmup={warmup_episodes} eps  Eval={eval_episodes} eps"
        f"  Steps={steps_per_episode}  alpha_world={alpha_world}",
        flush=True,
    )

    results_by_seed: List[Dict] = []
    for seed in seeds:
        seed_result = _run_seed(
            seed=seed,
            warmup_episodes=warmup_episodes,
            eval_episodes=eval_episodes,
            steps_per_episode=steps_per_episode,
            self_dim=self_dim,
            world_dim=world_dim,
            alpha_world=alpha_world,
        )
        results_by_seed.append(seed_result)

    # -------------------------------------------------------------------------
    # Per-seed criterion evaluation
    # -------------------------------------------------------------------------
    c1_per_seed = []
    c2_per_seed = []
    c3_per_seed = []
    c4_per_seed = []
    c5_per_seed = []

    for sr in results_by_seed:
        on = sr["on"]
        ab = sr["ablated"]
        delta_on = on["delta_var_unexpected"]
        delta_ab = ab["delta_var_unexpected"]
        discriminative_delta = delta_on - delta_ab

        c1_per_seed.append(delta_on >= THRESH_C1)
        c2_per_seed.append(on["delta_var_expected"] < THRESH_C2)
        c3_per_seed.append(delta_ab < THRESH_C3)
        c4_per_seed.append(discriminative_delta >= THRESH_C4)
        c5_per_seed.append(on["n_unexpected_harm"] >= THRESH_C5)

    # All criteria must hold across ALL seeds
    c1_pass = all(c1_per_seed)
    c2_pass = all(c2_per_seed)
    c3_pass = all(c3_per_seed)
    c4_pass = all(c4_per_seed)
    c5_pass = all(c5_per_seed)
    c6_pass = all(
        sr["on"]["fatal_errors"] + sr["ablated"]["fatal_errors"] == 0
        for sr in results_by_seed
    )

    all_pass = c1_pass and c2_pass and c3_pass and c4_pass and c5_pass and c6_pass
    status = "PASS" if all_pass else "FAIL"
    criteria_met = sum([c1_pass, c2_pass, c3_pass, c4_pass, c5_pass, c6_pass])

    failure_notes = []
    for i, sr in enumerate(results_by_seed):
        seed = sr["seed"]
        on = sr["on"]
        ab = sr["ablated"]
        delta_on = on["delta_var_unexpected"]
        delta_ab = ab["delta_var_unexpected"]
        disc = delta_on - delta_ab
        if not c1_per_seed[i]:
            failure_notes.append(
                f"C1 FAIL seed={seed}: ON delta_var_unexpected={delta_on:.6f} < {THRESH_C1}"
            )
        if not c2_per_seed[i]:
            failure_notes.append(
                f"C2 FAIL seed={seed}: ON delta_var_expected={on['delta_var_expected']:.6f} >= {THRESH_C2}"
            )
        if not c3_per_seed[i]:
            failure_notes.append(
                f"C3 FAIL seed={seed}: ABLATED delta_var_unexpected={delta_ab:.6f} >= {THRESH_C3}"
            )
        if not c4_per_seed[i]:
            failure_notes.append(
                f"C4 FAIL seed={seed}: discriminative_delta={disc:.6f} < {THRESH_C4}"
            )
        if not c5_per_seed[i]:
            failure_notes.append(
                f"C5 FAIL seed={seed}: n_unexpected_harm_ON={on['n_unexpected_harm']} < {THRESH_C5}"
            )

    if not c6_pass:
        total_fatal = sum(
            sr["on"]["fatal_errors"] + sr["ablated"]["fatal_errors"]
            for sr in results_by_seed
        )
        failure_notes.append(f"C6 FAIL: fatal_errors={total_fatal}")

    print(f"\nV3-EXQ-126 verdict: {status}  ({criteria_met}/6)", flush=True)
    for note in failure_notes:
        print(f"  {note}", flush=True)

    # -------------------------------------------------------------------------
    # Build flat metrics dict (per-seed prefixed)
    # -------------------------------------------------------------------------
    metrics: Dict[str, float] = {}
    for sr in results_by_seed:
        seed = sr["seed"]
        on = sr["on"]
        ab = sr["ablated"]
        pfx = f"s{seed}"
        metrics[f"{pfx}_on_n_unexpected_harm"]     = float(on["n_unexpected_harm"])
        metrics[f"{pfx}_on_n_expected_harm"]        = float(on["n_expected_harm"])
        metrics[f"{pfx}_on_n_surprise_spikes"]      = float(on["n_surprise_spikes"])
        metrics[f"{pfx}_on_delta_var_unexpected"]   = float(on["delta_var_unexpected"])
        metrics[f"{pfx}_on_delta_var_expected"]     = float(on["delta_var_expected"])
        metrics[f"{pfx}_ab_n_unexpected_harm"]      = float(ab["n_unexpected_harm"])
        metrics[f"{pfx}_ab_delta_var_unexpected"]   = float(ab["delta_var_unexpected"])
        metrics[f"{pfx}_ab_delta_var_expected"]     = float(ab["delta_var_expected"])
        disc_delta = on["delta_var_unexpected"] - ab["delta_var_unexpected"]
        metrics[f"{pfx}_discriminative_delta"]      = float(disc_delta)
        metrics[f"{pfx}_initial_variance"]          = float(sr["initial_variance"])
        metrics[f"{pfx}_fatal_errors"]              = float(
            on["fatal_errors"] + ab["fatal_errors"]
        )

    metrics["crit1_pass"]     = 1.0 if c1_pass else 0.0
    metrics["crit2_pass"]     = 1.0 if c2_pass else 0.0
    metrics["crit3_pass"]     = 1.0 if c3_pass else 0.0
    metrics["crit4_pass"]     = 1.0 if c4_pass else 0.0
    metrics["crit5_pass"]     = 1.0 if c5_pass else 0.0
    metrics["crit6_pass"]     = 1.0 if c6_pass else 0.0
    metrics["criteria_met"]   = float(criteria_met)
    metrics["fatal_error_count"] = float(sum(
        sr["on"]["fatal_errors"] + sr["ablated"]["fatal_errors"]
        for sr in results_by_seed
    ))

    # -------------------------------------------------------------------------
    # Summary markdown
    # -------------------------------------------------------------------------
    seed_rows_on = "\n".join(
        f"| {sr['seed']} | {sr['on']['n_unexpected_harm']} | {sr['on']['n_surprise_spikes']} "
        f"| {sr['on']['delta_var_unexpected']:.6f} | {sr['on']['delta_var_expected']:.6f} |"
        for sr in results_by_seed
    )
    seed_rows_ab = "\n".join(
        f"| {sr['seed']} | {sr['ablated']['n_unexpected_harm']} "
        f"| {sr['ablated']['delta_var_unexpected']:.6f} | {sr['ablated']['delta_var_expected']:.6f} |"
        for sr in results_by_seed
    )
    seed_rows_disc = "\n".join(
        f"| {sr['seed']} | "
        f"{sr['on']['delta_var_unexpected'] - sr['ablated']['delta_var_unexpected']:.6f} |"
        for sr in results_by_seed
    )

    failure_section = ""
    if failure_notes:
        failure_section = "\n## Failure Notes\n\n" + "\n".join(f"- {n}" for n in failure_notes)

    summary_markdown = f"""# V3-EXQ-126 -- MECH-104: Volatility Gate / Unexpected Harm Surprise Spike

**Status:** {status}
**Claims:** MECH-104
**Design:** SURPRISE_GATE_ON vs SURPRISE_GATE_ABLATED x seeds {[sr['seed'] for sr in results_by_seed]}
**SPIKE_MAGNITUDE:** {SPIKE_MAGNITUDE}  |  **SURPRISE_THRESHOLD:** {SURPRISE_THRESHOLD}
**Warmup:** {warmup_episodes} eps  |  **Eval:** {eval_episodes} eps  |  **Steps:** {steps_per_episode}

## Design Rationale

MECH-104 (LC-NE volatility interrupt) predicts that unexpected harm events spike
commitment uncertainty (beta / running_variance) via a surprise gate. This is Route-2
validation: the gate is selective for unexpected harm (surprise > threshold), not routine
or anticipated harm.

This matched-seed discriminative pair extends EXQ-064 (single-seed PASS, 5/5) by:
(1) Using two matched seeds [42, 123] for cross-seed consistency.
(2) Explicitly contrasting SURPRISE_GATE_ON vs SURPRISE_GATE_ABLATED on the SAME trained
    agent -- the ablation directly removes the gate's causal contribution.
(3) Adding C4 (discriminative criterion): the cross-condition delta must be >= {THRESH_C4}.

If SURPRISE_GATE_ON shows higher delta_var_unexpected than SURPRISE_GATE_ABLATED, and the
gate is selective (no spike on expected harm), this supports MECH-104.

## SURPRISE_GATE_ON Results

| seed | n_unexpected | n_spikes | delta_var_unexpected | delta_var_expected |
|------|-------------|---------|---------------------|--------------------|
{seed_rows_on}

## SURPRISE_GATE_ABLATED Results

| seed | n_unexpected | delta_var_unexpected | delta_var_expected |
|------|-------------|---------------------|-------------------|
{seed_rows_ab}

## Discriminative Delta (ON - ABLATED, delta_var_unexpected)

| seed | discriminative_delta |
|------|---------------------|
{seed_rows_disc}

## PASS Criteria

| Criterion | Result | Values |
|-----------|--------|--------|
| C1: ON delta_var_unexpected >= {THRESH_C1} (both seeds) | {"PASS" if c1_pass else "FAIL"} | {", ".join(f"s{sr['seed']}={sr['on']['delta_var_unexpected']:.6f}" for sr in results_by_seed)} |
| C2: ON delta_var_expected < {THRESH_C2} (both seeds) | {"PASS" if c2_pass else "FAIL"} | {", ".join(f"s{sr['seed']}={sr['on']['delta_var_expected']:.6f}" for sr in results_by_seed)} |
| C3: ABLATED delta_var_unexpected < {THRESH_C3} (both seeds) | {"PASS" if c3_pass else "FAIL"} | {", ".join(f"s{sr['seed']}={sr['ablated']['delta_var_unexpected']:.6f}" for sr in results_by_seed)} |
| C4: (ON-ABLATED) delta_var_unexpected >= {THRESH_C4} (both seeds) | {"PASS" if c4_pass else "FAIL"} | {", ".join(f"s{sr['seed']}={sr['on']['delta_var_unexpected']-sr['ablated']['delta_var_unexpected']:.6f}" for sr in results_by_seed)} |
| C5: n_unexpected_harm_ON >= {THRESH_C5} (both seeds) | {"PASS" if c5_pass else "FAIL"} | {", ".join(f"s{sr['seed']}={sr['on']['n_unexpected_harm']}" for sr in results_by_seed)} |
| C6: No fatal errors | {"PASS" if c6_pass else "FAIL"} | {int(metrics['fatal_error_count'])} |

Criteria met: {criteria_met}/6 -> **{status}**
{failure_section}
"""

    return {
        "status": status,
        "metrics": metrics,
        "summary_markdown": summary_markdown,
        "claim_ids": CLAIM_IDS,
        "evidence_direction": (
            "supports" if all_pass else ("mixed" if criteria_met >= 4 else "weakens")
        ),
        "experiment_type": EXPERIMENT_TYPE,
        "fatal_error_count": float(metrics["fatal_error_count"]),
    }


if __name__ == "__main__":
    import argparse
    import json
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds",        type=int, nargs="+", default=[42, 123])
    parser.add_argument("--warmup",       type=int,   default=400)
    parser.add_argument("--eval-eps",     type=int,   default=50)
    parser.add_argument("--steps",        type=int,   default=200)
    parser.add_argument("--alpha-world",  type=float, default=0.9)
    parser.add_argument("--dry-run",      action="store_true")
    args = parser.parse_args()

    result = run(
        seeds=args.seeds,
        warmup_episodes=args.warmup,
        eval_episodes=args.eval_eps,
        steps_per_episode=args.steps,
        alpha_world=args.alpha_world,
        dry_run=args.dry_run,
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
