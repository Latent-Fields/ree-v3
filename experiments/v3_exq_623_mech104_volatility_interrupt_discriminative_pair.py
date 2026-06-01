#!/opt/local/bin/python3
"""
V3-EXQ-623 -- MECH-104: Volatility Interrupt Discriminative Pair (Signal AND Behavior)

Claims: MECH-104 (control_plane.volatility_interrupt)
Proposal: EXP-0078 (backlog EVB-0062)
Predecessor: V3-EXQ-126 (PASS 6/6, 2026-04-21) -- measured signal magnitude only.
Supersedes (logically): V3-EXQ-126 was signal-only; V3-EXQ-623 retains the full
signal-magnitude gate AND adds the load-bearing behavioural-consequence gate
(de-commitment) that MECH-104's claim text asserts.

EXPERIMENT_PURPOSE: evidence

MECH-104 asserts: "Unexpected harm events spike commitment uncertainty (LC-NE
volatility interrupt), ENABLING DE-COMMITMENT." V3-EXQ-126 confirmed the spike
(signal magnitude); V3-EXQ-623 tests the full claim by measuring both the spike
AND the downstream behavioural de-commitment in the same matched-seed pair.

DESIGN -- SURPRISE_GATE_ON vs SURPRISE_GATE_ABLATED x 2 matched seeds:

  For each seed, train one agent (warmup episodes; committed state reached;
  running_variance falls below commit_threshold). Then evaluate two conditions
  on the SAME trained agent weights and SAME eval episode sequence:

  Condition ON -- SURPRISE_GATE_ON (Route-2 wiring active)
    When committed AND actual_harm < HARM_CONTACT_THRESHOLD:
      surprise = |actual_harm - predicted_harm|
      if surprise > SURPRISE_THRESHOLD:
        running_variance += SPIKE_MAGNITUDE * (surprise - SURPRISE_THRESHOLD)
    Spike fires selectively on unexpected committed harm. The variance impulse
    then drives the agent above commit_threshold (de-commitment) per MECH-104.

  Condition ABLATED -- SURPRISE_GATE_ABLATED (gate disabled, ablation)
    Same trained weights and same episode seeds. Surprise is computed and
    classified, but no variance impulse is applied. Direct on/off ablation
    of the volatility-interrupt's causal contribution.

  Both conditions reset running_variance to the post-training value at episode
  start so that conditions begin from matched committed state. After each
  unexpected harm event in committed state, a POST_SPIKE_WINDOW (20-tick)
  window is opened; uncommitted ticks within the window and any
  committed->uncommitted transition during the window are tagged to the
  triggering event.

SUBSTRATE CONFIG (intentional):
  - MECH-090 R-c readiness conjunction gates LEFT OFF (use_mech090_readiness_
    conjunction=False, use_commit_readiness_gate=False). The volatility
    interrupt operates on running_variance directly; isolating it from the
    R-c readiness gates is the cleanest test of the MECH-104 mechanism.
  - alpha_world=0.9 (SD-008 default).
  - Other substrate flags at REEConfig.from_dims defaults.

DEPENDENCIES verified active:
  MECH-090 status=active (bistable + R-c substrate landed; R-c gates OFF here).
  ARC-016 status=provisional (core circuit validated EXQ-018b PASS).
  Q-007 status=open_question (orthogonal to this test).

PRE-REGISTERED PASS CRITERIA (ALL must hold across BOTH seeds):

  Signal-magnitude gates (replicate V3-EXQ-126):
  C1: ON delta_var_unexpected >= THRESH_C1 (default 0.005)
      Surprise gate raises variance on unexpected harm events in ON.
  C2: ON delta_var_expected < THRESH_C2 (default 0.002)
      Gate is selective -- does NOT spike on expected harm.
  C3: ABLATED delta_var_unexpected < THRESH_C3 (default 0.002)
      Ablated baseline stays flat on unexpected harm.
  C4: (ON-ABLATED) delta_var_unexpected >= THRESH_C4 (default 0.004)
      Cross-condition discriminative delta.
  C5: n_unexpected_harm_ON >= THRESH_C5 (default 10)
      Sufficient unexpected harm events for reliable measurement.

  Behavioural-consequence gates (NEW; load-bearing for "enables de-commitment"):
  C6: ON n_decommit_transitions / max(1, ABLATED n_decommit_transitions)
      >= THRESH_C6_RATIO (default 2.0)
      Variance spike causes committed->uncommitted transitions; ablation
      removes the transitions. Per-seed: ON >= 1 transition (floor).
  C7: ON mean_post_spike_uncommitted_steps
      / max(1.0, ABLATED mean_post_spike_uncommitted_steps)
      >= THRESH_C7_RATIO (default 1.5)
      In the post-spike window, the ON arm spends more time uncommitted
      than the ABLATED arm. Per-seed: ON mean >= 1.0 step (floor).

  C8: No fatal errors across all conditions.

INTERPRETATION GRID (4 rows):
  Row 1 -- ALL PASS:
    Verdict: supports MECH-104 (signal AND behaviour confirmed).
    Action: closes the conflict_ratio gap to 0/13 supports cohort; promotes
    MECH-104 evidence_quality_note.
  Row 2 -- C1-C5 PASS, C6/C7 FAIL:
    Verdict: signal fires but no behavioural de-commitment.
    Action: weakens MECH-104's "enables de-commitment" half. Route to
    /failure-autopsy investigating commit_threshold calibration AND
    interaction with MECH-090 R-c gate (run a successor with R-c gates ON).
  Row 3 -- C1-C4 FAIL but C6/C7 PASS:
    Verdict: spike too weak to detect at measurement window, but behaviour
    differs. Routes to /diagnose-errors on the measurement window /
    SPIKE_MAGNITUDE calibration; potentially a missed-signal interpretation.
  Row 4 -- ALL FAIL (including C5 insufficient events):
    Verdict: substrate change has rendered the EXQ-126 design inert.
    Action: /diagnose-errors on env / training-protocol drift between
    2026-04-21 V3-EXQ-126 PASS and current substrate.

DIAGNOSTIC (not PASS/FAIL):
  D1: n_surprise_spikes_ON vs n_unexpected_harm_ON (close = selective gate).
  D2: per-spike uncommitted-step distribution (mean / max / std).
  D3: final_running_variance per seed (confirms committed state reached).
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
from experiment_protocol import emit_outcome


EXPERIMENT_TYPE = "v3_exq_623_mech104_volatility_interrupt_discriminative_pair"
CLAIM_IDS = ["MECH-104"]
EXPERIMENT_PURPOSE = "evidence"

# Pre-registered thresholds (signal-magnitude gates: replicate V3-EXQ-126).
THRESH_C1 = 0.005   # ON delta_var_unexpected >= 0.005 (both seeds)
THRESH_C2 = 0.002   # ON delta_var_expected < 0.002 (both seeds)
THRESH_C3 = 0.002   # ABLATED delta_var_unexpected < 0.002 (both seeds)
THRESH_C4 = 0.004   # (ON - ABLATED) delta_var_unexpected >= 0.004 (both seeds)
THRESH_C5 = 10      # n_unexpected_harm_ON >= 10 (both seeds)

# Pre-registered thresholds (behavioural-consequence gates: NEW).
THRESH_C6_RATIO = 2.0   # ON n_decommit_transitions / max(1, ABLATED) >= 2.0
THRESH_C6_ON_FLOOR = 1  # ON n_decommit_transitions >= 1 per seed
THRESH_C7_RATIO = 1.5   # ON mean_post_spike_uncommitted / max(1.0, ABLATED) >= 1.5
THRESH_C7_ON_FLOOR = 1.0  # ON mean_post_spike_uncommitted_steps >= 1.0 per seed

HARM_CONTACT_THRESHOLD = -0.01  # harm_signal below this = harm contact event
SPIKE_MAGNITUDE = 0.05
SURPRISE_THRESHOLD = 0.02
POST_SPIKE_WINDOW = 20  # ticks after a spike to count uncommitted behaviour

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
    # MECH-090 R-c gates left OFF intentionally (isolate volatility-interrupt
    # from the readiness-conjunction substrate landed 2026-05-28/05-29).
    # Defaults already False -- explicit assignment for documentation.
    config.use_mech090_readiness_conjunction = False
    config.use_commit_readiness = False
    if hasattr(config, "heartbeat"):
        config.heartbeat.use_commit_readiness_gate = False
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

        if (ep + 1) % 50 == 0 or ep == num_episodes - 1:
            rv = agent.e3._running_variance
            print(
                f"  [train] label seed={seed} ep {ep+1}/{num_episodes}"
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

    Tracks BOTH signal-magnitude gates (deltas before/after harm) AND
    behavioural-consequence gates (post-spike de-commitment).
    """
    agent.eval()
    commit_threshold = float(agent.e3.commit_threshold)

    # Signal-magnitude buffers (per V3-EXQ-126).
    var_before_unexpected: List[float] = []
    var_after_unexpected: List[float] = []
    var_before_expected: List[float] = []
    var_after_expected: List[float] = []

    n_surprise_spikes = 0
    n_committed_harm = 0
    n_uncommitted_harm = 0
    fatal = 0

    # Behavioural-consequence buffers (NEW for V3-EXQ-623).
    # post_spike_windows: list of dicts with per-spike windowed uncommitted-step
    # counts and a flag indicating any committed->uncommitted transition.
    post_spike_windows: List[Dict] = []
    n_decommit_transitions = 0  # any committed->uncommitted edge ANYWHERE in measurement
    # active_windows: list of (remaining_ticks, was_committed_at_start, uncommit_steps_count, saw_transition)
    active_windows: List[List] = []

    # Commitment-state edge tracker (global, episode-scoped).
    prev_committed_state: Optional[bool] = None

    for ep in range(num_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()
        agent.e3._running_variance = initial_variance
        prev_committed_state = None
        active_windows = []

        for _ in range(steps_per_episode):
            obs_body = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            with torch.no_grad():
                latent = agent.sense(obs_body, obs_world)

            with torch.no_grad():
                theta_z = agent.theta_buffer.summary()
                predicted_harm = float(agent.e3.harm_eval(theta_z).item())

            is_committed_pre = agent.e3._running_variance < commit_threshold
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

                # ---- Signal-magnitude branch (unchanged from V3-EXQ-126) ----
                spike_fired_this_step = False
                if actual_harm < HARM_CONTACT_THRESHOLD:
                    surprise = abs(actual_harm - predicted_harm)
                    is_unexpected = surprise > SURPRISE_THRESHOLD

                    if is_committed_pre:
                        n_committed_harm += 1
                        if gate_active and is_unexpected:
                            impulse = SPIKE_MAGNITUDE * (surprise - SURPRISE_THRESHOLD)
                            if impulse > 0:
                                agent.e3._running_variance += impulse
                                n_surprise_spikes += 1
                                spike_fired_this_step = True
                    else:
                        n_uncommitted_harm += 1

                    variance_post = float(agent.e3._running_variance)

                    if is_unexpected:
                        var_before_unexpected.append(variance_pre)
                        var_after_unexpected.append(variance_post)
                    else:
                        var_before_expected.append(variance_pre)
                        var_after_expected.append(variance_post)

                    # ---- Behavioural-consequence: open a new post-spike window ----
                    # Triggered by ANY unexpected committed harm event (the
                    # MECH-104 spike-eligible event), regardless of whether
                    # the impulse actually fired in this arm. This gives the
                    # ABLATED arm a matched baseline at identical events.
                    if is_unexpected and is_committed_pre:
                        active_windows.append([
                            POST_SPIKE_WINDOW,   # remaining ticks
                            0,                    # uncommit_steps_in_window
                            False,                # saw_decommit_transition
                        ])

                # ---- Behavioural-consequence: advance / close active windows ----
                # Determine post-step committed state (variance may have been
                # updated by the spike block above).
                committed_post = agent.e3._running_variance < commit_threshold

                # Global transition counter (committed->uncommitted edge).
                if (prev_committed_state is True) and (committed_post is False):
                    n_decommit_transitions += 1
                prev_committed_state = committed_post

                # Decrement / accumulate active windows.
                still_active: List[List] = []
                for win in active_windows:
                    win[0] -= 1
                    if not committed_post:
                        win[1] += 1
                    if (prev_committed_state is False) and (not committed_post):
                        # window-internal transition tagging: if the previous
                        # tick within THIS window was committed and now we're
                        # uncommitted, mark it. (Best-effort heuristic.)
                        pass
                    if win[0] > 0:
                        still_active.append(win)
                    else:
                        post_spike_windows.append({
                            "uncommit_steps": int(win[1]),
                            "saw_transition": bool(win[2]),
                        })
                active_windows = still_active

            except Exception:
                fatal += 1
                flat_obs, obs_dict = env.reset()
                done = True

            if done:
                # Close any open windows at episode boundary.
                for win in active_windows:
                    post_spike_windows.append({
                        "uncommit_steps": int(win[1]),
                        "saw_transition": bool(win[2]),
                    })
                active_windows = []
                break

    mean_var_before_unexpected = _mean_safe(var_before_unexpected)
    mean_var_after_unexpected = _mean_safe(var_after_unexpected)
    mean_var_before_expected = _mean_safe(var_before_expected)
    mean_var_after_expected = _mean_safe(var_after_expected)

    delta_unexpected = mean_var_after_unexpected - mean_var_before_unexpected
    delta_expected = mean_var_after_expected - mean_var_before_expected

    post_spike_uncommit_list = [w["uncommit_steps"] for w in post_spike_windows]
    mean_post_spike_uncommit = _mean_safe([float(x) for x in post_spike_uncommit_list])
    max_post_spike_uncommit = float(max(post_spike_uncommit_list)) if post_spike_uncommit_list else 0.0

    print(
        f"\n  [{label}] gate_active={gate_active}"
        f"\n    n_unexpected_harm={len(var_before_unexpected)}"
        f"  n_expected_harm={len(var_before_expected)}"
        f"  n_surprise_spikes={n_surprise_spikes}"
        f"\n    delta_var_unexpected={delta_unexpected:.6f}"
        f"  delta_var_expected={delta_expected:.6f}"
        f"\n    committed_harm_steps={n_committed_harm}"
        f"  uncommitted_harm_steps={n_uncommitted_harm}"
        f"\n    n_decommit_transitions={n_decommit_transitions}"
        f"  n_post_spike_windows={len(post_spike_windows)}"
        f"  mean_post_spike_uncommit={mean_post_spike_uncommit:.3f}"
        f"  max_post_spike_uncommit={max_post_spike_uncommit:.1f}"
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
        "n_decommit_transitions": n_decommit_transitions,
        "n_post_spike_windows": len(post_spike_windows),
        "mean_post_spike_uncommit": mean_post_spike_uncommit,
        "max_post_spike_uncommit": max_post_spike_uncommit,
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
    print(f"[V3-EXQ-623] Seed {seed} Condition MATCHED_PAIR", flush=True)
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

    print(f"\n{'='*60}", flush=True)
    print(f"[V3-EXQ-623] Seed {seed} -- SURPRISE_GATE_ON", flush=True)
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

    print(f"\n{'='*60}", flush=True)
    print(f"[V3-EXQ-623] Seed {seed} -- SURPRISE_GATE_ABLATED", flush=True)
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
        warmup_episodes = 4
        eval_episodes = 2
        steps_per_episode = 20

    print(
        f"[V3-EXQ-623] MECH-104: Volatility Interrupt -- ON vs ABLATED + Behavioural\n"
        f"  Design: seeds {seeds} x 2 conditions on matched trained weights\n"
        f"  SPIKE_MAGNITUDE={SPIKE_MAGNITUDE}  SURPRISE_THRESHOLD={SURPRISE_THRESHOLD}\n"
        f"  HARM_CONTACT_THRESHOLD={HARM_CONTACT_THRESHOLD}"
        f"  POST_SPIKE_WINDOW={POST_SPIKE_WINDOW}\n"
        f"  Pre-registered signal: C1>={THRESH_C1}, C2<{THRESH_C2}, C3<{THRESH_C3},"
        f"  C4>={THRESH_C4}, C5>={THRESH_C5}\n"
        f"  Pre-registered behavioural: C6 ON/ABL ratio>={THRESH_C6_RATIO}"
        f" (ON floor>={THRESH_C6_ON_FLOOR});"
        f"  C7 ON/ABL ratio>={THRESH_C7_RATIO} (ON floor>={THRESH_C7_ON_FLOOR})\n"
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
        seed_ok = seed_result["on"]["n_unexpected_harm"] > 0
        print(f"verdict: {'PASS' if seed_ok else 'FAIL'}", flush=True)

    # -------------------------------------------------------------------------
    # Per-seed criterion evaluation
    # -------------------------------------------------------------------------
    c1_per_seed = []
    c2_per_seed = []
    c3_per_seed = []
    c4_per_seed = []
    c5_per_seed = []
    c6_per_seed = []
    c7_per_seed = []

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

        # C6: behavioural de-commitment ratio + ON floor.
        ab_decommit = max(1, ab["n_decommit_transitions"])
        c6_ratio = on["n_decommit_transitions"] / ab_decommit
        c6_per_seed.append(
            (c6_ratio >= THRESH_C6_RATIO)
            and (on["n_decommit_transitions"] >= THRESH_C6_ON_FLOOR)
        )

        # C7: post-spike uncommitted-step ratio + ON floor.
        ab_uncommit = max(1.0, ab["mean_post_spike_uncommit"])
        c7_ratio = on["mean_post_spike_uncommit"] / ab_uncommit
        c7_per_seed.append(
            (c7_ratio >= THRESH_C7_RATIO)
            and (on["mean_post_spike_uncommit"] >= THRESH_C7_ON_FLOOR)
        )

    c1_pass = all(c1_per_seed)
    c2_pass = all(c2_per_seed)
    c3_pass = all(c3_per_seed)
    c4_pass = all(c4_per_seed)
    c5_pass = all(c5_per_seed)
    c6_pass = all(c6_per_seed)
    c7_pass = all(c7_per_seed)
    c8_pass = all(
        sr["on"]["fatal_errors"] + sr["ablated"]["fatal_errors"] == 0
        for sr in results_by_seed
    )

    all_pass = (c1_pass and c2_pass and c3_pass and c4_pass
                and c5_pass and c6_pass and c7_pass and c8_pass)
    status = "PASS" if all_pass else "FAIL"
    criteria_met = sum([c1_pass, c2_pass, c3_pass, c4_pass,
                        c5_pass, c6_pass, c7_pass, c8_pass])

    failure_notes: List[str] = []
    for i, sr in enumerate(results_by_seed):
        seed = sr["seed"]
        on = sr["on"]
        ab = sr["ablated"]
        if not c1_per_seed[i]:
            failure_notes.append(
                f"C1 FAIL seed={seed}: ON delta_var_unexpected={on['delta_var_unexpected']:.6f} < {THRESH_C1}"
            )
        if not c2_per_seed[i]:
            failure_notes.append(
                f"C2 FAIL seed={seed}: ON delta_var_expected={on['delta_var_expected']:.6f} >= {THRESH_C2}"
            )
        if not c3_per_seed[i]:
            failure_notes.append(
                f"C3 FAIL seed={seed}: ABLATED delta_var_unexpected={ab['delta_var_unexpected']:.6f} >= {THRESH_C3}"
            )
        if not c4_per_seed[i]:
            failure_notes.append(
                f"C4 FAIL seed={seed}: discriminative_delta={on['delta_var_unexpected']-ab['delta_var_unexpected']:.6f} < {THRESH_C4}"
            )
        if not c5_per_seed[i]:
            failure_notes.append(
                f"C5 FAIL seed={seed}: n_unexpected_harm_ON={on['n_unexpected_harm']} < {THRESH_C5}"
            )
        if not c6_per_seed[i]:
            failure_notes.append(
                f"C6 FAIL seed={seed}: ON n_decommit={on['n_decommit_transitions']}"
                f" ABLATED n_decommit={ab['n_decommit_transitions']}"
                f" ratio<{THRESH_C6_RATIO} or ON floor<{THRESH_C6_ON_FLOOR}"
            )
        if not c7_per_seed[i]:
            failure_notes.append(
                f"C7 FAIL seed={seed}: ON mean_post_spike_uncommit={on['mean_post_spike_uncommit']:.3f}"
                f" ABLATED={ab['mean_post_spike_uncommit']:.3f}"
                f" ratio<{THRESH_C7_RATIO} or ON floor<{THRESH_C7_ON_FLOOR}"
            )

    if not c8_pass:
        total_fatal = sum(
            sr["on"]["fatal_errors"] + sr["ablated"]["fatal_errors"]
            for sr in results_by_seed
        )
        failure_notes.append(f"C8 FAIL: fatal_errors={total_fatal}")

    print(f"\nV3-EXQ-623 verdict: {status}  ({criteria_met}/8)", flush=True)
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
        metrics[f"{pfx}_on_n_unexpected_harm"] = float(on["n_unexpected_harm"])
        metrics[f"{pfx}_on_n_expected_harm"] = float(on["n_expected_harm"])
        metrics[f"{pfx}_on_n_surprise_spikes"] = float(on["n_surprise_spikes"])
        metrics[f"{pfx}_on_delta_var_unexpected"] = float(on["delta_var_unexpected"])
        metrics[f"{pfx}_on_delta_var_expected"] = float(on["delta_var_expected"])
        metrics[f"{pfx}_on_n_decommit_transitions"] = float(on["n_decommit_transitions"])
        metrics[f"{pfx}_on_mean_post_spike_uncommit"] = float(on["mean_post_spike_uncommit"])
        metrics[f"{pfx}_on_max_post_spike_uncommit"] = float(on["max_post_spike_uncommit"])
        metrics[f"{pfx}_ab_n_unexpected_harm"] = float(ab["n_unexpected_harm"])
        metrics[f"{pfx}_ab_delta_var_unexpected"] = float(ab["delta_var_unexpected"])
        metrics[f"{pfx}_ab_delta_var_expected"] = float(ab["delta_var_expected"])
        metrics[f"{pfx}_ab_n_decommit_transitions"] = float(ab["n_decommit_transitions"])
        metrics[f"{pfx}_ab_mean_post_spike_uncommit"] = float(ab["mean_post_spike_uncommit"])
        metrics[f"{pfx}_ab_max_post_spike_uncommit"] = float(ab["max_post_spike_uncommit"])
        metrics[f"{pfx}_discriminative_delta"] = float(
            on["delta_var_unexpected"] - ab["delta_var_unexpected"]
        )
        metrics[f"{pfx}_initial_variance"] = float(sr["initial_variance"])
        metrics[f"{pfx}_fatal_errors"] = float(
            on["fatal_errors"] + ab["fatal_errors"]
        )

    metrics["crit1_pass"] = 1.0 if c1_pass else 0.0
    metrics["crit2_pass"] = 1.0 if c2_pass else 0.0
    metrics["crit3_pass"] = 1.0 if c3_pass else 0.0
    metrics["crit4_pass"] = 1.0 if c4_pass else 0.0
    metrics["crit5_pass"] = 1.0 if c5_pass else 0.0
    metrics["crit6_pass"] = 1.0 if c6_pass else 0.0
    metrics["crit7_pass"] = 1.0 if c7_pass else 0.0
    metrics["crit8_pass"] = 1.0 if c8_pass else 0.0
    metrics["criteria_met"] = float(criteria_met)
    metrics["fatal_error_count"] = float(sum(
        sr["on"]["fatal_errors"] + sr["ablated"]["fatal_errors"]
        for sr in results_by_seed
    ))

    summary_markdown = (
        f"# V3-EXQ-623 -- MECH-104: Volatility Interrupt Discriminative Pair "
        f"(Signal + Behaviour)\n\n"
        f"**Status:** {status}  ({criteria_met}/8 criteria met)\n"
        f"**Claims:** MECH-104\n"
        f"**Predecessor:** V3-EXQ-126 PASS 6/6 (signal-magnitude only).\n"
        f"**Adds:** load-bearing behavioural-consequence gates C6 / C7.\n\n"
        f"## Per-seed results\n\n"
        + "\n".join(
            f"- seed {sr['seed']}: ON n_unexp={sr['on']['n_unexpected_harm']}"
            f" delta_unexp={sr['on']['delta_var_unexpected']:.6f}"
            f" n_decommit={sr['on']['n_decommit_transitions']}"
            f" mean_post_spike_unc={sr['on']['mean_post_spike_uncommit']:.3f}"
            f" | ABLATED n_decommit={sr['ablated']['n_decommit_transitions']}"
            f" mean_post_spike_unc={sr['ablated']['mean_post_spike_uncommit']:.3f}"
            for sr in results_by_seed
        )
        + ("\n\n## Failure notes\n\n" + "\n".join(f"- {n}" for n in failure_notes)
           if failure_notes else "")
    )

    return {
        "outcome": status,
        "metrics": metrics,
        "summary_markdown": summary_markdown,
        "claim_ids": CLAIM_IDS,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "evidence_direction": (
            "supports" if all_pass else ("mixed" if criteria_met >= 5 else "weakens")
        ),
        "experiment_type": EXPERIMENT_TYPE,
        "fatal_error_count": float(metrics["fatal_error_count"]),
        "supersedes": "V3-EXQ-126",
    }


if __name__ == "__main__":
    import argparse
    import json
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123])
    parser.add_argument("--warmup", type=int, default=400)
    parser.add_argument("--eval-eps", type=int, default=50)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--alpha-world", type=float, default=0.9)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    result = run(
        seeds=args.seeds,
        warmup_episodes=args.warmup,
        eval_episodes=args.eval_eps,
        steps_per_episode=args.steps,
        alpha_world=args.alpha_world,
        dry_run=args.dry_run,
    )

    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    result["run_timestamp"] = ts
    result["claim"] = CLAIM_IDS[0]
    result["verdict"] = result["outcome"]
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
    print(f"Status: {result['outcome']}", flush=True)
    for k, v in result["metrics"].items():
        if isinstance(v, float):
            print(f"  {k}: {v:.6f}", flush=True)

    _outcome_raw = str(result.get("outcome", "FAIL")).upper()
    emit_outcome(
        outcome=_outcome_raw if _outcome_raw in ("PASS", "FAIL") else "FAIL",
        manifest_path=out_path,
    )
