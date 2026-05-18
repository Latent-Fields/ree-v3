"""
committed_mode_curriculum.py -- GAP-11 harness helper.

3-phase committed-mode elicitation helper for experiment scripts that need
the agent to actually enter committed mode before measuring a target metric.

NOT a ree_core substrate scheduler.  Lives alongside infant_curriculum.py as a
pure training-loop helper that experiment scripts import.

Problem it solves
-----------------
running_variance starts at 0.5 > commit_threshold(0.40) -> agent never commits
-> beta_gate never elevates -> target metrics are always zero (EXQ-321/261/325
all-zero signature).  The only fix is sustained E2 world-forward training that
drives running_variance below the gate.

Three phases
------------
P0 -- world-model + navigation warmup on an *easy* env configuration.
      Trains E1 prediction loss + E2 world-forward replay.
      Exits when running_variance < effective_commit_threshold for 3
      consecutive probe checkpoints AND optional nav_competence_proxy >= floor.
      Mid-curriculum abort fires at mid_probe_frac of budget if not converging
      (R1 escalation: gate may be mis-calibrated vs achievable world-model error).

P1 -- commitment consolidation on a target (harder) env configuration.
      Continues E1+E2 training.  Exits when median committed_steps_per_episode
      over last `stability_window` episodes >= commitment_floor.
      Mid-curriculum abort fires at mid_probe_frac of P1 budget if commitment
      has not yet emerged.

P2 -- frozen-policy evaluation.  Measures committed_steps, hold_rate,
      rule_state_norm.  The actual governance metric goes here.

Mandatory contrast (O-2)
------------------------
Every behavioural arm MUST run BOTH the emergent arm (P0->P1->P2) AND a
forced-rv control arm (clone_trained_agent + caller sets _running_variance).
The contrast isolates whether the target metric requires *emergent* commitment
or merely the *committed state*.  Use run_p2_eval() for both.

Usage outline
-------------
    from experiments.committed_mode_curriculum import (
        run_p0_warmup, run_p1_consolidation, run_p2_eval,
        clone_trained_agent, CommittedModeMetrics, P0Result, P1Result,
    )

    # P0 on easy env
    easy_env = CausalGridWorldV2(size=10, num_hazards=2, ...)
    p0 = run_p0_warmup(agent, easy_env, device, budget=400)
    if p0.aborted:
        # R1: escalate as substrate mis-calibration finding
        return {"outcome": "commitment_not_elicited", "p0": p0}

    # P1 on target env (optional for simple arms)
    target_env = CausalGridWorldV2(size=10, num_hazards=4, ...)
    p1 = run_p1_consolidation(agent, target_env, device, budget=400)
    if p1.aborted:
        return {"outcome": "commitment_not_elicited", "p1": p1}

    # P2 emergent arm
    metrics_emergent = run_p2_eval(agent, target_env, device, n_eps=50)

    # P2 forced-rv control arm (O-2 mandatory contrast)
    agent_forced = clone_trained_agent(agent, bistable=True, device=device)
    agent_forced.e3._running_variance = 0.001
    metrics_forced = run_p2_eval(agent_forced, target_env, device, n_eps=50)

O-3 (gate threshold relaxation)
--------------------------------
At most ONE documented commitment_threshold step (0.40 -> 0.45).  Pass
threshold_relaxation=0.125 to run_p0_warmup once.  If commitment still does
not emerge, stop and escalate -- NOT another tuning step.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class P0Result:
    """Outcome of run_p0_warmup()."""
    converged: bool
    aborted: bool
    abort_reason: str          # "" if not aborted
    n_episodes: int
    final_rv: float
    commit_threshold_used: float
    nav_competence_final: float  # in-order waypoint rate (0.0 if unavailable)
    probe_log: List[Dict]


@dataclass
class P1Result:
    """Outcome of run_p1_consolidation()."""
    commitment_emerged: bool
    aborted: bool
    abort_reason: str
    n_episodes: int
    final_rv: float
    final_committed_steps_per_ep: float  # median over last stability_window
    probe_log: List[Dict]


@dataclass
class CommittedModeMetrics:
    """Eval metrics from run_p2_eval()."""
    total_committed_steps: int
    total_beta_elevated: int
    hold_rate: float              # beta_elevated / committed (0.0 if 0 committed)
    mean_committed_steps_per_ep: float
    rule_state_norm: float        # agent.lateral_pfc.rule_state.norm() if available
    n_eval_episodes: int
    per_episode: List[Dict]       # raw per-episode breakdown


# ---------------------------------------------------------------------------
# Internal training loop (shared between P0 and P1)
# ---------------------------------------------------------------------------

def _one_episode_train(
    agent,
    env,
    device: torch.device,
    e1_opt: optim.Optimizer,
    wf_opt: optim.Optimizer,
    wf_buf: Deque,
    batch_size: int,
    steps_per_episode: int,
    world_dim: int,
) -> Tuple[float, float, int]:
    """
    Run one training episode.

    Returns (final_rv, ep_reward, n_committed_steps_this_ep).
    Modifies agent, wf_buf, opts in place.
    """
    _, obs_dict = env.reset()
    agent.reset()

    z_world_prev: Optional[torch.Tensor] = None
    action_prev: Optional[torch.Tensor] = None
    ep_reward = 0.0
    ep_committed_steps = 0

    for _ in range(steps_per_episode):
        obs_body = obs_dict["body_state"].to(device)
        obs_world = obs_dict["world_state"].to(device)
        latent = agent.sense(obs_body, obs_world)
        z_world_curr = latent.z_world.detach()

        if z_world_prev is not None and action_prev is not None:
            wf_buf.append((
                z_world_prev.cpu(),
                action_prev.cpu(),
                z_world_curr.cpu(),
            ))

        ticks = agent.clock.advance()
        e1_prior = (
            agent._e1_tick(latent) if ticks.get("e1_tick")
            else torch.zeros(1, world_dim, device=device)
        )
        candidates = agent.generate_trajectories(latent, e1_prior, ticks)
        action = agent.select_action(candidates, ticks)
        action_idx = int(action.argmax(dim=-1).item())

        # E1 loss
        e1_opt.zero_grad()
        e1_loss = agent.compute_prediction_loss()
        if e1_loss.requires_grad:
            e1_loss.backward()
            torch.nn.utils.clip_grad_norm_(list(agent.e1.parameters()), 1.0)
            e1_opt.step()

        # E2 world-forward loss -- drives running_variance toward convergence
        if len(wf_buf) >= batch_size:
            k = min(batch_size, len(wf_buf))
            idxs = torch.randperm(len(wf_buf))[:k].tolist()
            zw_b = torch.cat([wf_buf[i][0] for i in idxs]).to(device)
            a_b = torch.cat([wf_buf[i][1] for i in idxs]).to(device)
            zw1_b = torch.cat([wf_buf[i][2] for i in idxs]).to(device)
            wf_pred = agent.e2.world_forward(zw_b, a_b)
            wf_loss = F.mse_loss(wf_pred, zw1_b)
            if wf_loss.requires_grad:
                wf_opt.zero_grad()
                wf_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(agent.e2.world_transition.parameters()) +
                    list(agent.e2.world_action_encoder.parameters()),
                    1.0,
                )
                wf_opt.step()
            with torch.no_grad():
                agent.e3.update_running_variance(
                    (wf_pred.detach() - zw1_b).detach()
                )

        if agent.e3._committed_trajectory is not None:
            ep_committed_steps += 1

        z_world_prev = z_world_curr
        action_prev = action.detach()
        _, harm_signal, done, _, obs_dict = env.step(action_idx)
        ep_reward += float(harm_signal)
        if done:
            break

    return float(agent.e3._running_variance), ep_reward, ep_committed_steps


# ---------------------------------------------------------------------------
# P0: world-model + navigation warmup
# ---------------------------------------------------------------------------

def run_p0_warmup(
    agent,
    env,
    device: torch.device,
    *,
    budget: int = 400,
    steps_per_episode: int = 200,
    lr_e1: float = 1e-4,
    lr_e2_wf: float = 1e-3,
    batch_size: int = 32,
    wf_buf_max: int = 2000,
    probe_interval: int = 40,
    mid_probe_frac: float = 0.60,
    convergence_stable_checkpoints: int = 3,
    nav_competence_floor: float = 0.0,
    threshold_relaxation: float = 0.0,
) -> P0Result:
    """
    Phase 0: train E1+E2 on easy env until running_variance crosses the
    commit gate.

    threshold_relaxation: fractional relaxation of commit_threshold (O-3).
        0.0 = use threshold as-is (0.40 default).
        0.125 = accept rv < 0.45 as converged (the single allowed step).
        Never call with a second non-zero value -- that is O-3 escalation territory.

    nav_competence_floor: minimum waypoint contact rate for exit.
        0.0 (default) means only rv gate matters.

    Returns P0Result.  If aborted, caller should treat this as an R1 substrate
    finding (gate mis-calibrated vs achievable world-model error) and stop.
    """
    agent.train()
    world_dim = agent.config.latent.world_dim
    base_threshold = float(agent.e3.commit_threshold)
    effective_threshold = base_threshold * (1.0 + threshold_relaxation)

    e1_opt = optim.Adam(list(agent.e1.parameters()), lr=lr_e1)
    wf_opt = optim.Adam(
        list(agent.e2.world_transition.parameters()) +
        list(agent.e2.world_action_encoder.parameters()),
        lr=lr_e2_wf,
    )
    wf_buf: Deque = deque(maxlen=wf_buf_max)

    mid_probe_episode = math.ceil(mid_probe_frac * budget)
    probe_log: List[Dict] = []
    stable_count = 0
    aborted = False
    abort_reason = ""
    nav_competence_final = 0.0

    for ep in range(budget):
        rv, ep_reward, ep_committed = _one_episode_train(
            agent, env, device, e1_opt, wf_opt, wf_buf,
            batch_size, steps_per_episode, world_dim,
        )

        if (ep + 1) % probe_interval == 0 or ep == budget - 1:
            probe = {
                "episode": ep + 1,
                "running_variance": rv,
                "effective_threshold": effective_threshold,
                "converging": rv < effective_threshold,
                "nav_competence": nav_competence_final,
            }
            probe_log.append(probe)
            print(
                f"  [P0 probe] ep {ep+1}/{budget}  rv={rv:.5f}"
                f"  threshold={effective_threshold:.6g}"
                f"  converging={rv < effective_threshold}",
                flush=True,
            )

            # Mid-probe abort (R1 detection)
            if (ep + 1) >= mid_probe_episode and rv >= effective_threshold:
                print(
                    f"  [P0 ABORT] rv={rv:.5f} >= threshold={effective_threshold:.6g}"
                    f" at ep={ep+1} ({int(mid_probe_frac*100)}% of budget={budget})."
                    f" Escalate as R1 substrate mis-calibration -- do NOT retune gate.",
                    flush=True,
                )
                aborted = True
                abort_reason = "commitment_not_elicited"
                break

            # Convergence: rv below threshold for stable_checkpoints consecutive probes
            if rv < effective_threshold and nav_competence_final >= nav_competence_floor:
                stable_count += 1
                if stable_count >= convergence_stable_checkpoints:
                    print(
                        f"  [P0 CONVERGED] rv={rv:.5f} < {effective_threshold:.6g}"
                        f" stable for {stable_count} probes. P0 done at ep={ep+1}.",
                        flush=True,
                    )
                    break
            else:
                stable_count = 0

        if (ep + 1) % (probe_interval * 2) == 0:
            print(
                f"  [P0 train] ep {ep+1}/{budget}  rv={rv:.5f}"
                f"  reward={ep_reward:.4f}",
                flush=True,
            )

    final_rv = float(agent.e3._running_variance)
    converged = final_rv < effective_threshold and not aborted

    return P0Result(
        converged=converged,
        aborted=aborted,
        abort_reason=abort_reason,
        n_episodes=min(ep + 1, budget),
        final_rv=final_rv,
        commit_threshold_used=effective_threshold,
        nav_competence_final=nav_competence_final,
        probe_log=probe_log,
    )


# ---------------------------------------------------------------------------
# P1: commitment consolidation
# ---------------------------------------------------------------------------

def run_p1_consolidation(
    agent,
    env,
    device: torch.device,
    *,
    budget: int = 400,
    steps_per_episode: int = 200,
    lr_e1: float = 1e-4,
    lr_e2_wf: float = 1e-3,
    batch_size: int = 32,
    wf_buf_max: int = 2000,
    probe_interval: int = 40,
    mid_probe_frac: float = 0.60,
    commitment_floor: int = 100,
    stability_window: int = 5,
) -> P1Result:
    """
    Phase 1: continue E1+E2 training on target (harder) env until
    total_committed_steps per episode is sustained >= commitment_floor.

    commitment_floor: acceptance criterion -- SD-021 substrate_queue sets 100.
    stability_window: number of consecutive episodes whose median must meet floor.

    Returns P1Result.  If aborted, treat as R1 escalation.
    """
    agent.train()
    world_dim = agent.config.latent.world_dim
    effective_threshold = float(agent.e3.commit_threshold)

    e1_opt = optim.Adam(list(agent.e1.parameters()), lr=lr_e1)
    wf_opt = optim.Adam(
        list(agent.e2.world_transition.parameters()) +
        list(agent.e2.world_action_encoder.parameters()),
        lr=lr_e2_wf,
    )
    wf_buf: Deque = deque(maxlen=wf_buf_max)

    mid_probe_episode = math.ceil(mid_probe_frac * budget)
    probe_log: List[Dict] = []
    committed_window: Deque[int] = deque(maxlen=stability_window)
    aborted = False
    abort_reason = ""
    commitment_emerged = False

    for ep in range(budget):
        rv, ep_reward, ep_committed = _one_episode_train(
            agent, env, device, e1_opt, wf_opt, wf_buf,
            batch_size, steps_per_episode, world_dim,
        )
        committed_window.append(ep_committed)

        if (ep + 1) % probe_interval == 0 or ep == budget - 1:
            median_committed = float(np.median(list(committed_window)))
            probe = {
                "episode": ep + 1,
                "running_variance": rv,
                "ep_committed_steps": ep_committed,
                "median_committed_window": median_committed,
                "emerging": median_committed >= commitment_floor,
            }
            probe_log.append(probe)
            print(
                f"  [P1 probe] ep {ep+1}/{budget}  rv={rv:.5f}"
                f"  ep_committed={ep_committed}"
                f"  median_window={median_committed:.1f}"
                f"  floor={commitment_floor}",
                flush=True,
            )

            # Mid-probe abort: commitment not emerged by mid_probe_frac
            if (ep + 1) >= mid_probe_episode and median_committed < 1:
                print(
                    f"  [P1 ABORT] no committed steps in probe window at ep={ep+1}"
                    f" ({int(mid_probe_frac*100)}% of budget={budget})."
                    f" Escalate -- curriculum cannot elicit commitment on this env.",
                    flush=True,
                )
                aborted = True
                abort_reason = "commitment_not_elicited"
                break

            # Exit criterion: sustained commitment above floor
            if len(committed_window) >= stability_window and median_committed >= commitment_floor:
                print(
                    f"  [P1 EMERGED] median committed steps/ep={median_committed:.1f}"
                    f" >= {commitment_floor} sustained over {stability_window} eps."
                    f" P1 done at ep={ep+1}.",
                    flush=True,
                )
                commitment_emerged = True
                break

        if (ep + 1) % (probe_interval * 2) == 0:
            print(
                f"  [P1 train] ep {ep+1}/{budget}  rv={rv:.5f}"
                f"  ep_committed={ep_committed}  reward={ep_reward:.4f}",
                flush=True,
            )

    median_final = float(np.median(list(committed_window))) if committed_window else 0.0

    return P1Result(
        commitment_emerged=commitment_emerged,
        aborted=aborted,
        abort_reason=abort_reason,
        n_episodes=min(ep + 1, budget),
        final_rv=float(agent.e3._running_variance),
        final_committed_steps_per_ep=median_final,
        probe_log=probe_log,
    )


# ---------------------------------------------------------------------------
# P2: frozen-policy evaluation
# ---------------------------------------------------------------------------

def run_p2_eval(
    agent,
    env,
    device: torch.device,
    *,
    n_eps: int = 50,
    steps_per_episode: int = 200,
) -> CommittedModeMetrics:
    """
    Phase 2: frozen-policy eval.  No gradient updates.

    Use for both the emergent arm and the forced-rv control arm (O-2).
    For the forced-rv arm: caller sets agent.e3._running_variance = 0.001
    before calling this function.
    """
    agent.eval()
    world_dim = agent.config.latent.world_dim

    total_committed_steps = 0
    total_beta_elevated = 0
    per_episode: List[Dict] = []

    rule_state_norms: List[float] = []

    with torch.no_grad():
        for ep in range(n_eps):
            _, obs_dict = env.reset()
            agent.reset()
            ep_committed = 0
            ep_elevated = 0

            for _ in range(steps_per_episode):
                obs_body = obs_dict["body_state"].to(device)
                obs_world = obs_dict["world_state"].to(device)
                latent = agent.sense(obs_body, obs_world)

                ticks = agent.clock.advance()
                e1_prior = (
                    agent._e1_tick(latent) if ticks.get("e1_tick")
                    else torch.zeros(1, world_dim, device=device)
                )
                candidates = agent.generate_trajectories(latent, e1_prior, ticks)
                action = agent.select_action(candidates, ticks)
                action_idx = int(action.argmax(dim=-1).item())

                if agent.e3._committed_trajectory is not None:
                    ep_committed += 1
                if agent.beta_gate.is_elevated:
                    ep_elevated += 1

                _, _, done, _, obs_dict = env.step(action_idx)
                if done:
                    break

            # Capture rule_state norm if SD-033a lateral PFC is enabled
            if hasattr(agent, "lateral_pfc") and agent.lateral_pfc is not None:
                rs_norm = float(agent.lateral_pfc.rule_state.norm().item())
                rule_state_norms.append(rs_norm)

            total_committed_steps += ep_committed
            total_beta_elevated += ep_elevated
            per_episode.append({
                "episode": ep,
                "committed_steps": ep_committed,
                "beta_elevated_steps": ep_elevated,
            })

    hold_rate = (
        total_beta_elevated / total_committed_steps
        if total_committed_steps > 0 else 0.0
    )
    mean_committed = total_committed_steps / max(1, n_eps)
    rule_state_norm = float(np.mean(rule_state_norms)) if rule_state_norms else 0.0

    return CommittedModeMetrics(
        total_committed_steps=total_committed_steps,
        total_beta_elevated=total_beta_elevated,
        hold_rate=hold_rate,
        mean_committed_steps_per_ep=mean_committed,
        rule_state_norm=rule_state_norm,
        n_eval_episodes=n_eps,
        per_episode=per_episode,
    )


# ---------------------------------------------------------------------------
# Clone helper (EXQ-321b clone_for_condition pattern)
# ---------------------------------------------------------------------------

def clone_trained_agent(trained_agent, bistable: bool, device: torch.device):
    """
    Clone trained_agent into a fresh REEAgent with the given bistable flag.

    Uses load_state_dict (deepcopy fails on autograd non-leaf tensors).
    Copies _running_variance manually so the gate threshold is respected.
    Resets the beta_gate so eval starts uncommitted.

    For the forced-rv control arm: after cloning, set
        agent.e3._running_variance = 0.001
    to simulate post-training convergence without emergent training.
    """
    import copy
    from ree_core.heartbeat.beta_gate import BetaGate
    from ree_core.agent import REEAgent

    # Deep-copy the entire config so all non-default fields are preserved,
    # then flip only the bistable flag.  Using from_dims() would lose any
    # config fields that from_dims doesn't expose.
    cfg_clone = copy.deepcopy(trained_agent.config)
    cfg_clone.heartbeat.beta_gate_bistable = bistable

    agent_clone = REEAgent(cfg_clone).to(device)

    state = {k: v.detach().clone() for k, v in trained_agent.state_dict().items()}
    try:
        agent_clone.load_state_dict(state)
    except RuntimeError:
        agent_clone.load_state_dict(state, strict=False)

    agent_clone.e3._running_variance = float(trained_agent.e3._running_variance)
    # Fresh gate -- eval starts uncommitted.
    # bistable=True: raise completion_release_threshold so only the variance
    # gate (not completion signal) controls release during standalone eval.
    agent_clone.beta_gate = BetaGate(
        completion_release_threshold=2.0 if bistable else 0.75
    )

    return agent_clone
