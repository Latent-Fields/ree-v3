"""
V3-EXQ-199 -- MECH-025: Action-Doing Mode Probe (BreathOscillator)

Claims: MECH-025
Supersedes: V3-EXQ-050b

Motivation:
  MECH-025: committed (doing) mode should show a distinct causal signature vs
  uncommitted (exploring) mode. V2 FAIL because SD-003 attribution and dynamic
  precision were not wired. EXQ-050b attempted to solve the uncommitted-window
  problem by calibrating commitment_threshold from training variance, but that
  approach is fragile -- after sufficient training, variance converges below any
  fixed multiple of the mean, yielding permanent commitment and C2 failure.

  This experiment uses the BreathOscillator (MECH-108) instead: a config-driven
  periodic oscillator that reduces the effective commit_threshold during sweep
  phases, guaranteeing periodic uncommitted windows regardless of training
  convergence. This is the biologically-motivated fix -- breathing rhythm
  creates natural exploration/exploitation oscillation.

  BreathOscillator config:
    breath_period=50:         full cycle length (steps)
    breath_sweep_amplitude=0.30:  fractional threshold reduction during sweep
    breath_sweep_duration=10: sweep phase duration per cycle

  With SD-008 (alpha_world=0.9) and SD-009 (event classifier), z_world responds
  to events and E2.world_forward is functional. The causal signature
    E3(E2(z_world, a_actual)) - E3(E2(z_world, a_cf))
  should be larger during committed steps (the agent is executing a deliberate
  trajectory and its actions have larger causal consequence) than during
  uncommitted/sweep steps (the agent is exploring, actions more uniform).

Design:
  - Train agent for 500 episodes (warmup), then eval for 50 episodes
  - BreathOscillator enabled via heartbeat config
  - During eval, record causal_signature per step
  - Separate into committed vs uncommitted steps (E3._committed_trajectory)
  - Compare mean |causal_sig| between modes

PASS criteria (ALL 5 must hold):
  C1: doing_mode_delta > 0.002       (committed has higher |causal sig|)
  C2: uncommitted_step_count >= 50   (enough uncommitted samples per seed)
  C3: world_forward_r2 > 0.05       (E2 world model functional)
  C4: harm_pred_std > 0.01          (E3 not collapsed)
  C5: No fatal errors

Seeds: [42, 123]
Steps per episode: 200
Env: CausalGridWorldV2(size=6, n_hazards=4, nav_bias=0.45)
"""

import sys
import json
import random
import argparse
import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F
import torch.optim as optim

from ree_core.agent import REEAgent
from ree_core.environment.causal_grid_world import CausalGridWorldV2
from ree_core.utils.config import REEConfig


EXPERIMENT_TYPE = "v3_exq_199_mech025_doing_mode_breath"
CLAIM_IDS = ["MECH-025"]


def _action_to_onehot(action_idx: int, n: int, device) -> torch.Tensor:
    v = torch.zeros(1, n, device=device)
    v[0, action_idx] = 1.0
    return v


def _mean_safe(lst: List[float]) -> float:
    return float(sum(lst) / len(lst)) if lst else 0.0


def _train(
    agent: REEAgent,
    env,
    optimizer: optim.Optimizer,
    wf_optimizer: optim.Optimizer,
    harm_eval_optimizer: optim.Optimizer,
    num_episodes: int,
    steps_per_episode: int,
    world_dim: int,
    nav_bias: float,
) -> Dict:
    """Standard full-pipeline training to get a functional E3 + E2.world_forward."""
    agent.train()
    harm_buf_pos: List[torch.Tensor] = []
    harm_buf_neg: List[torch.Tensor] = []
    wf_buf: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    total_harm = 0
    e3_tick_total = 0

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
            theta_z = agent.theta_buffer.summary()
            z_world_curr = latent.z_world.detach()

            if ticks.get("e3_tick", False) and candidates:
                e3_tick_total += 1
                result = agent.e3.select(candidates, temperature=1.0)
                action = result.selected_action.detach()
                agent._last_action = action
            else:
                action = agent._last_action
                if action is None:
                    action = _action_to_onehot(
                        random.randint(0, env.action_dim - 1), env.action_dim, agent.device
                    )
                    agent._last_action = action

            # nav_bias: with probability nav_bias, override action to move toward
            # nearest hazard -- increases harm exposure for training signal.
            if random.random() < nav_bias:
                agent_pos = getattr(env, "agent_pos", None)
                if agent_pos is not None and hasattr(env, "hazard_positions") and env.hazard_positions:
                    ax, ay = agent_pos
                    nearest = min(env.hazard_positions, key=lambda h: abs(h[0]-ax)+abs(h[1]-ay))
                    dx, dy = nearest[0] - ax, nearest[1] - ay
                    if abs(dx) >= abs(dy):
                        nav_act = 1 if dx > 0 else 0  # down / up
                    else:
                        nav_act = 3 if dy > 0 else 2  # right / left
                    action = _action_to_onehot(nav_act, env.action_dim, agent.device)
                    agent._last_action = action

            flat_obs, harm_signal, done, info, obs_dict = env.step(action)

            if z_world_prev is not None and action_prev is not None:
                wf_buf.append((z_world_prev.cpu(), action_prev.cpu(), z_world_curr.cpu()))
                if len(wf_buf) > 2000:
                    wf_buf = wf_buf[-2000:]

            if harm_signal < 0:
                total_harm += 1
                harm_buf_pos.append(theta_z.detach())
                if len(harm_buf_pos) > 1000:
                    harm_buf_pos = harm_buf_pos[-1000:]
            else:
                harm_buf_neg.append(theta_z.detach())
                if len(harm_buf_neg) > 1000:
                    harm_buf_neg = harm_buf_neg[-1000:]

            # E1 prediction loss
            e1_loss = agent.compute_prediction_loss()
            if e1_loss.requires_grad:
                optimizer.zero_grad()
                e1_loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.e1.parameters(), 1.0)
                optimizer.step()

            # E2 world forward training
            if len(wf_buf) >= 16:
                k = min(32, len(wf_buf))
                idxs = torch.randperm(len(wf_buf))[:k].tolist()
                zw_b  = torch.cat([wf_buf[i][0] for i in idxs]).to(agent.device)
                a_b   = torch.cat([wf_buf[i][1] for i in idxs]).to(agent.device)
                zw1_b = torch.cat([wf_buf[i][2] for i in idxs]).to(agent.device)
                wf_loss = F.mse_loss(agent.e2.world_forward(zw_b, a_b), zw1_b)
                if wf_loss.requires_grad:
                    wf_optimizer.zero_grad()
                    wf_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        list(agent.e2.world_transition.parameters()) +
                        list(agent.e2.world_action_encoder.parameters()),
                        1.0,
                    )
                    wf_optimizer.step()

            # E3 harm eval training
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
                    torch.nn.utils.clip_grad_norm_(
                        agent.e3.harm_eval_head.parameters(), 0.5
                    )
                    harm_eval_optimizer.step()

            z_world_prev = z_world_curr
            z_self_prev  = latent.z_self.detach()
            action_prev  = action.detach()
            if done:
                break

        if (ep + 1) % 100 == 0 or ep == num_episodes - 1:
            print(
                f"  [train] ep {ep+1}/{num_episodes}  harm={total_harm}"
                f"  e3_ticks={e3_tick_total}",
                flush=True,
            )

    return {
        "total_harm": total_harm,
        "wf_buf": wf_buf,
        "e3_tick_total": e3_tick_total,
    }


def _compute_world_forward_r2(agent: REEAgent, wf_buf: List, n_test: int = 200) -> float:
    if len(wf_buf) < n_test:
        return 0.0
    idxs = list(range(len(wf_buf) - n_test, len(wf_buf)))
    with torch.no_grad():
        zw  = torch.cat([wf_buf[i][0] for i in idxs]).to(agent.device)
        a   = torch.cat([wf_buf[i][1] for i in idxs]).to(agent.device)
        zw1 = torch.cat([wf_buf[i][2] for i in idxs]).to(agent.device)
        pred = agent.e2.world_forward(zw, a)
        ss_res = ((zw1 - pred) ** 2).sum()
        ss_tot = ((zw1 - zw1.mean(dim=0, keepdim=True)) ** 2).sum()
    return float((1 - ss_res / (ss_tot + 1e-8)).item())


def _eval_doing_mode(
    agent: REEAgent,
    env,
    num_episodes: int,
    steps_per_episode: int,
    world_dim: int,
) -> Dict:
    """
    Probe action-doing mode by comparing causal signature during committed vs
    uncommitted steps.  BreathOscillator creates periodic uncommitted windows
    via sweep_threshold_reduction in E3.select().

    causal_sig = E3(E2(z_world, a_actual)) - E3(E2(z_world, a_cf))
    """
    agent.eval()
    causal_sigs_committed:   List[float] = []
    causal_sigs_uncommitted: List[float] = []
    all_harm_preds: List[float] = []
    fatal = 0
    sweep_step_count = 0

    for _ in range(num_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()
        z_self_prev:  Optional[torch.Tensor] = None
        action_prev:  Optional[torch.Tensor] = None

        for _ in range(steps_per_episode):
            obs_body  = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]

            with torch.no_grad():
                latent = agent.sense(obs_body, obs_world)
                if z_self_prev is not None and action_prev is not None:
                    agent.record_transition(z_self_prev, action_prev, latent.z_self.detach())

                ticks    = agent.clock.advance()
                e1_prior = (
                    agent._e1_tick(latent) if ticks.get("e1_tick", False)
                    else torch.zeros(1, world_dim, device=agent.device)
                )
                candidates = agent.generate_trajectories(latent, e1_prior, ticks)

            try:
                # Note: agent.select_action() passes sweep_threshold_reduction
                # from BreathOscillator automatically.  We call agent.select_action
                # to get the full MECH-108 integration, but fall back to direct
                # E3.select() if the agent method is not available.
                if ticks.get("e3_tick", False) and candidates:
                    with torch.no_grad():
                        # Use the agent's select_action which includes sweep logic
                        if hasattr(agent, "select_action"):
                            action = agent.select_action(candidates, ticks, temperature=1.0)
                        else:
                            # Fallback: replicate sweep logic manually
                            sweep_reduction = (
                                agent.clock.sweep_amplitude
                                if agent.clock.sweep_active else 0.0
                            )
                            result = agent.e3.select(
                                candidates, temperature=1.0,
                                sweep_threshold_reduction=sweep_reduction,
                            )
                            action = result.selected_action.detach()
                            agent._last_action = action
                else:
                    action = agent._last_action
                    if action is None:
                        action = _action_to_onehot(
                            random.randint(0, env.action_dim - 1),
                            env.action_dim, agent.device,
                        )
                        agent._last_action = action

                # Track sweep state for diagnostics
                if agent.clock.sweep_active:
                    sweep_step_count += 1

                is_committed = agent.e3._committed_trajectory is not None

                # Compute causal signature via SD-003 counterfactual
                with torch.no_grad():
                    z_world = latent.z_world  # [1, world_dim]
                    # Counterfactual: random different action
                    actual_idx = action.argmax(dim=-1).item()
                    cf_idx = (random.randint(0, env.action_dim - 2) + 1 + actual_idx) % env.action_dim
                    a_cf = _action_to_onehot(int(cf_idx), env.action_dim, agent.device)

                    z_actual = agent.e2.world_forward(z_world, action)
                    z_cf     = agent.e2.world_forward(z_world, a_cf)
                    h_actual = float(agent.e3.harm_eval(z_actual).item())
                    h_cf     = float(agent.e3.harm_eval(z_cf).item())
                    causal_sig = h_actual - h_cf
                    all_harm_preds.append(h_actual)

                if is_committed:
                    causal_sigs_committed.append(causal_sig)
                else:
                    causal_sigs_uncommitted.append(causal_sig)

            except Exception:
                fatal += 1
                action = _action_to_onehot(
                    random.randint(0, env.action_dim - 1), env.action_dim, agent.device
                )
                agent._last_action = action

            flat_obs, _, done, info, obs_dict = env.step(action)
            z_self_prev = latent.z_self.detach()
            action_prev = action.detach()
            if done:
                break

    mean_committed   = _mean_safe([abs(x) for x in causal_sigs_committed])
    mean_uncommitted = _mean_safe([abs(x) for x in causal_sigs_uncommitted])
    doing_mode_delta = mean_committed - mean_uncommitted
    harm_pred_std = float(
        torch.tensor(all_harm_preds).std().item()
    ) if len(all_harm_preds) > 1 else 0.0

    print(
        f"  |causal_sig| committed={mean_committed:.4f}  uncommitted={mean_uncommitted:.4f}"
        f"  doing_mode_delta={doing_mode_delta:+.4f}"
        f"  n_committed={len(causal_sigs_committed)}  n_uncommitted={len(causal_sigs_uncommitted)}"
        f"  sweep_steps={sweep_step_count}",
        flush=True,
    )

    return {
        "mean_abs_causal_sig_committed":   mean_committed,
        "mean_abs_causal_sig_uncommitted": mean_uncommitted,
        "doing_mode_delta":                doing_mode_delta,
        "committed_step_count":            len(causal_sigs_committed),
        "uncommitted_step_count":          len(causal_sigs_uncommitted),
        "harm_pred_std":                   harm_pred_std,
        "fatal_errors":                    fatal,
        "sweep_step_count":                sweep_step_count,
    }


def run(
    seed: int = 0,
    warmup_episodes: int = 500,
    eval_episodes: int = 50,
    steps_per_episode: int = 200,
    alpha_world: float = 0.9,
    alpha_self: float = 0.3,
    harm_scale: float = 0.02,
    proximity_scale: float = 0.05,
    nav_bias: float = 0.45,
    lr: float = 1e-3,
    self_dim: int = 32,
    world_dim: int = 32,
    breath_period: int = 50,
    breath_sweep_amplitude: float = 0.30,
    breath_sweep_duration: int = 10,
    **kwargs,
) -> dict:
    torch.manual_seed(seed)
    random.seed(seed)

    env = CausalGridWorldV2(
        seed=seed, size=6, num_hazards=4, num_resources=5,
        hazard_harm=harm_scale,
        env_drift_interval=5, env_drift_prob=0.1,
        proximity_harm_scale=proximity_scale,
        proximity_benefit_scale=proximity_scale * 0.6,
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
        reafference_action_dim=env.action_dim,
        use_event_classifier=True,   # SD-009
    )
    # BreathOscillator config (MECH-108)
    config.heartbeat.breath_period = breath_period
    config.heartbeat.breath_sweep_amplitude = breath_sweep_amplitude
    config.heartbeat.breath_sweep_duration = breath_sweep_duration

    agent = REEAgent(config)

    optimizer = optim.Adam(list(agent.e1.parameters()), lr=lr)
    wf_optimizer = optim.Adam(
        list(agent.e2.world_transition.parameters()) +
        list(agent.e2.world_action_encoder.parameters()),
        lr=1e-3,
    )
    harm_eval_optimizer = optim.Adam(
        list(agent.e3.harm_eval_head.parameters()), lr=1e-4,
    )

    print(
        f"[V3-EXQ-199] MECH-025: Action-Doing Mode Probe (BreathOscillator)\n"
        f"  seed={seed}  warmup={warmup_episodes}  eval={eval_episodes}"
        f"  alpha_world={alpha_world}\n"
        f"  breath_period={breath_period}  sweep_amplitude={breath_sweep_amplitude}"
        f"  sweep_duration={breath_sweep_duration}\n"
        f"  nav_bias={nav_bias}  size=6  n_hazards=4",
        flush=True,
    )

    train_out = _train(
        agent, env, optimizer, wf_optimizer, harm_eval_optimizer,
        warmup_episodes, steps_per_episode, world_dim, nav_bias,
    )
    world_forward_r2 = _compute_world_forward_r2(agent, train_out["wf_buf"])
    print(f"  world_forward_r2: {world_forward_r2:.4f}", flush=True)

    print(f"\n[V3-EXQ-199] Eval -- probing action-doing mode...", flush=True)
    eval_out = _eval_doing_mode(agent, env, eval_episodes, steps_per_episode, world_dim)

    # PASS / FAIL
    c1_pass = eval_out["doing_mode_delta"] > 0.002
    c2_pass = eval_out["uncommitted_step_count"] >= 50
    c3_pass = world_forward_r2 > 0.05
    c4_pass = eval_out["harm_pred_std"] > 0.01
    c5_pass = eval_out["fatal_errors"] == 0

    all_pass    = c1_pass and c2_pass and c3_pass and c4_pass and c5_pass
    status      = "PASS" if all_pass else "FAIL"
    criteria_met = sum([c1_pass, c2_pass, c3_pass, c4_pass, c5_pass])

    failure_notes = []
    if not c1_pass:
        failure_notes.append(
            f"C1 FAIL: doing_mode_delta={eval_out['doing_mode_delta']:.4f} <= 0.002"
        )
    if not c2_pass:
        failure_notes.append(
            f"C2 FAIL: uncommitted_step_count={eval_out['uncommitted_step_count']} < 50"
        )
    if not c3_pass:
        failure_notes.append(f"C3 FAIL: world_forward_r2={world_forward_r2:.4f} <= 0.05")
    if not c4_pass:
        failure_notes.append(
            f"C4 FAIL: harm_pred_std={eval_out['harm_pred_std']:.4f} <= 0.01"
        )
    if not c5_pass:
        failure_notes.append(f"C5 FAIL: fatal_errors={eval_out['fatal_errors']}")

    print(f"\nV3-EXQ-199 verdict: {status}  ({criteria_met}/5)", flush=True)
    for note in failure_notes:
        print(f"  {note}", flush=True)

    metrics = {
        "mean_abs_causal_sig_committed":   float(eval_out["mean_abs_causal_sig_committed"]),
        "mean_abs_causal_sig_uncommitted": float(eval_out["mean_abs_causal_sig_uncommitted"]),
        "doing_mode_delta":                float(eval_out["doing_mode_delta"]),
        "committed_step_count":            float(eval_out["committed_step_count"]),
        "uncommitted_step_count":          float(eval_out["uncommitted_step_count"]),
        "harm_pred_std":                   float(eval_out["harm_pred_std"]),
        "world_forward_r2":               float(world_forward_r2),
        "e3_tick_total":                   float(train_out["e3_tick_total"]),
        "total_harm_train":                float(train_out["total_harm"]),
        "fatal_error_count":               float(eval_out["fatal_errors"]),
        "sweep_step_count":                float(eval_out["sweep_step_count"]),
        "breath_period":                   float(breath_period),
        "breath_sweep_amplitude":          float(breath_sweep_amplitude),
        "breath_sweep_duration":           float(breath_sweep_duration),
        "crit1_pass": 1.0 if c1_pass else 0.0,
        "crit2_pass": 1.0 if c2_pass else 0.0,
        "crit3_pass": 1.0 if c3_pass else 0.0,
        "crit4_pass": 1.0 if c4_pass else 0.0,
        "crit5_pass": 1.0 if c5_pass else 0.0,
        "criteria_met": float(criteria_met),
    }

    failure_section = ""
    if failure_notes:
        failure_section = "\n## Failure Notes\n\n" + "\n".join(f"- {n}" for n in failure_notes)

    summary_markdown = f"""# V3-EXQ-199 -- MECH-025: Action-Doing Mode Probe (BreathOscillator)

**Status:** {status}
**Claim:** MECH-025 -- action-doing mode produces distinct internal signature
**Supersedes:** V3-EXQ-050b (manual threshold calibration -> BreathOscillator)
**Key change:** BreathOscillator (MECH-108) creates periodic uncommitted windows
  via config-driven sweep phases, replacing fragile variance-calibrated threshold.
**alpha_world:** {alpha_world} (SD-008)
**use_event_classifier:** True (SD-009)
**Warmup:** {warmup_episodes} eps  |  Eval: {eval_episodes} eps
**Seed:** {seed}

## BreathOscillator Config

| Parameter | Value |
|-----------|-------|
| breath_period | {breath_period} |
| breath_sweep_amplitude | {breath_sweep_amplitude} |
| breath_sweep_duration | {breath_sweep_duration} |
| sweep_steps_observed (eval) | {eval_out['sweep_step_count']} |

## Motivation

MECH-025 (V2 FAIL): agent in doing mode should show a distinct internal signature.
V3 fix: SD-003 attribution works (EXQ-030b PASS). BreathOscillator (MECH-108)
creates periodic uncommitted windows by reducing the effective commit_threshold
during sweep phases. During committed action execution, the causal signature
E3(E2(z,a_actual)) - E3(E2(z,a_cf)) should be higher than during free
exploration (uncommitted/sweep steps).

Committed state read from: agent.e3._committed_trajectory is not None

## Causal Signature by Mode

| Mode | mean |causal_sig| | n_steps |
|------|---------------------|---------|
| Committed (doing) | {eval_out['mean_abs_causal_sig_committed']:.4f} | {eval_out['committed_step_count']} |
| Uncommitted (exploring) | {eval_out['mean_abs_causal_sig_uncommitted']:.4f} | {eval_out['uncommitted_step_count']} |

- **doing_mode_delta**: {eval_out['doing_mode_delta']:+.4f}  (committed - uncommitted)
- world_forward_r2: {world_forward_r2:.4f}
- harm_pred_std: {eval_out['harm_pred_std']:.4f}

## PASS Criteria

| Criterion | Result | Value |
|---|---|---|
| C1: doing_mode_delta > 0.002 (committed has higher |causal sig|) | {"PASS" if c1_pass else "FAIL"} | {eval_out['doing_mode_delta']:+.4f} |
| C2: uncommitted_step_count >= 50 (enough uncommitted samples) | {"PASS" if c2_pass else "FAIL"} | {eval_out['uncommitted_step_count']} |
| C3: world_forward_r2 > 0.05 (E2 functional) | {"PASS" if c3_pass else "FAIL"} | {world_forward_r2:.4f} |
| C4: harm_pred_std > 0.01 (E3 not collapsed) | {"PASS" if c4_pass else "FAIL"} | {eval_out['harm_pred_std']:.4f} |
| C5: No fatal errors | {"PASS" if c5_pass else "FAIL"} | {eval_out['fatal_errors']} |

Criteria met: {criteria_met}/5 -> **{status}**
{failure_section}
"""

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
        "fatal_error_count": eval_out["fatal_errors"],
    }


def _run_multi_seed(seeds, **run_kwargs) -> dict:
    """Run across multiple seeds, aggregate metrics, produce combined result."""
    per_seed = {}
    all_metrics_keys = None

    for seed in seeds:
        print(f"\n{'='*60}", flush=True)
        print(f"  Seed {seed}", flush=True)
        print(f"{'='*60}", flush=True)
        result = run(seed=seed, **run_kwargs)
        per_seed[seed] = result
        if all_metrics_keys is None:
            all_metrics_keys = list(result["metrics"].keys())

    # Aggregate: mean across seeds for numeric metrics
    agg_metrics = {}
    for key in all_metrics_keys:
        vals = [per_seed[s]["metrics"][key] for s in seeds]
        agg_metrics[key] = _mean_safe(vals)

    # Overall verdict: ALL seeds must pass all criteria
    all_pass = all(per_seed[s]["status"] == "PASS" for s in seeds)
    overall_status = "PASS" if all_pass else "FAIL"
    total_criteria = sum(int(per_seed[s]["metrics"]["criteria_met"]) for s in seeds)
    max_criteria = 5 * len(seeds)

    # Evidence direction from aggregate
    if all_pass:
        evidence_direction = "supports"
    elif total_criteria >= 3 * len(seeds):
        evidence_direction = "mixed"
    else:
        evidence_direction = "weakens"

    # Per-seed summary lines
    seed_lines = []
    for s in seeds:
        m = per_seed[s]["metrics"]
        seed_lines.append(
            f"| {s} | {per_seed[s]['status']} | {int(m['criteria_met'])}/5 |"
            f" {m['doing_mode_delta']:+.4f} |"
            f" {int(m['uncommitted_step_count'])} |"
            f" {m['world_forward_r2']:.4f} |"
            f" {m['harm_pred_std']:.4f} |"
        )
    seed_table = "\n".join(seed_lines)

    failure_notes = []
    for s in seeds:
        if per_seed[s]["status"] != "PASS":
            for line in per_seed[s]["summary_markdown"].split("\n"):
                if line.startswith("- C") and "FAIL" in line:
                    failure_notes.append(f"seed {s}: {line.strip('- ')}")

    failure_section = ""
    if failure_notes:
        failure_section = "\n## Failure Notes\n\n" + "\n".join(f"- {n}" for n in failure_notes)

    bp = run_kwargs.get("breath_period", 50)
    bsa = run_kwargs.get("breath_sweep_amplitude", 0.30)
    bsd = run_kwargs.get("breath_sweep_duration", 10)

    summary_markdown = f"""# V3-EXQ-199 -- MECH-025: Action-Doing Mode Probe (BreathOscillator)

**Overall Status:** {overall_status}  ({total_criteria}/{max_criteria} criteria across {len(seeds)} seeds)
**Claim:** MECH-025 -- action-doing mode produces distinct internal signature
**Supersedes:** V3-EXQ-050b
**Key change:** BreathOscillator (MECH-108) creates periodic uncommitted windows
**Seeds:** {seeds}
**BreathOscillator:** period={bp}, amplitude={bsa}, duration={bsd}

## Per-Seed Results

| Seed | Status | Criteria | doing_mode_delta | n_uncommitted | wf_r2 | harm_std |
|------|--------|----------|-----------------|---------------|-------|----------|
{seed_table}

## Aggregate Metrics

| Metric | Value |
|--------|-------|
| doing_mode_delta (mean) | {agg_metrics['doing_mode_delta']:+.4f} |
| uncommitted_step_count (mean) | {agg_metrics['uncommitted_step_count']:.0f} |
| world_forward_r2 (mean) | {agg_metrics['world_forward_r2']:.4f} |
| harm_pred_std (mean) | {agg_metrics['harm_pred_std']:.4f} |
| sweep_step_count (mean) | {agg_metrics['sweep_step_count']:.0f} |
{failure_section}
"""

    return {
        "status": overall_status,
        "metrics": agg_metrics,
        "per_seed_metrics": {str(s): per_seed[s]["metrics"] for s in seeds},
        "per_seed_status": {str(s): per_seed[s]["status"] for s in seeds},
        "summary_markdown": summary_markdown,
        "claim_ids": CLAIM_IDS,
        "evidence_direction": evidence_direction,
        "experiment_type": EXPERIMENT_TYPE,
        "fatal_error_count": int(agg_metrics.get("fatal_error_count", 0)),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds",       type=str,   default="42,123")
    parser.add_argument("--warmup",      type=int,   default=500)
    parser.add_argument("--eval-eps",    type=int,   default=50)
    parser.add_argument("--steps",       type=int,   default=200)
    parser.add_argument("--alpha-world", type=float, default=0.9)
    parser.add_argument("--harm-scale",  type=float, default=0.02)
    parser.add_argument("--nav-bias",    type=float, default=0.45)
    parser.add_argument("--breath-period",    type=int,   default=50)
    parser.add_argument("--breath-amplitude", type=float, default=0.30)
    parser.add_argument("--breath-duration",  type=int,   default=10)
    parser.add_argument("--dry-run",     action="store_true",
                        help="Quick smoke test with minimal episodes")
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",")]

    run_kwargs = dict(
        warmup_episodes=args.warmup,
        eval_episodes=args.eval_eps,
        steps_per_episode=args.steps,
        alpha_world=args.alpha_world,
        harm_scale=args.harm_scale,
        nav_bias=args.nav_bias,
        breath_period=args.breath_period,
        breath_sweep_amplitude=args.breath_amplitude,
        breath_sweep_duration=args.breath_duration,
    )

    if args.dry_run:
        run_kwargs["warmup_episodes"] = 5
        run_kwargs["eval_episodes"] = 3
        run_kwargs["steps_per_episode"] = 30
        seeds = [42]
        print("[DRY RUN] Minimal config for smoke test", flush=True)

    result = _run_multi_seed(seeds, **run_kwargs)

    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result["run_timestamp"] = ts
    result["claim"] = CLAIM_IDS[0]
    result["verdict"] = result["status"]
    result["run_id"] = f"{EXPERIMENT_TYPE}_{ts}_v3"
    result["architecture_epoch"] = "ree_hybrid_guardrails_v1"
    result["supersedes"] = "V3-EXQ-050b"

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
