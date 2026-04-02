#!/opt/local/bin/python3
"""
V3-EXQ-202 -- MECH-026: Ready Vigilance via BreathOscillator

Claims: MECH-026
Supersedes: V3-EXQ-067

Motivation:
  Ready vigilance = heightened sensitivity (harm prediction responsiveness)
  during uncommitted threat-proximate steps compared to uncommitted neutral
  steps. EXQ-067 used a two-agent (threat vs neutral env) design that
  confounded environment exposure with vigilance. This redesign uses a SINGLE
  agent trained in a hazard-rich environment and classifies uncommitted eval
  steps by spatial proximity to hazards. The BreathOscillator creates periodic
  uncommitted windows, solving EXQ-067's problem of always-committed agents.

Design:
  - Train agent for 500 episodes (200 steps/ep), then eval for 50 episodes
  - BreathOscillator enabled: breath_period=50, sweep_amplitude=0.30,
    sweep_duration=10 -- creates ~20% uncommitted window per cycle
  - During eval, classify each uncommitted step as:
      "threat-proximate": Manhattan distance to nearest hazard <= 2
      "neutral":          Manhattan distance to nearest hazard > 2
  - Compare harm_eval magnitude between uncommitted-threat vs uncommitted-neutral
  - The "readiness gap" = higher harm_eval responsiveness near threat even
    without motor commitment

Config:
  - REEConfig defaults + alpha_world=0.9 (SD-008), use_event_classifier=True (SD-009)
  - CausalGridWorldV2(size=6, num_hazards=4)
  - Heartbeat: breath_period=50, breath_sweep_amplitude=0.30, breath_sweep_duration=10
  - nav_bias=0.45 (random exploration fraction in action selection)
  - 2 seeds: [42, 123]
  - Steps per episode: 200

Pre-registered PASS criteria (need 4/5):
  C1: uncommitted_step_count >= 50 per seed
  C2: uncommitted_threat_steps >= 20 per seed (enough near-hazard uncommitted samples)
  C3: readiness_gap > 0.01 (harm_eval_mean_threat > harm_eval_mean_neutral + 0.01)
  C4: harm_pred_std > 0.01 (E3 not collapsed)
  C5: no fatal errors
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


EXPERIMENT_TYPE = "v3_exq_202_mech026_vigilance_breath"
CLAIM_IDS = ["MECH-026"]

# Environment
ENV_SIZE = 6
NUM_HAZARDS = 4
BODY_OBS_DIM = 12   # CausalGridWorldV2 (use_proxy_fields=True)
WORLD_OBS_DIM = 250
ACTION_DIM = 5       # up, down, left, right, stay

# BreathOscillator parameters
BREATH_PERIOD = 50
SWEEP_AMPLITUDE = 0.30
SWEEP_DURATION = 10

# Threat proximity threshold (Manhattan distance)
THREAT_DISTANCE_THRESHOLD = 2

# Training
TRAIN_EPISODES = 500
EVAL_EPISODES = 50
STEPS_PER_EPISODE = 200
NAV_BIAS = 0.45
SEEDS = [42, 123]

# PASS thresholds
C1_MIN_UNCOMMITTED = 50
C2_MIN_THREAT_STEPS = 20
C3_READINESS_GAP = 0.01
C4_MIN_HARM_STD = 0.01


def _action_to_onehot(action_idx: int, n: int, device) -> torch.Tensor:
    v = torch.zeros(1, n, device=device)
    v[0, action_idx] = 1.0
    return v


def _mean_safe(lst: List[float]) -> float:
    return float(sum(lst) / len(lst)) if lst else 0.0


def _std_safe(lst: List[float]) -> float:
    if len(lst) < 2:
        return 0.0
    return float(torch.tensor(lst).std().item())


def _min_manhattan_to_hazard(env) -> int:
    """Manhattan distance from agent to nearest hazard."""
    ax, ay = env.agent_x, env.agent_y
    if not env.hazards:
        return 999
    return min(abs(ax - hx) + abs(ay - hy) for hx, hy in env.hazards)


def _make_env(seed: int) -> CausalGridWorldV2:
    return CausalGridWorldV2(
        seed=seed,
        size=ENV_SIZE,
        num_hazards=NUM_HAZARDS,
        num_resources=4,
        hazard_harm=0.02,
        env_drift_interval=5,
        env_drift_prob=0.1,
        proximity_harm_scale=0.05,
        proximity_benefit_scale=0.03,
        proximity_approach_threshold=0.15,
        hazard_field_decay=0.5,
    )


def _make_agent(env, seed: int) -> REEAgent:
    torch.manual_seed(seed)
    random.seed(seed)
    config = REEConfig.from_dims(
        body_obs_dim=env.body_obs_dim,
        world_obs_dim=env.world_obs_dim,
        action_dim=env.action_dim,
        self_dim=32,
        world_dim=32,
        alpha_world=0.9,       # SD-008
        alpha_self=0.3,
        reafference_action_dim=env.action_dim,
        use_event_classifier=True,  # SD-009
    )
    # BreathOscillator config
    config.heartbeat.breath_period = BREATH_PERIOD
    config.heartbeat.breath_sweep_amplitude = SWEEP_AMPLITUDE
    config.heartbeat.breath_sweep_duration = SWEEP_DURATION
    return REEAgent(config)


def _train(
    agent: REEAgent,
    env,
    num_episodes: int,
    steps_per_episode: int,
    world_dim: int,
    seed: int,
    nav_bias: float,
    dry_run: bool,
) -> Dict:
    """Standard training loop: E1 + E2 world_forward + E3 harm_eval."""
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
    total_harm = 0

    if dry_run:
        num_episodes = 3

    for ep in range(num_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()
        z_world_prev: Optional[torch.Tensor] = None
        action_prev: Optional[torch.Tensor] = None
        z_self_prev: Optional[torch.Tensor] = None

        for _ in range(steps_per_episode):
            obs_body = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            latent = agent.sense(obs_body, obs_world)

            if z_self_prev is not None and action_prev is not None:
                agent.record_transition(z_self_prev, action_prev, latent.z_self.detach())

            ticks = agent.clock.advance()
            e1_prior = (
                agent._e1_tick(latent) if ticks.get("e1_tick", False)
                else torch.zeros(1, world_dim, device=agent.device)
            )
            candidates = agent.generate_trajectories(latent, e1_prior, ticks)
            theta_z = agent.theta_buffer.summary()
            z_world_curr = latent.z_world.detach()

            # Action selection with nav_bias exploration
            if random.random() < nav_bias:
                action = _action_to_onehot(
                    random.randint(0, env.action_dim - 1), env.action_dim, agent.device
                )
                agent._last_action = action
            else:
                action = agent.select_action(candidates, ticks, temperature=1.0)
                if action is None:
                    action = _action_to_onehot(
                        random.randint(0, env.action_dim - 1), env.action_dim, agent.device
                    )
                    agent._last_action = action

            flat_obs, harm_signal, done, info, obs_dict = env.step(action)

            # E2 world forward training
            if z_world_prev is not None and action_prev is not None:
                wf_buf.append((z_world_prev.cpu(), action_prev.cpu(), z_world_curr.cpu()))
                if len(wf_buf) > 2000:
                    wf_buf = wf_buf[-2000:]

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
                            list(agent.e2.world_action_encoder.parameters()),
                            1.0,
                        )
                        wf_optimizer.step()
                    with torch.no_grad():
                        agent.e3.update_running_variance((wf_pred.detach() - zw1_b).detach())

            # Harm buffer for E3 training
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

            # E3 harm_eval training
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
            z_self_prev = latent.z_self.detach()
            action_prev = action.detach()
            if done:
                break

        if (ep + 1) % 100 == 0 or ep == num_episodes - 1:
            rv = agent.e3._running_variance
            print(
                f"  [seed={seed} train] ep {ep+1}/{num_episodes}"
                f"  harm={total_harm}  running_var={rv:.6f}",
                flush=True,
            )

    return {"total_harm": total_harm, "final_running_variance": agent.e3._running_variance}


def _eval_vigilance(
    agent: REEAgent,
    env,
    num_episodes: int,
    steps_per_episode: int,
    world_dim: int,
    seed: int,
    nav_bias: float,
    dry_run: bool,
) -> Dict:
    """
    Eval: classify uncommitted steps by proximity to hazards.
    Uncommitted = running_variance >= commit_threshold (agent not in doing mode).
    Threat-proximate = Manhattan distance to nearest hazard <= THREAT_DISTANCE_THRESHOLD.
    Record harm_eval output for each category.
    """
    agent.eval()
    agent.beta_gate.reset()

    harm_preds_threat: List[float] = []     # harm_eval at uncommitted threat-proximate steps
    harm_preds_neutral: List[float] = []    # harm_eval at uncommitted neutral steps
    all_harm_preds: List[float] = []        # all harm_eval values (for std check)
    uncommitted_step_count = 0
    committed_step_count = 0
    uncommitted_threat_count = 0
    uncommitted_neutral_count = 0
    fatal = 0

    if dry_run:
        num_episodes = 2

    for ep_i in range(num_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()

        for step_i in range(steps_per_episode):
            obs_body = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]

            try:
                with torch.no_grad():
                    latent = agent.sense(obs_body, obs_world)
                    ticks = agent.clock.advance()
                    e1_prior = (
                        agent._e1_tick(latent) if ticks.get("e1_tick", False)
                        else torch.zeros(1, world_dim, device=agent.device)
                    )
                    candidates = agent.generate_trajectories(latent, e1_prior, ticks)

                    # Action selection with nav_bias
                    if random.random() < nav_bias:
                        action = _action_to_onehot(
                            random.randint(0, env.action_dim - 1), env.action_dim, agent.device
                        )
                        agent._last_action = action
                    else:
                        action = agent.select_action(candidates, ticks, temperature=1.0)
                        if action is None:
                            action = _action_to_onehot(
                                random.randint(0, env.action_dim - 1),
                                env.action_dim, agent.device,
                            )
                            agent._last_action = action

                    is_committed = agent.e3._running_variance < agent.e3.commit_threshold
                    harm_pred = float(agent.e3.harm_eval(latent.z_world).item())
                    all_harm_preds.append(harm_pred)

                    if is_committed:
                        committed_step_count += 1
                    else:
                        uncommitted_step_count += 1
                        # Classify by proximity to nearest hazard
                        dist = _min_manhattan_to_hazard(env)
                        if dist <= THREAT_DISTANCE_THRESHOLD:
                            uncommitted_threat_count += 1
                            harm_preds_threat.append(harm_pred)
                        else:
                            uncommitted_neutral_count += 1
                            harm_preds_neutral.append(harm_pred)

            except Exception:
                fatal += 1
                action = _action_to_onehot(
                    random.randint(0, env.action_dim - 1), env.action_dim, agent.device
                )
                agent._last_action = action

            flat_obs, _, done, _, obs_dict = env.step(action)
            if done:
                break

    harm_eval_mean_threat = _mean_safe(harm_preds_threat)
    harm_eval_mean_neutral = _mean_safe(harm_preds_neutral)
    harm_eval_std_threat = _std_safe(harm_preds_threat)
    harm_eval_std_neutral = _std_safe(harm_preds_neutral)
    harm_pred_std = _std_safe(all_harm_preds)

    print(
        f"  [seed={seed} eval]"
        f"  uncommitted={uncommitted_step_count}"
        f"  (threat={uncommitted_threat_count}, neutral={uncommitted_neutral_count})"
        f"  committed={committed_step_count}"
        f"  harm_mean_threat={harm_eval_mean_threat:.4f}"
        f"  harm_mean_neutral={harm_eval_mean_neutral:.4f}"
        f"  harm_pred_std={harm_pred_std:.4f}"
        f"  fatal={fatal}",
        flush=True,
    )

    return {
        "uncommitted_step_count": uncommitted_step_count,
        "committed_step_count": committed_step_count,
        "uncommitted_threat_count": uncommitted_threat_count,
        "uncommitted_neutral_count": uncommitted_neutral_count,
        "harm_eval_mean_threat": harm_eval_mean_threat,
        "harm_eval_mean_neutral": harm_eval_mean_neutral,
        "harm_eval_std_threat": harm_eval_std_threat,
        "harm_eval_std_neutral": harm_eval_std_neutral,
        "harm_pred_std": harm_pred_std,
        "fatal_errors": fatal,
    }


def run(
    seeds: Optional[List[int]] = None,
    train_episodes: int = TRAIN_EPISODES,
    eval_episodes: int = EVAL_EPISODES,
    steps_per_episode: int = STEPS_PER_EPISODE,
    nav_bias: float = NAV_BIAS,
    dry_run: bool = False,
    **kwargs,
) -> dict:
    if seeds is None:
        seeds = list(SEEDS)

    world_dim = 32

    print(
        f"[V3-EXQ-202] MECH-026: Ready Vigilance via BreathOscillator\n"
        f"  train={train_episodes}  eval={eval_episodes}  steps={steps_per_episode}\n"
        f"  breath_period={BREATH_PERIOD}  sweep_amplitude={SWEEP_AMPLITUDE}"
        f"  sweep_duration={SWEEP_DURATION}\n"
        f"  nav_bias={nav_bias}  seeds={seeds}\n"
        f"  threat_distance_threshold={THREAT_DISTANCE_THRESHOLD}\n"
        f"  Supersedes: V3-EXQ-067",
        flush=True,
    )

    # -- Per-seed results -------------------------------------------------------
    seed_results: List[Dict] = []

    for seed in seeds:
        print(f"\n{'='*60}", flush=True)
        print(f"[V3-EXQ-202] Seed {seed}", flush=True)
        print("=" * 60, flush=True)

        torch.manual_seed(seed)
        random.seed(seed)

        env = _make_env(seed)
        agent = _make_agent(env, seed)

        _train(
            agent, env, train_episodes, steps_per_episode, world_dim,
            seed, nav_bias, dry_run,
        )
        eval_res = _eval_vigilance(
            agent, env, eval_episodes, steps_per_episode, world_dim,
            seed, nav_bias, dry_run,
        )
        seed_results.append(eval_res)

    # -- Aggregate across seeds -------------------------------------------------
    agg_uncommitted = sum(r["uncommitted_step_count"] for r in seed_results)
    agg_threat = sum(r["uncommitted_threat_count"] for r in seed_results)
    agg_neutral = sum(r["uncommitted_neutral_count"] for r in seed_results)

    # Pool all harm predictions for readiness gap
    all_threat_means = [r["harm_eval_mean_threat"] for r in seed_results]
    all_neutral_means = [r["harm_eval_mean_neutral"] for r in seed_results]
    harm_eval_mean_threat = _mean_safe(all_threat_means)
    harm_eval_mean_neutral = _mean_safe(all_neutral_means)
    readiness_gap = harm_eval_mean_threat - harm_eval_mean_neutral

    # harm_pred_std: average across seeds
    harm_pred_std = _mean_safe([r["harm_pred_std"] for r in seed_results])

    # Per-seed C1/C2 check (must hold for each seed)
    per_seed_c1 = all(r["uncommitted_step_count"] >= C1_MIN_UNCOMMITTED for r in seed_results)
    per_seed_c2 = all(r["uncommitted_threat_count"] >= C2_MIN_THREAT_STEPS for r in seed_results)

    total_fatal = sum(r["fatal_errors"] for r in seed_results)

    c1_pass = per_seed_c1
    c2_pass = per_seed_c2
    c3_pass = readiness_gap > C3_READINESS_GAP
    c4_pass = harm_pred_std > C4_MIN_HARM_STD
    c5_pass = total_fatal == 0

    criteria_met = sum([c1_pass, c2_pass, c3_pass, c4_pass, c5_pass])
    all_pass = criteria_met >= 4
    status = "PASS" if all_pass else "FAIL"

    print(f"\n{'='*60}", flush=True)
    print("[V3-EXQ-202] Aggregate Results", flush=True)
    print("=" * 60, flush=True)
    print(
        f"  total_uncommitted={agg_uncommitted}"
        f"  (threat={agg_threat}, neutral={agg_neutral})",
        flush=True,
    )
    print(
        f"  harm_eval_mean_threat={harm_eval_mean_threat:.4f}"
        f"  harm_eval_mean_neutral={harm_eval_mean_neutral:.4f}"
        f"  readiness_gap={readiness_gap:+.4f}",
        flush=True,
    )
    print(f"  harm_pred_std={harm_pred_std:.4f}  fatal={total_fatal}", flush=True)

    failure_notes = []
    if not c1_pass:
        per_vals = [r["uncommitted_step_count"] for r in seed_results]
        failure_notes.append(
            f"C1 FAIL: per-seed uncommitted_steps={per_vals} (need >= {C1_MIN_UNCOMMITTED} each)"
        )
    if not c2_pass:
        per_vals = [r["uncommitted_threat_count"] for r in seed_results]
        failure_notes.append(
            f"C2 FAIL: per-seed uncommitted_threat_steps={per_vals} (need >= {C2_MIN_THREAT_STEPS} each)"
        )
    if not c3_pass:
        failure_notes.append(
            f"C3 FAIL: readiness_gap={readiness_gap:+.4f} <= {C3_READINESS_GAP}"
        )
    if not c4_pass:
        failure_notes.append(
            f"C4 FAIL: harm_pred_std={harm_pred_std:.4f} <= {C4_MIN_HARM_STD}"
        )
    if not c5_pass:
        failure_notes.append(f"C5 FAIL: fatal_errors={total_fatal}")

    print(f"\nV3-EXQ-202 verdict: {status}  ({criteria_met}/5, need 4/5)", flush=True)
    for note in failure_notes:
        print(f"  {note}", flush=True)

    # -- Build per-seed detail table rows for markdown --------------------------
    seed_table_rows = ""
    for i, (s, r) in enumerate(zip(seeds, seed_results)):
        seed_table_rows += (
            f"| {s} | {r['uncommitted_step_count']} | {r['uncommitted_threat_count']}"
            f" | {r['uncommitted_neutral_count']} | {r['committed_step_count']}"
            f" | {r['harm_eval_mean_threat']:.4f} | {r['harm_eval_mean_neutral']:.4f}"
            f" | {r['harm_pred_std']:.4f} | {r['fatal_errors']} |\n"
        )

    failure_section = ""
    if failure_notes:
        failure_section = "\n## Failure Notes\n\n" + "\n".join(f"- {n}" for n in failure_notes)

    metrics = {
        "readiness_gap":                float(readiness_gap),
        "harm_eval_mean_threat":        float(harm_eval_mean_threat),
        "harm_eval_mean_neutral":       float(harm_eval_mean_neutral),
        "agg_uncommitted_step_count":   float(agg_uncommitted),
        "agg_uncommitted_threat_count": float(agg_threat),
        "agg_uncommitted_neutral_count":float(agg_neutral),
        "harm_pred_std":                float(harm_pred_std),
        "fatal_error_count":            float(total_fatal),
        "crit1_pass": 1.0 if c1_pass else 0.0,
        "crit2_pass": 1.0 if c2_pass else 0.0,
        "crit3_pass": 1.0 if c3_pass else 0.0,
        "crit4_pass": 1.0 if c4_pass else 0.0,
        "crit5_pass": 1.0 if c5_pass else 0.0,
        "criteria_met": float(criteria_met),
    }
    # Add per-seed metrics
    for i, (s, r) in enumerate(zip(seeds, seed_results)):
        for k, v in r.items():
            metrics[f"seed_{s}_{k}"] = float(v)

    summary_markdown = f"""# V3-EXQ-202 -- MECH-026: Ready Vigilance via BreathOscillator

**Status:** {status}
**Claim:** MECH-026 -- ready vigilance = heightened sensitivity without action commitment
**Supersedes:** V3-EXQ-067
**Design:** Single agent, within-eval spatial comparison (threat-proximate vs neutral uncommitted steps)
**BreathOscillator:** period={BREATH_PERIOD}, amplitude={SWEEP_AMPLITUDE}, duration={SWEEP_DURATION}
**Seeds:** {seeds} | **Train:** {train_episodes} eps | **Eval:** {eval_episodes} eps | **Steps:** {steps_per_episode}

## Design Rationale

EXQ-067 compared two separate agents (one trained with hazards, one without). This confounded
environment exposure with vigilance state. EXQ-202 trains a SINGLE agent in a hazard environment
and uses the BreathOscillator to create periodic uncommitted windows. During eval, uncommitted
steps are classified by spatial proximity: threat-proximate (Manhattan distance <= {THREAT_DISTANCE_THRESHOLD}
to nearest hazard) vs neutral (distance > {THREAT_DISTANCE_THRESHOLD}). The readiness gap measures
whether harm_eval output is higher near threats even during uncommitted (non-doing) steps.

## Per-Seed Results

| Seed | Uncommitted | Threat | Neutral | Committed | harm_mean_threat | harm_mean_neutral | harm_std | Fatal |
|------|-------------|--------|---------|-----------|-----------------|------------------|----------|-------|
{seed_table_rows}
## Aggregate

| Metric | Value |
|--------|-------|
| readiness_gap | {readiness_gap:+.4f} |
| harm_eval_mean_threat | {harm_eval_mean_threat:.4f} |
| harm_eval_mean_neutral | {harm_eval_mean_neutral:.4f} |
| total_uncommitted | {agg_uncommitted} |
| total_uncommitted_threat | {agg_threat} |
| total_uncommitted_neutral | {agg_neutral} |
| harm_pred_std (mean across seeds) | {harm_pred_std:.4f} |

## PASS Criteria (need 4/5)

| Criterion | Result | Value |
|---|---|---|
| C1: uncommitted_step_count >= {C1_MIN_UNCOMMITTED} per seed | {"PASS" if c1_pass else "FAIL"} | {[r['uncommitted_step_count'] for r in seed_results]} |
| C2: uncommitted_threat_steps >= {C2_MIN_THREAT_STEPS} per seed | {"PASS" if c2_pass else "FAIL"} | {[r['uncommitted_threat_count'] for r in seed_results]} |
| C3: readiness_gap > {C3_READINESS_GAP} | {"PASS" if c3_pass else "FAIL"} | {readiness_gap:+.4f} |
| C4: harm_pred_std > {C4_MIN_HARM_STD} | {"PASS" if c4_pass else "FAIL"} | {harm_pred_std:.4f} |
| C5: no fatal errors | {"PASS" if c5_pass else "FAIL"} | {total_fatal} |

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
        "fatal_error_count": float(total_fatal),
        "supersedes": "V3-EXQ-067",
    }


if __name__ == "__main__":
    import argparse
    import json
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--train-eps", type=int, default=TRAIN_EPISODES)
    parser.add_argument("--eval-eps", type=int, default=EVAL_EPISODES)
    parser.add_argument("--steps", type=int, default=STEPS_PER_EPISODE)
    parser.add_argument("--nav-bias", type=float, default=NAV_BIAS)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    result = run(
        seeds=args.seed,
        train_episodes=args.train_eps,
        eval_episodes=args.eval_eps,
        steps_per_episode=args.steps,
        nav_bias=args.nav_bias,
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
        if not k.startswith("seed_"):
            print(f"  {k}: {v}", flush=True)
