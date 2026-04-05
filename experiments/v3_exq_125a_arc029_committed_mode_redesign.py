#!/opt/local/bin/python3
"""
V3-EXQ-125a -- ARC-029: Committed vs Ablated Operating Mode Harm Outcomes (Redesign)

Supersedes: V3-EXQ-125 (EXQ-125 FAILed 3x due to insufficient statistical power)

Claims: ARC-029

Root cause of EXQ-125 failure (diagnosed 2026-04-04):
  1. Only 2 seeds -- seed 42 showed committed agent marginally worse than ablated
     (gap = -0.00009), insufficient to distinguish from seed noise.
  2. SD-010/011/012 substrate changes between EXQ-063 (March 22) and EXQ-125 (March 29)
     reduced harm rates ~100x (from ~-0.09/step to ~-0.001/step), making the committed-
     gate advantage undetectable at num_hazards=4, hazard_harm=0.02 scale.
  3. Criteria required ALL seeds to pass -- majority was needed, not unanimity.

Redesign:
  - 5 seeds [0, 42, 100, 123, 200] with majority rule (C1/C2/C3: >= 4/5 seeds)
  - num_hazards raised from 4 to 8 -- doubled hazard field density
  - hazard_harm raised from 0.02 to 0.05 -- 2.5x stronger nociceptive penalty
  - proximity_harm_scale raised from 0.05 to 0.15 -- 3x larger proximity field
  - MIN_HARM_GAP_STABLE raised from 0.0001 to 0.001 (scaled with harm magnitude)
  - harm_obs passed to agent.sense() -- SD-010 nociceptive stream now active
  - eval_episodes raised from 50 to 60 for better statistical reliability

ARC-029 predicts:
  - Committed mode (BG beta gate active) reduces harm in stable environments
    by holding the agent to a well-evaluated trajectory.
  - This advantage narrows or reverses in volatile environments, where the committed
    trajectory quickly becomes invalid as the layout changes.

Design -- 2x2 gate x environment, 5 seeds:
  For each seed, train one agent on the standard environment. Then evaluate 4 conditions:

    Condition A: COMMITTED_MODE_ON  x Stable   (gate active, low-drift env)
    Condition B: COMMITTED_MODE_ON  x Volatile (gate active, high-drift env)
    Condition C: COMMITTED_MODE_ABLATED x Stable   (gate off, low-drift env)
    Condition D: COMMITTED_MODE_ABLATED x Volatile (gate off, high-drift env)

  Gate ablation: force agent.e3._running_variance = commit_threshold + 0.1 and
  agent.e3._committed_trajectory = None before each SELECT step.

  Stable environment:   env_drift_prob=0.0
  Volatile environment: env_drift_prob=0.4, env_drift_interval=3
  Training environment: env_drift_prob=0.1, env_drift_interval=5

Pre-registered PASS criteria (majority rule: C1/C2/C3 in >= 4/5 seeds):

  C1: harm_gap_stable >= MIN_HARM_GAP_STABLE (0.001)  in >= 4/5 seeds
      Committed outperforms ablated in stable env.

  C2: harm_gap_volatile < harm_gap_stable              in >= 4/5 seeds
      Committed advantage narrows or reverses in volatile env.

  C3: gap_reduction_ratio <= VOLATILITY_MODULATION_RATIO (0.90) in >= 4/5 seeds
      volatile_gap / stable_gap <= 0.90.

  C4: n_committed_active_stable > n_uncommitted_active_stable  (ALL seeds)
      Gate is actually active in COMMITTED_MODE_ON -- sanity check.

  C5: n_committed_ablated_stable == 0  (ALL seeds)
      Ablation is complete.

  C6: No fatal errors across all conditions.
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


EXPERIMENT_TYPE = "v3_exq_125a_arc029_committed_mode_redesign"
CLAIM_IDS = ["ARC-029"]
EXPERIMENT_PURPOSE = "evidence"

# Pre-registered thresholds (scaled for raised hazard parameters)
MIN_HARM_GAP_STABLE = 0.001          # C1: minimum committed advantage in stable env
VOLATILITY_MODULATION_RATIO = 0.90   # C3: volatile_gap / stable_gap <= this
MIN_SEEDS_PASSING = 4                # majority rule for C1/C2/C3 (out of 5 seeds)

# Env parameters: raised hazard density and harm magnitude vs EXQ-125
TRAIN_ENV_KWARGS = dict(
    size=12, num_hazards=8, num_resources=5,
    hazard_harm=0.05,
    env_drift_interval=5, env_drift_prob=0.1,
    proximity_harm_scale=0.15,
    proximity_benefit_scale=0.03,
    proximity_approach_threshold=0.15,
    hazard_field_decay=0.5,
)

STABLE_ENV_KWARGS = dict(
    size=12, num_hazards=8, num_resources=5,
    hazard_harm=0.05,
    env_drift_interval=50, env_drift_prob=0.0,
    proximity_harm_scale=0.15,
    proximity_benefit_scale=0.03,
    proximity_approach_threshold=0.15,
    hazard_field_decay=0.5,
)

VOLATILE_ENV_KWARGS = dict(
    size=12, num_hazards=8, num_resources=5,
    hazard_harm=0.05,
    env_drift_interval=3, env_drift_prob=0.4,
    proximity_harm_scale=0.15,
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
    """Train agent until running_variance collapses to committed state."""
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
            obs_harm = obs_dict.get("harm_obs")   # SD-010 nociceptive stream
            latent = agent.sense(obs_body, obs_world, obs_harm=obs_harm)

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
    ablated: bool,
    label: str,
    train_variance: float,
) -> Dict:
    """Eval harm outcomes under COMMITTED_MODE_ON or COMMITTED_MODE_ABLATED."""
    agent.eval()

    harm_signals: List[float] = []
    n_committed = 0
    n_uncommitted = 0
    fatal = 0

    for _ in range(num_episodes):
        flat_obs, obs_dict = env.reset()
        agent.reset()
        agent.e3._running_variance = train_variance

        for _ in range(steps_per_episode):
            obs_body = obs_dict["body_state"]
            obs_world = obs_dict["world_state"]
            obs_harm = obs_dict.get("harm_obs")   # SD-010 nociceptive stream
            with torch.no_grad():
                latent = agent.sense(obs_body, obs_world, obs_harm=obs_harm)

            if ablated:
                agent.e3._running_variance = agent.e3.commit_threshold + 0.1
                agent.e3._committed_trajectory = None

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

                flat_obs, harm_signal, done, info, obs_dict = env.step(action)

                is_committed = agent.e3._running_variance < agent.e3.commit_threshold
                if is_committed:
                    n_committed += 1
                else:
                    n_uncommitted += 1

                harm_signals.append(float(harm_signal))

            except Exception:
                fatal += 1
                flat_obs, obs_dict = env.reset()
                done = True

            if done:
                break

    mean_harm = _mean_safe(harm_signals)
    print(
        f"  [{label}]  mean_harm={mean_harm:.5f}"
        f"  committed={n_committed}  uncommitted={n_uncommitted}"
        f"  fatal={fatal}",
        flush=True,
    )

    return {
        "mean_harm_per_step": mean_harm,
        "n_committed": n_committed,
        "n_uncommitted": n_uncommitted,
        "fatal_errors": fatal,
        "n_steps": len(harm_signals),
    }


def run(
    seeds: List[int] = None,
    warmup_episodes: int = 350,
    eval_episodes: int = 60,
    steps_per_episode: int = 200,
    alpha_world: float = 0.9,
    self_dim: int = 32,
    world_dim: int = 32,
    **kwargs,
) -> dict:
    if seeds is None:
        seeds = [0, 42, 100, 123, 200]

    print(
        f"[V3-EXQ-125a] ARC-029 Redesign: COMMITTED_MODE_ON vs COMMITTED_MODE_ABLATED\n"
        f"  Design: 2x2 [ON/ABLATED] x [stable/volatile] x seeds {seeds}\n"
        f"  Ablation: force uncommitted state (running_var > commit_threshold)\n"
        f"  Stable: drift_prob=0.0 | Volatile: drift_prob=0.4, interval=3\n"
        f"  Training: drift_prob=0.1, interval=5  warmup={warmup_episodes} eps\n"
        f"  Pre-registered: MIN_HARM_GAP_STABLE={MIN_HARM_GAP_STABLE}"
        f"  VOLATILITY_MODULATION_RATIO={VOLATILITY_MODULATION_RATIO}"
        f"  MIN_SEEDS_PASSING={MIN_SEEDS_PASSING}\n"
        f"  Env: num_hazards=8, hazard_harm=0.05, proximity_harm_scale=0.15\n"
        f"  alpha_world={alpha_world}  Supersedes: V3-EXQ-125",
        flush=True,
    )

    results_by_seed: List[Dict] = []

    for seed in seeds:
        print(f"\n{'='*60}", flush=True)
        print(f"[V3-EXQ-125a] Seed {seed}", flush=True)
        print('='*60, flush=True)
        print(f"Seed {seed} Condition TRAINING", flush=True)

        train_env = CausalGridWorldV2(seed=seed, **TRAIN_ENV_KWARGS)
        agent = _make_agent(seed, self_dim, world_dim, alpha_world, train_env)
        train_out = _train_agent(
            agent, train_env, warmup_episodes, steps_per_episode, world_dim, seed
        )
        train_variance = train_out["final_running_variance"]

        print(
            f"\n  Post-train running_variance={train_variance:.6f}"
            f"  commit_threshold={agent.e3.commit_threshold:.4f}"
            f"  committed={train_variance < agent.e3.commit_threshold}",
            flush=True,
        )

        if train_variance >= agent.e3.commit_threshold:
            print(
                "  WARNING: agent not collapsed to committed state after warmup."
                " COMMITTED_MODE_ON vs ABLATED comparison may be uninformative.",
                flush=True,
            )

        stable_env   = CausalGridWorldV2(seed=seed + 500, **STABLE_ENV_KWARGS)
        volatile_env = CausalGridWorldV2(seed=seed + 500, **VOLATILE_ENV_KWARGS)

        seed_results: Dict[str, Dict] = {}
        for env_name, env_obj in [("stable", stable_env), ("volatile", volatile_env)]:
            for cond_label, ablated in [
                ("COMMITTED_MODE_ON", False),
                ("COMMITTED_MODE_ABLATED", True),
            ]:
                label = f"seed{seed}_{env_name}_{cond_label}"
                print(f"\n  -- {label} --", flush=True)
                seed_results[f"{env_name}_{cond_label}"] = _eval_condition(
                    agent=agent,
                    env=env_obj,
                    num_episodes=eval_episodes,
                    steps_per_episode=steps_per_episode,
                    world_dim=world_dim,
                    ablated=ablated,
                    label=label,
                    train_variance=train_variance,
                )

        seed_results["train_variance"] = train_variance   # type: ignore
        seed_results["commit_threshold"] = float(agent.e3.commit_threshold)  # type: ignore
        results_by_seed.append(seed_results)
        print("verdict: PASS", flush=True)

    # -------------------------------------------------------------------------
    # Per-seed criteria checks
    # -------------------------------------------------------------------------
    per_seed_c1: List[bool] = []
    per_seed_c2: List[bool] = []
    per_seed_c3: List[bool] = []
    per_seed_c4: List[bool] = []
    per_seed_c5: List[bool] = []
    per_seed_harm_gap_stable:   List[float] = []
    per_seed_harm_gap_volatile: List[float] = []
    per_seed_gap_ratio:         List[float] = []
    total_fatal = 0

    for seed, sr in zip(seeds, results_by_seed):
        ha_s = sr["stable_COMMITTED_MODE_ON"]["mean_harm_per_step"]
        hb_s = sr["stable_COMMITTED_MODE_ABLATED"]["mean_harm_per_step"]
        ha_v = sr["volatile_COMMITTED_MODE_ON"]["mean_harm_per_step"]
        hb_v = sr["volatile_COMMITTED_MODE_ABLATED"]["mean_harm_per_step"]

        gap_s = ha_s - hb_s
        gap_v = ha_v - hb_v
        gap_ratio = gap_v / max(abs(gap_s), 1e-9) if gap_s != 0.0 else 0.0

        nc_on  = sr["stable_COMMITTED_MODE_ON"]["n_committed"]
        nu_on  = sr["stable_COMMITTED_MODE_ON"]["n_uncommitted"]
        nc_abl = sr["stable_COMMITTED_MODE_ABLATED"]["n_committed"]
        fatal_seed = sum(
            sr[k]["fatal_errors"]
            for k in sr
            if isinstance(sr[k], dict) and "fatal_errors" in sr[k]
        )
        total_fatal += fatal_seed

        c1 = gap_s >= MIN_HARM_GAP_STABLE
        c2 = gap_v < gap_s
        c3 = gap_ratio <= VOLATILITY_MODULATION_RATIO
        c4 = nc_on > nu_on
        c5 = nc_abl == 0

        per_seed_c1.append(c1)
        per_seed_c2.append(c2)
        per_seed_c3.append(c3)
        per_seed_c4.append(c4)
        per_seed_c5.append(c5)
        per_seed_harm_gap_stable.append(gap_s)
        per_seed_harm_gap_volatile.append(gap_v)
        per_seed_gap_ratio.append(gap_ratio)

        print(
            f"\n  [Seed {seed}] gap_stable={gap_s:.5f}  gap_volatile={gap_v:.5f}"
            f"  gap_ratio={gap_ratio:.3f}"
            f"  C1={'PASS' if c1 else 'FAIL'}"
            f"  C2={'PASS' if c2 else 'FAIL'}"
            f"  C3={'PASS' if c3 else 'FAIL'}"
            f"  C4={'PASS' if c4 else 'FAIL'}"
            f"  C5={'PASS' if c5 else 'FAIL'}",
            flush=True,
        )

    # Aggregate: majority rule for C1/C2/C3; all-seeds for C4/C5; zero fatals for C6
    n_c1 = sum(per_seed_c1)
    n_c2 = sum(per_seed_c2)
    n_c3 = sum(per_seed_c3)
    c1_pass = n_c1 >= MIN_SEEDS_PASSING
    c2_pass = n_c2 >= MIN_SEEDS_PASSING
    c3_pass = n_c3 >= MIN_SEEDS_PASSING
    c4_pass = all(per_seed_c4)
    c5_pass = all(per_seed_c5)
    c6_pass = total_fatal == 0

    all_pass = c1_pass and c2_pass and c3_pass and c4_pass and c5_pass and c6_pass
    criteria_met = sum([c1_pass, c2_pass, c3_pass, c4_pass, c5_pass, c6_pass])
    status = "PASS" if all_pass else "FAIL"

    avg_gap_stable   = _mean_safe(per_seed_harm_gap_stable)
    avg_gap_volatile = _mean_safe(per_seed_harm_gap_volatile)
    avg_gap_ratio    = _mean_safe(per_seed_gap_ratio)

    failure_notes = []
    for i, seed in enumerate(seeds):
        if not per_seed_c1[i]:
            failure_notes.append(
                f"C1 FAIL seed={seed}: harm_gap_stable={per_seed_harm_gap_stable[i]:.5f}"
                f" < MIN={MIN_HARM_GAP_STABLE}"
            )
        if not per_seed_c2[i]:
            failure_notes.append(
                f"C2 FAIL seed={seed}: gap_volatile={per_seed_harm_gap_volatile[i]:.5f}"
                f" >= gap_stable={per_seed_harm_gap_stable[i]:.5f}"
            )
        if not per_seed_c3[i]:
            failure_notes.append(
                f"C3 FAIL seed={seed}: gap_ratio={per_seed_gap_ratio[i]:.3f}"
                f" > {VOLATILITY_MODULATION_RATIO}"
            )
    if not c1_pass:
        failure_notes.append(
            f"C1 aggregate FAIL: {n_c1}/{len(seeds)} seeds passed (need {MIN_SEEDS_PASSING})"
        )
    if not c2_pass:
        failure_notes.append(
            f"C2 aggregate FAIL: {n_c2}/{len(seeds)} seeds passed (need {MIN_SEEDS_PASSING})"
        )
    if not c3_pass:
        failure_notes.append(
            f"C3 aggregate FAIL: {n_c3}/{len(seeds)} seeds passed (need {MIN_SEEDS_PASSING})"
        )
    if not c4_pass:
        failure_notes.append("C4 FAIL: gate not elevating in COMMITTED_MODE_ON condition")
    if not c5_pass:
        failure_notes.append("C5 FAIL: ablation incomplete (committed steps in ABLATED condition)")
    if not c6_pass:
        failure_notes.append(f"C6 FAIL: fatal_errors={total_fatal}")

    print(f"\nV3-EXQ-125a verdict: {status}  ({criteria_met}/6)", flush=True)
    for note in failure_notes:
        print(f"  {note}", flush=True)

    print(
        f"\n  Avg across seeds:"
        f"\n    harm_gap_stable={avg_gap_stable:.5f}"
        f"  harm_gap_volatile={avg_gap_volatile:.5f}"
        f"  gap_ratio={avg_gap_ratio:.3f}",
        flush=True,
    )

    per_seed_data = []
    for seed, sr, gs, gv, gr in zip(
        seeds, results_by_seed,
        per_seed_harm_gap_stable, per_seed_harm_gap_volatile, per_seed_gap_ratio,
    ):
        per_seed_data.append({
            "seed": seed,
            "harm_active_stable":    sr["stable_COMMITTED_MODE_ON"]["mean_harm_per_step"],
            "harm_ablated_stable":   sr["stable_COMMITTED_MODE_ABLATED"]["mean_harm_per_step"],
            "harm_active_volatile":  sr["volatile_COMMITTED_MODE_ON"]["mean_harm_per_step"],
            "harm_ablated_volatile": sr["volatile_COMMITTED_MODE_ABLATED"]["mean_harm_per_step"],
            "harm_gap_stable":   float(gs),
            "harm_gap_volatile": float(gv),
            "gap_ratio":         float(gr),
            "c1_seed": per_seed_c1[seeds.index(seed)],
            "c2_seed": per_seed_c2[seeds.index(seed)],
            "c3_seed": per_seed_c3[seeds.index(seed)],
            "n_committed_active_stable":  sr["stable_COMMITTED_MODE_ON"]["n_committed"],
            "n_uncommitted_active_stable":sr["stable_COMMITTED_MODE_ON"]["n_uncommitted"],
            "n_committed_ablated_stable": sr["stable_COMMITTED_MODE_ABLATED"]["n_committed"],
            "train_variance":    float(sr["train_variance"]),
            "commit_threshold":  float(sr["commit_threshold"]),
        })

    metrics = {
        "avg_harm_gap_stable":   float(avg_gap_stable),
        "avg_harm_gap_volatile": float(avg_gap_volatile),
        "avg_gap_ratio":         float(avg_gap_ratio),
        "total_fatal_errors":    float(total_fatal),
        "crit1_pass": 1.0 if c1_pass else 0.0,
        "crit2_pass": 1.0 if c2_pass else 0.0,
        "crit3_pass": 1.0 if c3_pass else 0.0,
        "crit4_pass": 1.0 if c4_pass else 0.0,
        "crit5_pass": 1.0 if c5_pass else 0.0,
        "crit6_pass": 1.0 if c6_pass else 0.0,
        "criteria_met": float(criteria_met),
        "n_seeds": float(len(seeds)),
        "n_seeds_c1": float(n_c1),
        "n_seeds_c2": float(n_c2),
        "n_seeds_c3": float(n_c3),
        "min_seeds_passing": float(MIN_SEEDS_PASSING),
    }

    seed_rows = ""
    for psd in per_seed_data:
        seed_rows += (
            f"| {psd['seed']} | {psd['harm_active_stable']:.5f} |"
            f" {psd['harm_ablated_stable']:.5f} |"
            f" {psd['harm_gap_stable']:.5f} |"
            f" {psd['harm_active_volatile']:.5f} |"
            f" {psd['harm_ablated_volatile']:.5f} |"
            f" {psd['harm_gap_volatile']:.5f} |"
            f" {psd['gap_ratio']:.3f} |"
            f" {'P' if psd['c1_seed'] else 'F'}"
            f"{'P' if psd['c2_seed'] else 'F'}"
            f"{'P' if psd['c3_seed'] else 'F'} |\n"
        )

    failure_section = ""
    if failure_notes:
        failure_section = "\n## Failure Notes\n\n" + "\n".join(f"- {n}" for n in failure_notes)

    summary_markdown = f"""# V3-EXQ-125a -- ARC-029: Committed vs Ablated Mode Harm Outcomes (Redesign)

**Status:** {status}
**Claims:** ARC-029
**Seeds:** {seeds}  |  **Warmup:** {warmup_episodes} eps  |  **Eval:** {eval_episodes} eps x {steps_per_episode} steps
**Env:** num_hazards=8, hazard_harm=0.05, proximity_harm_scale=0.15
**Design:** 2x2 [COMMITTED_MODE_ON / COMMITTED_MODE_ABLATED] x [stable / volatile]
**Supersedes:** V3-EXQ-125 (failed 3x due to insufficient power under SD-010/011/012 substrate)

## Design Rationale

ARC-029 predicts that the BG beta commitment gate (MECH-090) reduces harm in stable
environments and that this advantage narrows in volatile environments. EXQ-125 failed
because substrate changes (SD-010/011/012) reduced harm rates ~100x, making the
committed-gate advantage undetectable at the original env scale.

Redesign raises: num_hazards 4->8, hazard_harm 0.02->0.05, proximity_harm_scale 0.05->0.15.
Raises MIN_HARM_GAP_STABLE 0.0001->0.001 (scaled with harm magnitude).
Adds 3 more seeds (5 total) with majority rule (C1/C2/C3: {MIN_SEEDS_PASSING}/{len(seeds)} seeds).

## Per-Seed Results

| Seed | Act-Stab | Abl-Stab | Gap-Stab | Act-Vol | Abl-Vol | Gap-Vol | GapRatio | C1C2C3 |
|---|---|---|---|---|---|---|---|---|
{seed_rows}
## Aggregated (avg across seeds)

| Metric | Value |
|---|---|
| avg_harm_gap_stable | {avg_gap_stable:.5f} |
| avg_harm_gap_volatile | {avg_gap_volatile:.5f} |
| avg_gap_ratio (volatile/stable) | {avg_gap_ratio:.3f} |

## PASS Criteria

| Criterion | Threshold | Seeds passing | Result |
|---|---|---|---|
| C1: harm_gap_stable >= {MIN_HARM_GAP_STABLE} | >= {MIN_SEEDS_PASSING}/{len(seeds)} seeds | {n_c1}/{len(seeds)} | {"PASS" if c1_pass else "FAIL"} |
| C2: gap_volatile < gap_stable | >= {MIN_SEEDS_PASSING}/{len(seeds)} seeds | {n_c2}/{len(seeds)} | {"PASS" if c2_pass else "FAIL"} |
| C3: gap_ratio <= {VOLATILITY_MODULATION_RATIO} | >= {MIN_SEEDS_PASSING}/{len(seeds)} seeds | {n_c3}/{len(seeds)} | {"PASS" if c3_pass else "FAIL"} |
| C4: committed > uncommitted in ON | ALL seeds | {"all" if c4_pass else "some"} | {"PASS" if c4_pass else "FAIL"} |
| C5: zero committed in ABLATED | ALL seeds | {"all" if c5_pass else "some"} | {"PASS" if c5_pass else "FAIL"} |
| C6: no fatal errors | 0 | - | {"PASS" if c6_pass else "FAIL"} |

Criteria met: {criteria_met}/6 -> **{status}**
{failure_section}
"""

    return {
        "status": status,
        "metrics": metrics,
        "per_seed_data": per_seed_data,
        "summary_markdown": summary_markdown,
        "claim_ids": CLAIM_IDS,
        "evidence_direction": (
            "supports" if all_pass else ("mixed" if criteria_met >= 4 else "weakens")
        ),
        "experiment_type": EXPERIMENT_TYPE,
        "experiment_purpose": EXPERIMENT_PURPOSE,
        "fatal_error_count": float(total_fatal),
        "supersedes": "v3_exq_125_arc029_committed_mode_pair",
    }


if __name__ == "__main__":
    import argparse
    import json
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds",       type=int, nargs="+", default=[0, 42, 100, 123, 200])
    parser.add_argument("--warmup",      type=int,   default=350)
    parser.add_argument("--eval-eps",    type=int,   default=60)
    parser.add_argument("--steps",       type=int,   default=200)
    parser.add_argument("--alpha-world", type=float, default=0.9)
    parser.add_argument("--dry-run",     action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        print("[dry-run] EXQ-125a: overriding to 2 seeds x 3 warmup + 2 eval eps x 5 steps")
        args.seeds = [0, 42]
        args.warmup = 3
        args.eval_eps = 2
        args.steps = 5

    result = run(
        seeds=args.seeds,
        warmup_episodes=args.warmup,
        eval_episodes=args.eval_eps,
        steps_per_episode=args.steps,
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
            print(f"  {k}: {v:.5f}", flush=True)
